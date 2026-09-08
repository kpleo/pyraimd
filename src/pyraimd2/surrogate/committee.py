"""Committee surrogate with trainable readout heads on frozen MACE backbones.

Each member copies a foundation model and trains only parameters selected by
``trainable_filters``. Members can share a backbone or use compatible backbones
with the same data interface, units and element mapping. Readouts start from
seeded perturbations and are reset before each bootstrap fit.

The prediction is the member mean. Per-atom uncertainty is the population RMS
of force-vector deviations from that mean, rather than the standard deviation
of their norms. The scalar switching score is the largest per-atom spread.

Fine-tuning uses Adam and shuffled minibatches of eight configurations. The
loss combines squared energy error per atom with weighted mean squared force
error. A constant energy offset is fitted for each member to align the reference
energy conventions. It is applied to energies during training and prediction;
forces and their gradients are unchanged. Results are deterministic for fixed
seeds and labels, subject to the numerical backend.

This adapter returns energies and forces; committee stress is unavailable.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from ase import Atoms

from pyraimd2.engines.base import EngineResult
from pyraimd2.surrogate.base import SurrogatePrediction, TrainReport

# Configurations per optimizer step within one epoch (see module docstring).
MINIBATCH_CONFIGS = 8


class CommitteeSurrogate:
    """K-member readout-head committee on a frozen MACE foundation model."""

    def __init__(
        self,
        model: str | Sequence[str] = "small",
        device: str = "cpu",
        default_dtype: str = "float64",
        n_members: int = 4,
        seed: int = 0,
        perturbation: float = 0.01,
        epochs: int = 50,
        lr: float = 1e-3,
        force_weight: float = 10.0,
        trainable_filters: tuple[str, ...] = ("readout",),
    ) -> None:
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if perturbation < 0.0:
            raise ValueError(f"perturbation must be >= 0, got {perturbation}")
        if not trainable_filters or not all(trainable_filters):
            raise ValueError(
                f"trainable_filters must be non-empty substrings, got {trainable_filters}"
            )
        # 0.4 supports the CPU committee data path only: accepting a ``device``
        # field must not silently imply GPU availability. The single-model GPU
        # path is MaceSurrogate, declared and tested separately.
        if device != "cpu":
            raise ValueError(
                f"CommitteeSurrogate supports device='cpu' only, got {device!r}; "
                "use MaceSurrogate for a single-model GPU path"
            )
        # Use one backbone for all members or supply compatible backbones.
        if isinstance(model, (str, os.PathLike)):
            self._model_specs = [str(model)] * n_members
        else:
            self._model_specs = [str(m) for m in model]
            if not self._model_specs:
                raise ValueError("model list must be non-empty")
            n_members = len(self._model_specs)  # explicit list defines K
        self.model = model
        self.device = device
        self.default_dtype = default_dtype
        self.n_members = n_members
        self.seed = seed
        self.perturbation = perturbation
        self.epochs = epochs
        self.lr = lr
        self.force_weight = force_weight
        # Parameter-name substrings select tensors to train.
        self.trainable_filters = tuple(trainable_filters)
        self._calc: Any = None  # batch-building machinery + unit conversions
        self._models: list[Any] = []  # the K members (built lazily with _calc)
        self._foundation_readouts: list[dict[str, Any]] = []  # per-member heads
        self._energy_shifts: list[float] = []  # per-member energy reference

    # -- loading ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._calc is not None:
            return
        from mace.calculators import mace_mp  # local import: torch is heavy

        # Load each distinct backbone once; member k deep-copies its spec's
        # foundation.  The data-building interface (z_table, head, r_max,
        # keys, unit conversions) must be identical across backbones —
        # member graphs are built once via the first backbone's calculator.
        calcs: dict[str, Any] = {}
        for spec in self._model_specs:
            if spec not in calcs:
                calcs[spec] = mace_mp(
                    model=spec, device=self.device, default_dtype=self.default_dtype
                )
        self._calc = calcs[self._model_specs[0]]
        for spec, calc in calcs.items():
            if calc is self._calc:
                continue
            if (
                calc.r_max != self._calc.r_max
                or list(calc.z_table.zs) != list(self._calc.z_table.zs)
                or getattr(calc, "head", None) != getattr(self._calc, "head", None)
                or calc.charges_key != self._calc.charges_key
                or calc.info_keys != self._calc.info_keys
                or calc.energy_units_to_eV != self._calc.energy_units_to_eV
                or calc.length_units_to_A != self._calc.length_units_to_A
            ):
                raise RuntimeError(
                    f"backbone {spec!r} has an incompatible data interface "
                    "(z_table/head/r_max/keys/units) — mixed committees need "
                    "matching interfaces"
                )
        self._models = [copy.deepcopy(calcs[spec].models[0]) for spec in self._model_specs]
        for member in self._models:
            n_trainable = 0
            for name, param in member.named_parameters():
                param.requires_grad_(any(f in name for f in self.trainable_filters))
                n_trainable += int(param.requires_grad)
            if n_trainable == 0:
                raise RuntimeError(
                    f"no parameters matching {self.trainable_filters!r} found "
                    "in the MACE model — refusing to fine-tune an empty parameter set"
                )
        # Per-member snapshots of the foundation heads: the fixed starting
        # points of every fine-tune (bounded forgetting, deterministic
        # restarts).  Members of one backbone share the same snapshot; a
        # mixed backbone gets its own.
        self._foundation_readouts = [
            {
                name: param.detach().clone()
                for name, param in member.named_parameters()
                if param.requires_grad
            }
            for member in self._models
        ]
        # Diversify members 1..K-1 at load time.  Without this, same-backbone
        # members are exact copies until the first fine-tune, so sigma == 0
        # and the conformal ratio r = e/(s+delta) explodes to e/delta — one
        # cold-start window of those ratios pins qhat at its max for a full
        # window (observed in smoke 7615191: qhat = 1000 x e_max).  Member 0
        # stays the clean foundation reference.
        for k in range(1, self.n_members):
            self._reset_readout(self._models[k], k)
        self._energy_shifts = [0.0] * self.n_members  # fitted on first finetune

    def _to_atomic_data(self, atoms: Atoms) -> Any:
        """Build a MACE ``AtomicData`` exactly as the MACECalculator does."""
        from mace import data as mace_data  # local import: torch is heavy
        from mace.tools import torch_tools

        keyspec = mace_data.KeySpecification(
            info_keys=self._calc.info_keys,
            arrays_keys={self._calc.charges_key: "charges"},
        )
        with torch_tools.default_dtype(self.default_dtype):
            config = mace_data.config_from_atoms(
                atoms, key_specification=keyspec, head_name=self._calc.head
            )
            return mace_data.AtomicData.from_config(
                config,
                z_table=self._calc.z_table,
                cutoff=self._calc.r_max,
                heads=self._calc.available_heads,
            )

    @staticmethod
    def _forward(member: Any, batch: Any, training: bool) -> tuple[dict, dict]:
        """Run one member on a batched graph.

        Returns ``(out, batch_dict)``; energies/forces in ``out`` are in model
        units and differentiable iff ``training`` (forces by autograd through
        positions, as MACE does internally).
        """
        batch_dict = batch.clone().to_dict()
        out = member(
            batch_dict,
            training=training,
            compute_stress=False,
            compute_virials=False,
            compute_force=True,
        )
        return out, batch_dict

    # -- Surrogate protocol ----------------------------------------------

    def predict(self, atoms: Atoms) -> SurrogatePrediction:
        self._ensure_loaded()
        import torch
        from mace.tools import torch_geometric

        batch = torch_geometric.Batch.from_data_list([self._to_atomic_data(atoms)])
        e_conv = self._calc.energy_units_to_eV
        f_conv = e_conv / self._calc.length_units_to_A
        energies, forces = [], []
        for k, member in enumerate(self._models):
            out, _ = self._forward(member, batch, training=False)
            energies.append(out["energy"].detach().reshape(1) + self._energy_shifts[k])
            forces.append(out["forces"].detach())
        energy_k = torch.stack(energies)  # (K, 1)
        force_k = torch.stack(forces)  # (K, N, 3)
        mean_force = force_k.mean(dim=0)  # (N, 3)
        # Per-atom spread: population RMS of the deviation vectors,
        # sigma_i = sqrt(mean_k |F_{i,k} - F̄_i|^2) — the standard committee-UQ
        # estimator (summed per-component variance).  Taking std_k of the
        # deviation *norms* instead is wrong: for K=2 the norms from the mean
        # are equal by construction so sigma == 0 identically, and for larger
        # K it reports only the asymmetry of the norms, badly understating
        # the spread.
        deviation_sq = (force_k - mean_force).norm(dim=2) ** 2  # (K, N)
        sigma = deviation_sq.mean(dim=0).sqrt()  # (N,)
        return SurrogatePrediction(
            energy=float(energy_k.mean()) * e_conv,
            forces=mean_force.numpy() * f_conv,
            stress=None,  # committee stress is not implemented
            uncertainty=sigma.numpy() * f_conv,
        )

    # -- fine-tuning ------------------------------------------------------

    def _reset_readout(self, member: Any, member_index: int) -> None:
        """Re-initialize member k's readout: foundation head + seeded noise
        (scale-aware: noise std = ``perturbation`` × the tensor's own std, so
        the kick is small relative to each weight's magnitude)."""
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 1) * 1_000_003 + member_index)
        with torch.no_grad():
            for name, param in member.named_parameters():
                if not param.requires_grad:
                    continue
                base = self._foundation_readouts[member_index][name]
                scale = self.perturbation * float(base.std())
                noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
                param.copy_(base + scale * noise)

    def finetune(self, labels: Iterable[tuple[Atoms, EngineResult]]) -> TrainReport:
        import torch
        from mace.tools import torch_geometric
        from mace.tools.scatter import scatter_mean

        self._ensure_loaded()
        label_list = list(labels)
        if not label_list:
            raise ValueError("finetune needs at least one (atoms, label) pair")

        t0 = time.perf_counter()
        dtype = next(self._models[0].parameters()).dtype
        e_conv = self._calc.energy_units_to_eV
        f_conv = e_conv / self._calc.length_units_to_A
        data_list = [self._to_atomic_data(atoms) for atoms, _ in label_list]
        # Targets in model units, per configuration.
        energy_true = [
            torch.tensor(result.energy / e_conv, dtype=dtype) for _, result in label_list
        ]
        force_true = [
            torch.as_tensor(np.asarray(result.forces) / f_conv, dtype=dtype)
            for _, result in label_list
        ]

        n_labels = len(label_list)
        initial_losses: list[float] = []
        final_losses: list[float] = []
        for k, member in enumerate(self._models):
            self._reset_readout(member, k)
            trainable = [p for p in member.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=self.lr)
            rng = np.random.default_rng([self.seed, k])
            bootstrap = rng.integers(0, n_labels, size=n_labels)  # with replacement
            data_b = [data_list[i] for i in bootstrap]
            energy_b = [energy_true[i] for i in bootstrap]
            force_b = [force_true[i] for i in bootstrap]  # bootstrap order
            # Energy referencing: least-squares constant offset for this member
            # (see module docstring).  A constant changes no forces/gradients.
            # Forward the bootstrap sample in minibatches — per-config
            # energies are independent, so this equals one full-batch forward
            # without materializing it (a single 52x474-atom batch needs a
            # ~29 GB TP intermediate and OOMed stage-3 jobs 7616839/7616847/
            # 7616868; MINIBATCH_CONFIGS chunks stay ~4.5 GB).
            e_ref_chunks: list[Any] = []
            for start in range(0, n_labels, MINIBATCH_CONFIGS):
                mb = torch_geometric.Batch.from_data_list(
                    data_b[start : start + MINIBATCH_CONFIGS]
                )
                out_c, _ = self._forward(member, mb, training=False)
                e_ref_chunks.append(out_c["energy"].detach())
            energy_all = torch.stack(energy_b)
            self._energy_shifts[k] = float(
                (energy_all - torch.cat(e_ref_chunks)).mean()
            )

            shuffle = np.random.default_rng([self.seed, k, 1])  # epoch order stream
            for epoch in range(self.epochs):
                order = shuffle.permutation(n_labels)
                epoch_losses: list[float] = []
                for start in range(0, n_labels, MINIBATCH_CONFIGS):
                    idx = order[start : start + MINIBATCH_CONFIGS]
                    batch = torch_geometric.Batch.from_data_list([data_b[i] for i in idx])
                    energy_mb = torch.stack([energy_b[i] for i in idx])
                    force_mb = torch.cat([force_b[i] for i in idx])
                    node_counts = (batch["ptr"][1:] - batch["ptr"][:-1]).to(dtype)
                    out, batch_dict = self._forward(member, batch, training=True)
                    energy, forces = out["energy"], out["forces"]
                    # Loss per config g: (ΔE_g)²/N_g + w·MSE_g(F); mean over
                    # the minibatch's configs.
                    residual = energy + self._energy_shifts[k] - energy_mb
                    loss_energy = ((residual**2) / node_counts).mean()
                    sq_node = ((forces - force_mb) ** 2).mean(dim=1)
                    loss_forces = scatter_mean(
                        sq_node, batch_dict["batch"], dim=0, dim_size=energy.shape[0]
                    ).mean()
                    loss = loss_energy + self.force_weight * loss_forces
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach()))
                    if epoch == 0 and start == 0:
                        initial_losses.append(float(loss.detach()))  # pre-update
                    if epoch == self.epochs - 1 and start + MINIBATCH_CONFIGS >= n_labels:
                        final_losses.append(float(loss.detach()))  # last step
                # Progress line per (member, epoch): fine-tunes are the wall-time
                # dominator — this is what separates "slow" from "hung" on the
                # cluster and enables rational epoch-budget choices.
                print(
                    f"finetune member={k} epoch={epoch}/{self.epochs} "
                    f"loss={np.mean(epoch_losses):.6f}",
                    flush=True,
                )

        return TrainReport(
            n_labels=n_labels,
            n_epochs=self.epochs,
            initial_loss=float(np.mean(initial_losses)),
            final_loss=float(np.mean(final_losses)),
            member_losses=tuple(final_losses),
            wall_time_s=time.perf_counter() - t0,
        )

    # -- checkpointing ------------------------------------------------------

    def state_dict(self) -> dict:
        """Full committee state for restart-safe campaigns: member weights,
        energy shifts, and the recipe echo for identity validation."""
        self._ensure_loaded()
        return {
            "model_specs": list(self._model_specs),
            "n_members": self.n_members,
            "seed": self.seed,
            "perturbation": self.perturbation,
            "epochs": self.epochs,
            "lr": self.lr,
            "force_weight": self.force_weight,
            "trainable_filters": self.trainable_filters,
            "member_state_dicts": [
                {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for m in self._models
            ],
            "energy_shifts": list(self._energy_shifts),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a :meth:`state_dict` snapshot.  Raises on any recipe or
        backbone mismatch — a resumed run must never silently continue with
        a different surrogate than the one that produced the stored labels.

        Every check runs before any model is loaded or modified: a rejected
        snapshot leaves this instance untouched.
        """
        problems: list[str] = []
        if list(state.get("model_specs", [])) != list(self._model_specs):
            problems.append(
                f"backbones {state.get('model_specs')!r} != {self._model_specs!r}"
            )
        if int(state.get("n_members", -1)) != self.n_members:
            problems.append(
                f"n_members {state.get('n_members')} != {self.n_members}"
            )
        recipe_keys = ("seed", "perturbation", "epochs", "lr", "force_weight")
        for key in recipe_keys:
            if key not in state or state[key] != getattr(self, key):
                problems.append(f"recipe {key} {state.get(key)!r} != {getattr(self, key)!r}")
        if "trainable_filters" not in state or tuple(state["trainable_filters"]) != self.trainable_filters:
            problems.append(
                f"trainable_filters {state.get('trainable_filters')!r} "
                f"!= {self.trainable_filters!r}"
            )
        member_states = state.get("member_state_dicts", [])
        if len(member_states) != self.n_members:
            problems.append(
                f"member_state_dicts has {len(member_states)} entries, "
                f"expected {self.n_members}"
            )
        energy_shifts = state.get("energy_shifts", [])
        if len(energy_shifts) != self.n_members:
            problems.append(
                f"energy_shifts has {len(energy_shifts)} entries, "
                f"expected {self.n_members}"
            )
        if problems:
            raise ValueError("invalid committee checkpoint: " + "; ".join(problems))
        self._ensure_loaded()
        for member, member_state in zip(self._models, member_states):
            member.load_state_dict(member_state)
        self._energy_shifts = list(energy_shifts)
