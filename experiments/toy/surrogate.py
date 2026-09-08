"""Toy surrogate: a K=4 MLP committee mapping Lorenz state -> vector field.

Mirrors the MD committee (src/pyraimd2/surrogate/committee.py) semantics:

- the 3 field components play the role of "forces on 3 atoms": the honest
  per-component spread is sigma_i = population std across members of
  pred_ik (the scalar-force case of sigma_i = sqrt(mean_k |F_ik - F̄_i|²)),
  and the scalar committee score is the conservative s = max_i sigma_i;
  the shadow error is e = max_i |mean_pred_i - true_i|;
- "foundation prior" (MACE-MP-0 analog): every member is pre-trained on
  2000 oracle samples from the rho=28 attractor BEFORE the run; the run
  itself only ever fine-tunes on online labels;
- fine-tune mechanics: per-member seeded bootstrap resample of the labels
  (``[seed, k]``), Adam lr=1e-3, seeded epoch shuffle, minibatch of 8;
  bounded forgetting — each fine-tune re-initializes member k from its
  post-pretrain snapshot plus the same seeded scale-aware perturbation the
  MD committee applies to its readout (same seed per member every time, so
  a fine-tune is deterministic given the label set).

Documented deviations from the MD letter (both inherent to the toy):
no energy term in the loss (pure vector-field MSE — there is no energy in
Lorenz-63), and states/fields are standardized with the pretrain-set
statistics (fixed constants) because raw Lorenz scales (z ~ 25, |dx| up to
~200) make a lr=1e-3 tanh MLP ill-conditioned.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable

import numpy as np

from pyraimd2.surrogate.base import TrainReport

# Configurations per optimizer step, as in CommitteeSurrogate (MINIBATCH_CONFIGS).
MINIBATCH_CONFIGS = 8


class ToyCommittee:
    """K small MLPs (2 hidden layers of 64, tanh): state (x,y,z) -> (dx,dy,dz)."""

    def __init__(
        self,
        n_members: int = 4,
        hidden: int = 64,
        seed: int = 0,
        perturbation: float = 0.01,
        epochs: int = 30,
        lr: float = 1e-3,
    ) -> None:
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if perturbation < 0.0:
            raise ValueError(f"perturbation must be >= 0, got {perturbation}")
        self.n_members = n_members
        self.hidden = hidden
        self.seed = seed
        self.perturbation = perturbation
        self.epochs = epochs
        self.lr = lr
        import torch  # local import: torch is heavy (repo idiom)

        self._torch = torch
        self._members: list = []
        for k in range(n_members):
            # Different seeds per member: the only diversity source at init.
            with torch.random.fork_rng():
                torch.manual_seed((seed + 1) * 1_000_003 + k)
                self._members.append(self._build_mlp(hidden))
        # Standardization constants, fitted by pretrain(); fixed afterwards.
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._prior: list[dict] | None = None  # post-pretrain snapshots

    def __deepcopy__(self, memo: dict):
        """Deep-copiable despite holding the torch module (a singleton —
        shared, not copied).  Lets the sweep stamp independent committees
        from one pre-trained prior."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            setattr(new, name, value if name == "_torch" else copy.deepcopy(value, memo))
        return new

    def _build_mlp(self, hidden: int):
        torch = self._torch
        return torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 3),
        )

    # -- normalization ------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._prior is None:
            raise RuntimeError("ToyCommittee is not pre-trained yet — call pretrain() first")

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self._x_mean) / self._x_std

    def _denorm_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._y_std + self._y_mean

    # -- prediction ---------------------------------------------------------

    def predict_members(self, state: np.ndarray) -> np.ndarray:
        """The (K, 3) raw member predictions at ``state`` (denormalized)."""
        self._require_fitted()
        torch = self._torch
        x = torch.as_tensor(self._norm_x(np.asarray(state, dtype=float)), dtype=torch.float32)
        with torch.no_grad():
            preds = torch.stack([member(x) for member in self._members])
        return self._denorm_y(preds.numpy())

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Committee mean field and spread s = max_i std_k(pred_ik)."""
        preds = self.predict_members(state)
        sigma = preds.std(axis=0, ddof=0)  # population std across members
        return preds.mean(axis=0), float(sigma.max())

    # -- training -------------------------------------------------------------

    def _reset_member(self, k: int) -> None:
        """Member k := post-pretrain snapshot + seeded scale-aware perturbation
        (bounded forgetting; mirrors CommitteeSurrogate._reset_readout — same
        seed on every fine-tune, so the reset is deterministic)."""
        torch = self._torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 1) * 1_000_003 + k)
        member = self._members[k]
        with torch.no_grad():
            for name, param in member.named_parameters():
                base = self._prior[k][name]
                scale = self.perturbation * float(base.std())
                noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
                param.copy_(base + scale * noise)

    def _train_member(self, k: int, x: np.ndarray, y: np.ndarray, epochs: int) -> tuple[float, float]:
        """Adam on member k's seeded bootstrap resample; returns (initial, final) loss."""
        torch = self._torch
        n = x.shape[0]
        rng = np.random.default_rng([self.seed, k])
        bootstrap = rng.integers(0, n, size=n)  # with replacement, as in MD
        xb = torch.as_tensor(x[bootstrap], dtype=torch.float32)
        yb = torch.as_tensor(y[bootstrap], dtype=torch.float32)
        member = self._members[k]
        optimizer = torch.optim.Adam(member.parameters(), lr=self.lr)
        shuffle = np.random.default_rng([self.seed, k, 1])  # epoch order stream (MD idiom)
        initial_loss = final_loss = float("nan")
        for epoch in range(epochs):
            order = shuffle.permutation(n)
            for start in range(0, n, MINIBATCH_CONFIGS):
                idx = order[start : start + MINIBATCH_CONFIGS]
                loss = torch.nn.functional.mse_loss(member(xb[idx]), yb[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == 0 and start == 0:
                    initial_loss = float(loss.detach())  # pre-update
                if epoch == epochs - 1 and start + MINIBATCH_CONFIGS >= n:
                    final_loss = float(loss.detach())  # last optimizer step
        return initial_loss, final_loss

    def pretrain(self, states: np.ndarray, fields: np.ndarray, epochs: int = 200) -> TrainReport:
        """The foundation prior: fit all members on oracle samples from the
        rho=28 attractor (per-member bootstrap keeps the spread nonzero).
        Idempotent: re-fits normalization and re-snapshots the prior."""
        torch = self._torch  # noqa: F841 (keeps the local-import idiom visible)
        states = np.asarray(states, dtype=float).reshape(-1, 3)
        fields = np.asarray(fields, dtype=float).reshape(-1, 3)
        if states.shape[0] != fields.shape[0] or states.shape[0] < 8:
            raise ValueError(f"pretrain needs >= 8 matched pairs, got {states.shape[0]}")
        self._x_mean = states.mean(axis=0)
        self._x_std = states.std(axis=0)
        self._y_mean = fields.mean(axis=0)
        self._y_std = fields.std(axis=0)
        x = self._norm_x(states)
        y = (fields - self._y_mean) / self._y_std

        t0 = time.perf_counter()
        initial_losses, final_losses = [], []
        for k in range(self.n_members):
            init_l, final_l = self._train_member(k, x, y, epochs)
            initial_losses.append(init_l)
            final_losses.append(final_l)
            print(f"pretrain member={k} loss {init_l:.6f} -> {final_l:.6f}", flush=True)
        # Snapshot the prior: the fixed starting point of every fine-tune.
        self._prior = [
            {name: p.detach().clone() for name, p in member.named_parameters()}
            for member in self._members
        ]
        return TrainReport(
            n_labels=int(states.shape[0]),
            n_epochs=epochs,
            initial_loss=float(np.mean(initial_losses)),
            final_loss=float(np.mean(final_losses)),
            member_losses=tuple(final_losses),
            wall_time_s=time.perf_counter() - t0,
        )

    def finetune(
        self, labels: Iterable[tuple[np.ndarray, np.ndarray]], epochs: int | None = None
    ) -> TrainReport:
        """Fine-tune on (state, true field) label pairs; MD semantics:
        reset-to-prior + seeded bootstrap, Adam lr=1e-3, ``epochs`` epochs
        (default ``self.epochs`` = 30).  Deterministic given the labels."""
        self._require_fitted()
        label_list = [(np.asarray(s, dtype=float), np.asarray(f, dtype=float)) for s, f in labels]
        if not label_list:
            raise ValueError("finetune needs at least one (state, field) pair")
        states = np.stack([s for s, _ in label_list])
        fields = np.stack([f for _, f in label_list])
        x = self._norm_x(states)
        y = (fields - self._y_mean) / self._y_std
        n_epochs = self.epochs if epochs is None else epochs

        t0 = time.perf_counter()
        initial_losses, final_losses = [], []
        for k in range(self.n_members):
            self._reset_member(k)
            init_l, final_l = self._train_member(k, x, y, n_epochs)
            initial_losses.append(init_l)
            final_losses.append(final_l)
        return TrainReport(
            n_labels=len(label_list),
            n_epochs=n_epochs,
            initial_loss=float(np.mean(initial_losses)),
            final_loss=float(np.mean(final_losses)),
            member_losses=tuple(final_losses),
            wall_time_s=time.perf_counter() - t0,
        )

    # -- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict:
        """Full committee state (members + normalization + prior), for
        restart-safe campaigns — mirrors CommitteeSurrogate.state_dict."""
        self._require_fitted()
        return {
            "n_members": self.n_members,
            "hidden": self.hidden,
            "seed": self.seed,
            "perturbation": self.perturbation,
            "epochs": self.epochs,
            "lr": self.lr,
            "norm": [self._x_mean, self._x_std, self._y_mean, self._y_std],
            "member_state_dicts": [copy.deepcopy(m.state_dict()) for m in self._members],
            "prior": [[(n, t.clone()) for n, t in prior.items()] for prior in self._prior],
        }
