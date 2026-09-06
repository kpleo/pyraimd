"""Runner: drives ASE VelocityVerlet with the SwitchingCalculator (§4.1).

Restart state: the Store holds, per logged
step, the atoms snapshot at force-evaluation time (positions and velocity
Verlet *half-step* momenta) plus the forces that drove the MD.
:meth:`Runner.resume` reconstructs the on-step momenta exactly as the
integrator computed them (``p += 0.5 * dt * forces``) and primes the ASE
result cache with the stored label, so the resumed run performs no
re-assessment, writes no duplicate rows, and continues bit-for-bit.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2.engines.base import Engine
from pyraimd2.loop.switching_calculator import SwitchingCalculator
from pyraimd2.store.store import Store
from pyraimd2.surrogate.base import Surrogate
from pyraimd2.switch.base import LabelObservation, Switch


@dataclass(frozen=True)
class RunSummary:
    """Outcome of one :meth:`Runner.run` segment.

    Attributes:
        n_steps: MD steps performed in this segment.
        n_dft: Engine evaluations in this segment.
        dft_fraction: n_dft / (all force evaluations in this segment).  A
            fresh run evaluates the initial geometry once (logged as step -1),
            so evaluations = n_steps + 1 for the first segment.
        force_mae_ev_a: Mean per-atom |F_surrogate - F_engine| over the dft
            steps of this segment (NaN if there were none).
        force_max_ev_a: Max per-atom |F_surrogate - F_engine| likewise.
        wall_time_s: Wall-clock seconds of this segment.
    """

    n_steps: int
    n_dft: int
    dft_fraction: float
    force_mae_ev_a: float
    force_max_ev_a: float
    wall_time_s: float


class Runner:
    """Owns the integrator; knows nothing about model internals."""

    def __init__(
        self,
        atoms: Atoms,
        surrogate: Surrogate,
        engine: Engine,
        switch: Switch,
        store: Store,
        run_id: str,
        timestep_fs: float = 0.5,
        temperature_K: float = 300.0,
        on_label: Callable[[LabelObservation], None] | None = None,
        explore_frac: float = 0.0,
        explore_rng: np.random.Generator | None = None,
        explore_seed: int | None = None,
    ) -> None:
        # Momenta are thermalized exactly once here; a resume restores them
        # from the Store, so thermalization is skipped when they already exist.
        if "momenta" not in atoms.arrays:
            thermalize_momenta(atoms, temperature_K)
        else:
            # Input-carried momenta take precedence over --temperature;
            # say so loudly (a silent skip once turned a "700 K control"
            # into a replica of the 300 K one).
            t_inst = atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB)
            print(f"runner: input momenta found, thermalization skipped "
                  f"(T_inst={t_inst:.1f} K; --temperature {temperature_K} "
                  f"not applied)", flush=True)
        self.atoms = atoms
        self.timestep_fs = timestep_fs
        self.calc = SwitchingCalculator(
            surrogate, engine, switch, store, run_id, on_label=on_label,
            explore_frac=explore_frac, explore_rng=explore_rng,
            explore_seed=explore_seed,
        )
        atoms.calc = self.calc
        self.dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    def run(self, n_steps: int) -> RunSummary:
        """Run ``n_steps`` MD steps; raise EngineError on engine failure."""
        n_dft_before = self.calc.n_dft
        n_ml_before = self.calc.n_ml
        n_err_before = len(self.calc.shadow_force_errors)

        t0 = time.perf_counter()
        self.dyn.run(n_steps)
        wall_time_s = time.perf_counter() - t0

        n_dft = self.calc.n_dft - n_dft_before
        n_ml = self.calc.n_ml - n_ml_before
        n_evaluations = n_dft + n_ml
        errors = self.calc.shadow_force_errors[n_err_before:]
        flat = np.concatenate(errors) if errors else np.empty(0)
        return RunSummary(
            n_steps=n_steps,
            n_dft=n_dft,
            dft_fraction=n_dft / n_evaluations if n_evaluations else float("nan"),
            force_mae_ev_a=float(flat.mean()) if flat.size else float("nan"),
            force_max_ev_a=float(flat.max()) if flat.size else float("nan"),
            wall_time_s=wall_time_s,
        )

    @classmethod
    def resume(
        cls,
        store: Store,
        run_id: str,
        surrogate: Surrogate,
        engine: Engine,
        switch: Switch,
        timestep_fs: float = 0.5,
        on_label: Callable[[LabelObservation], None] | None = None,
        explore_frac: float = 0.0,
        explore_rng: np.random.Generator | None = None,
        explore_seed: int | None = None,
    ) -> Runner:
        """Resume ``run_id`` from the Store so the run continues bit-for-bit.

        Restores positions, reconstructs on-step momenta from the stored
        half-step momenta and the stored driving forces, and re-primes the
        calculator cache and step counter — engine and switch are not
        consulted for the resume point itself.
        """
        atoms, last_step = store.latest_state(run_id)
        row = store._row_at_step(run_id, last_step)
        if row.data.get("metadata", {}).get("method") == "energetic_force_error":
            raise NotImplementedError(
                "Energetic trajectories require their own calibration and check state; "
                "legacy Runner.resume cannot restore them"
            )
        energy, forces = store.driving_label(run_id, last_step)
        # Logged momenta are the half-step momenta; reconstruct the on-step
        # momenta exactly as the integrator computed them (p += 0.5*dt*F).
        on_step_momenta = atoms.get_momenta() + 0.5 * (timestep_fs * units.fs) * forces
        atoms.set_momenta(on_step_momenta)

        runner = cls(
            atoms=atoms,
            surrogate=surrogate,
            engine=engine,
            switch=switch,
            store=store,
            run_id=run_id,
            timestep_fs=timestep_fs,
            on_label=on_label,
            explore_frac=explore_frac,
            explore_rng=explore_rng,
            explore_seed=explore_seed,
        )
        runner.calc.step = last_step + 1
        # Prime the ASE result cache with the stored label: the integrator's
        # first get_forces() of the resumed run is then a cache hit.
        runner.calc.atoms = atoms.copy()
        runner.calc.results = {
            "energy": float(energy),
            "forces": np.asarray(forces, dtype=float),
        }
        return runner
