"""M1 acceptance demo: 200-step adaptive NVE on H2O at 300 K, CPU only.

Chain: PyscfEngine (PBE/def2-SVP) + MaceSurrogate (MACE-MP-0 small, frozen)
+ ScheduledSwitch(period=10) + Store (SQLite) + Runner (VelocityVerlet).

Every force evaluation is logged; steps 0, 10, 20, ... are DFT checkpoints
with a shadow surrogate prediction, so the RunSummary reports the DFT
fraction and the surrogate-vs-DFT force error along the trajectory.

Usage:  uv run python examples/adaptive_h2o.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ase.build import molecule

from pyraimd2.engines import PyscfEngine
from pyraimd2.loop import Runner
from pyraimd2.store import Store
from pyraimd2.surrogate import MaceSurrogate
from pyraimd2.switch import ScheduledSwitch

MD_STEPS = 200
TIMESTEP_FS = 0.5
TEMPERATURE_K = 300.0
SWITCH_PERIOD = 10


def main() -> int:
    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10  # distorted, as in examples/minimal_loop.py

    db_path = Path(tempfile.mkdtemp(prefix="pyraimd2_")) / "adaptive_h2o.db"
    store = Store(db_path)
    engine = PyscfEngine(functional="pbe", basis="def2-svp", conv_tol=1e-9)
    surrogate = MaceSurrogate(model="small", device="cpu", default_dtype="float64")
    switch = ScheduledSwitch(period=SWITCH_PERIOD)

    runner = Runner(
        atoms,
        surrogate,
        engine,
        switch,
        store,
        run_id="h2o-nve-300K",
        timestep_fs=TIMESTEP_FS,
        temperature_K=TEMPERATURE_K,
    )
    summary = runner.run(MD_STEPS)

    n_labels = len(list(store.iter_labels("h2o-nve-300K")))
    print("\n=== PYRAIMD-2 M1 adaptive loop: H2O NVE ===")
    print("run id:          h2o-nve-300K")
    print(f"store:           {db_path}")
    print(f"MD steps:        {summary.n_steps} x {TIMESTEP_FS} fs "
          f"= {summary.n_steps * TIMESTEP_FS:.0f} fs at {TEMPERATURE_K:.0f} K")
    print(f"DFT checkpoints: {summary.n_dft} (period {SWITCH_PERIOD}; "
          f"labels in store: {n_labels})")
    print(f"DFT fraction:    {summary.dft_fraction:.3f} of force evaluations")
    print(f"surrogate vs DFT forces:  MAE {summary.force_mae_ev_a:.4f} eV/A, "
          f"max {summary.force_max_ev_a:.4f} eV/A (per-atom |dF|, shadow)")
    print(f"wall time:       {summary.wall_time_s:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
