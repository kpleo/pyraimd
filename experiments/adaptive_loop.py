"""Adaptive MD driver for a single allocated HPC node (M3).

Runs the full pyraimd2 loop inside ONE Slurm allocation: MACE committee
predicts, ConformalSwitch decides, and on "dft" the QeEngine launches pw.x
with mpirun on the same node (predict and label phases are sequential, so
torch threads and MPI ranks never contend). This is the production shape for
the electrolyte/interface campaigns — no per-step queue waits.

Usage (inside an sbatch script, whole node allocated):
    python experiments/adaptive_loop.py \
        --atomsXYZ path/to/box.xyz --cell 15.0 \
        --model $PROJECT/software/20231210mace128L0_energy_epoch249model \
        --pseudo-dir $PROJECT/inputs \
        --pw-cmd "mpirun -np 28 pw.x" \
        --run-id bulk-lp30-smoke --n-steps 200 --eps-acc 0.25

Local smoke (PySCF-free, QE-free): not applicable — this driver is for the
cluster; the local loop coverage is tests/unit/test_runner.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.engines.qe_engine import QeConfig, QeEngine
from pyraimd2.loop import OnlineUpdater, Runner
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atomsXYZ", required=True, type=Path)
    p.add_argument("--cell", required=True, type=float, help="cubic box edge, Angstrom")
    p.add_argument("--model", required=True, type=Path, nargs="+",
                   help="one MACE .model path (deep-copied to K members) or K "
                        "paths for a mixed-backbone committee (e.g. 0b3 0b3 mpa0 mpa0)")
    p.add_argument("--pseudo-dir", required=True, type=Path)
    p.add_argument("--pw-cmd", required=True, help='e.g. "mpirun -np 28 pw.x"')
    p.add_argument("--run-id", required=True)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--ecutwfc", type=float, default=50.0)
    p.add_argument("--conv-thr", type=float, default=None,
                   help="SCF conv_thr (Ry); default = engine 1e-8. Use 1e-6 for the "
                        "288-atom box: the QE accuracy norm is extensive, so 1e-6 there "
                        "is per-atom tighter than 1e-8 on the Si validation cell")
    p.add_argument("--nbnd", type=int, default=None,
                   help="KS bands; default = nelec/2 (zero empty bands — fragile "
                        "Davidson on large insulating cells, pass extras there)")
    p.add_argument("--eps-acc", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window", type=int, default=64,
                   help="conformal sliding-window size (labels); shorten only for "
                        "smoke tests so post-fine-tune pairs evict the cold-start era")
    p.add_argument("--n-label", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50,
                   help="fine-tune epochs per trigger (committee recipe; lower only "
                        "for smoke cost control)")
    p.add_argument("--w-min", type=int, default=16)
    p.add_argument("--metallic", action="store_true", help="Li slab cells: mv smearing")
    p.add_argument("--kpts", type=int, nargs=3, default=None)
    p.add_argument("--seed", type=int, default=20250819)
    p.add_argument("--workdir", required=True, type=Path, help="run dir (store + QE dirs)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.atomsXYZ)
    atoms.set_cell([args.cell] * 3)
    atoms.set_pbc(True)
    atoms.center()
    rng = np.random.default_rng(args.seed)

    cfg = QeConfig(
        pseudo_dir=str(args.pseudo_dir),
        pw_cmd=tuple(args.pw_cmd.split()),
        ecutwfc=args.ecutwfc,
        ecutrho=8.0 * args.ecutwfc,
        kpts=tuple(args.kpts) if args.kpts else None,
        nbnd=args.nbnd,
        conv_thr=args.conv_thr if args.conv_thr is not None else QeConfig.conv_thr,
        metallic=args.metallic,
        timeout_s=7200.0,
    )
    engine = QeEngine(cfg, run_root=args.workdir / "qe_runs")
    model_arg = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=model_arg if len(model_arg) > 1 else model_arg[0],
        n_members=4, seed=args.seed, epochs=args.epochs,
    )
    switch = ConformalSwitch(
        committee, alpha=args.alpha, eps_acc=args.eps_acc, w_min=args.w_min,
        window=args.window,
    )
    store = Store(args.workdir / "loop.db")
    updater = OnlineUpdater(committee, observe=switch.observe, n_label=args.n_label)

    # VelocityVerlet is deterministic given (R, v); seed the initial momenta
    # through the Runner (thermalize uses the global RNG — pin it).
    np.random.seed(args.seed)
    runner = Runner(
        atoms,
        surrogate=committee,
        engine=engine,
        switch=switch,
        store=store,
        run_id=args.run_id,
        timestep_fs=args.timestep_fs,
        temperature_K=args.temperature,
        on_label=updater,
    )
    del rng  # seed pinned through np.random.seed above

    t0 = time.perf_counter()
    summary = runner.run(args.n_steps)
    wall = time.perf_counter() - t0

    report = {
        "run_id": args.run_id,
        "n_steps": summary.n_steps,
        "n_dft": summary.n_dft,
        "dft_fraction": summary.dft_fraction,
        "force_mae_ev_a": summary.force_mae_ev_a,
        "force_max_ev_a": summary.force_max_ev_a,
        "wall_time_s": wall,
        "n_finetunes": updater.n_finetunes,
        "finetune_reports": [dataclasses.asdict(r) for r in updater.reports],
        "final_window": switch.window_size,
        "final_qhat": switch.qhat(),
        "config": vars(args) | {"atomsXYZ": str(args.atomsXYZ), "model": str(args.model)},
    }
    out = args.workdir / "summary.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
