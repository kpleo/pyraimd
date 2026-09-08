"""Adaptive MD driver for a single allocated HPC node.

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

Interface boxes (non-cubic) ship their cell in the xyz — omit --cell and
pass the metallic protocol: --metallic --smearing fd --degauss 0.02
--mixing-beta 0.20 --mixing-ndim 12 --diago-david-ndim 8
--nbnd-headroom 150 --electron-maxstep 400 --ecutwfc 60 --ecutrho 600
--conv-thr 1e-6.

Local smoke (PySCF-free, QE-free): not applicable — this driver is for the
cluster; the local loop coverage is tests/unit/test_runner.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.engines.qe_engine import (
    DEFAULT_PSEUDOS,
    QeConfig,
    QeEngine,
    valence_from_upf,
)
from pyraimd2.loop import OnlineUpdater, Runner
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.switch import ConformalSwitch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atomsXYZ", required=True, type=Path)
    p.add_argument("--cell", type=float, default=None,
                   help="cubic box edge, Angstrom; omit when the xyz ships its own "
                        "cell (the 474-atom interface box is 10.53x10.53x52.15 — "
                        "a cubic --cell would clobber it)")
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
    p.add_argument("--ecutrho", type=float, default=None,
                   help="density cutoff (Ry); default = 8x ecutwfc. The example "
                        "protocols use 10x (60/600 electrolyte, 70/700 W) — pass "
                        "explicitly to match the bootstrap labels")
    p.add_argument("--conv-thr", type=float, default=None,
                   help="SCF conv_thr (Ry); default = engine 1e-8. Use 1e-6 for the "
                        "288-atom box: the QE accuracy norm is extensive, so 1e-6 there "
                        "is per-atom tighter than 1e-8 on the Si validation cell")
    p.add_argument("--nbnd", type=int, default=None,
                   help="KS bands; default = nelec/2 (zero empty bands — fragile "
                        "Davidson on large insulating cells, pass extras there)")
    p.add_argument("--nbnd-headroom", type=int, default=None,
                   help="empty bands beyond ceil(nelec/2), computed from the UPF "
                        "valences (mutually exclusive with --nbnd). The metallic "
                        "interface protocol uses 150")
    p.add_argument("--smearing", default="mv", choices=["mv", "fd", "mp", "gauss"],
                   help="smearing type when --metallic; fd (Fermi-Dirac) converged "
                        "in 64-115 iterations on the Li interface where mv "
                        "plateau-oscillated (7616174 probe)")
    p.add_argument("--degauss", type=float, default=0.02,
                   help="smearing width (Ry) when --metallic; 0.02 with fd gave "
                        "digit-identical energies across 150/250 Ry degauss checks")
    p.add_argument("--mixing-beta", type=float, default=None,
                   help="charge-mixing beta; default = engine 0.3. Use 0.20 for the "
                        "metallic interface box (7616174 probe)")
    p.add_argument("--mixing-ndim", type=int, default=None,
                   help="Broyden history length; default = engine 8. 12 steadies "
                        "~1k-electron cells")
    p.add_argument("--diago-david-ndim", type=int, default=None,
                   help="Davidson subspace multiplier; default = engine 4. 8 for "
                        "~500-band systems")
    p.add_argument("--diago-full-acc", action="store_true",
                   help="tightly converge ALL bands each SCF step (bulk protocol)")
    p.add_argument("--electron-maxstep", type=int, default=None,
                   help="max SCF iterations; default = engine 200. The interface "
                        "box needed up to 115 with fd smearing — pass 400")
    p.add_argument("--startpot-chain", action="store_true",
                   help="chain DFT labels on the previous converged density "
                        "(startpot_file + wipe-on-failure retry in QeEngine); "
                        "saves the measured ~25%% SCF wall on trajectory-"
                        "correlated frames — the loop's labels all share "
                        "run_root/step, so the chain persists across steps")
    p.add_argument("--torch-threads", type=int, default=16,
                   help="in-process torch threads for predict/fine-tune phases "
                        "(probe 7616841: 16 threads = 20.6 s/label/epoch/member "
                        "on the 474-atom box, VmHWM 41.5 GB; env-side thread "
                        "vars provably do NOT control torch here)")
    p.add_argument("--eps-acc", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--streak-rho", type=float, default=0.0,
                   help="streak-inflation rate: B_k(s) = qhat*(s+delta)*(1+rho*k), "
                        "k = accepted steps since the last decision-driven DFT label. "
                        "0 disables streak inflation")
    p.add_argument("--explore-frac", type=float, default=0.0,
                   help="randomized exploration-label fraction ("
                        "OFF at 0.0). "
                        "On each ACCEPTED step, with probability p the engine "
                        "label is computed anyway (shadow label): it enters "
                        "the calibration window/updater as usual — pure "
                        "drift audit — but the step still propagates with the "
                        "surrogate forces. New labels may affect later "
                        "decisions and updates. Rows keep route=ml with "
                        "'explore-label' in the reason; the conformal streak "
                        "is NOT reset (it counts decision-driven labels). "
                        "Draws come from a dedicated default_rng(--seed) "
                        "stream, re-seeded per segment, so the explore "
                        "schedule is deterministic given (seed, segment)")
    p.add_argument("--window", type=int, default=64,
                   help="conformal sliding-window size (labels); shorten only for "
                        "smoke tests so post-fine-tune pairs evict the cold-start era")
    p.add_argument("--n-label", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30,
                   help="fine-tune epochs per trigger; 30 is the 7616069 probe "
                        "sweet spot (25 epochs ~= 50 at half the wall time)")
    p.add_argument("--train-window", type=int, default=0,
                   help="cap each fine-tune's training set to the most recent N "
                        "stored labels (0 = all). Bounds the linear cost growth "
                        "(~374 s/label) that dominates wall time on long runs")
    p.add_argument("--trainable-filters", nargs="+", default=["readout", "products"],
                   help="parameter-name substrings selecting trainable tensors; "
                        "readout+products won the 7615818 capacity probe "
                        "(held-out fMax 0.36 vs 0.85 readout-only)")
    p.add_argument("--w-min", type=int, default=16)
    p.add_argument("--metallic", action="store_true",
                   help="smearing occupations for Li-slab cells (see "
                        "--smearing/--degauss)")
    p.add_argument("--kpts", type=int, nargs=3, default=None)
    p.add_argument("--seed", type=int, default=20250819)
    p.add_argument("--resume", action="store_true",
                   help="resume run-id from workdir/loop.db: restores the MD state "
                        "bit-for-bit (Runner.resume), reloads workdir/committee.pt "
                        "into the surrogate, and rebuilds the conformal window from "
                        "the stored (s, e) stream")
    p.add_argument("--init-committee", type=Path, default=None,
                   help="warm-start checkpoint for a FRESH run (e.g. the bootstrap "
                        "committee_bootstrap.pt): loaded into the committee before "
                        "step 0. Mutually exclusive with --resume; the conformal "
                        "window still burns in from the live trajectory, so the "
                        "marginal guarantee is untouched")
    p.add_argument("--workdir", required=True, type=Path, help="run dir (store + QE dirs)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.init_committee is not None:
        raise ValueError(
            "--resume and --init-committee are mutually exclusive: --resume "
            "restores the run's own checkpoint/labels; --init-committee seeds a "
            "fresh run from an external checkpoint"
        )
    if not 0.0 <= args.explore_frac <= 1.0:
        raise ValueError(f"--explore-frac must be in [0, 1], got {args.explore_frac}")
    args.workdir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.atomsXYZ)
    if args.cell is not None:
        atoms.set_cell([args.cell] * 3)
        atoms.center()
    elif abs(np.linalg.det(atoms.cell)) < 1e-8:
        raise ValueError(
            "--cell not given and the xyz carries no cell; interface boxes must "
            "ship theirs"
        )
    atoms.set_pbc(True)
    rng = np.random.default_rng(args.seed)
    # Dedicated explore-label stream (see --explore-frac): a Generator on the
    # run's own seed, independent of the global RNG behind thermalization,
    # re-seeded identically every segment so draws are deterministic given
    # (seed, segment).
    explore_rng = np.random.default_rng(args.seed) if args.explore_frac > 0.0 else None

    nbnd = args.nbnd
    if args.nbnd_headroom is not None:
        if nbnd is not None:
            raise ValueError("--nbnd and --nbnd-headroom are mutually exclusive")
        nelec = sum(
            n * valence_from_upf(Path(args.pseudo_dir) / DEFAULT_PSEUDOS[sym])
            for sym, n in Counter(atoms.get_chemical_symbols()).items()
        )
        nbnd = math.ceil(nelec / 2) + args.nbnd_headroom
        print(f"nelec = {nelec:.0f} -> nbnd = {nbnd}", flush=True)

    cfg = QeConfig(
        pseudo_dir=str(args.pseudo_dir),
        pw_cmd=tuple(args.pw_cmd.split()),
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho if args.ecutrho is not None else 8.0 * args.ecutwfc,
        kpts=tuple(args.kpts) if args.kpts else None,
        nbnd=nbnd,
        conv_thr=args.conv_thr if args.conv_thr is not None else QeConfig.conv_thr,
        metallic=args.metallic,
        smearing=args.smearing,
        degauss=args.degauss,
        mixing_beta=(
            args.mixing_beta if args.mixing_beta is not None else QeConfig.mixing_beta
        ),
        mixing_ndim=(
            args.mixing_ndim if args.mixing_ndim is not None else QeConfig.mixing_ndim
        ),
        diago_david_ndim=(
            args.diago_david_ndim
            if args.diago_david_ndim is not None
            else QeConfig.diago_david_ndim
        ),
        diago_full_acc=args.diago_full_acc,
        electron_maxstep=(
            args.electron_maxstep
            if args.electron_maxstep is not None
            else QeConfig.electron_maxstep
        ),
        startpot_file=args.startpot_chain,
        timeout_s=10800.0,  # fresh SCF on the 474-atom box measured up to 8591 s
    )
    engine = QeEngine(cfg, run_root=args.workdir / "qe_runs")
    model_arg = [str(m) for m in args.model]
    committee = CommitteeSurrogate(
        model=model_arg if len(model_arg) > 1 else model_arg[0],
        n_members=4, seed=args.seed, epochs=args.epochs,
        trainable_filters=tuple(args.trainable_filters),
    )
    import torch  # process-wide pin for the predict/fine-tune phases (see help)
    torch.set_num_threads(args.torch_threads)
    store = Store(args.workdir / "loop.db")
    switch = ConformalSwitch(
        committee, alpha=args.alpha, eps_acc=args.eps_acc, w_min=args.w_min,
        window=args.window, streak_rho=args.streak_rho,
        initial_streak=store.trailing_ml_streak(args.run_id) if args.resume else 0,
    )

    def _save_committee() -> None:
        import torch  # local import: heavy, and only needed once fine-tunes fire

        torch.save(committee.state_dict(), args.workdir / "committee.pt")

    def _training_labels() -> list:
        # Full stored label set (including pre-resume segments), optionally
        # capped to the most recent --train-window entries.
        labels = list(store.iter_labels(args.run_id))
        if args.train_window > 0:
            labels = labels[-args.train_window :]
        return labels

    updater = OnlineUpdater(
        committee,
        observe=switch.observe,
        n_label=args.n_label,
        label_source=_training_labels,
        checkpoint=_save_committee,
    )

    if args.resume:
        import torch

        ckpt = args.workdir / "committee.pt"
        if ckpt.exists():
            committee.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else:
            # Warm start: no checkpoint (e.g. the pre-checkpoint-hook v4 run).
            # Bounded-forgetting semantics make one full-label fine-tune the
            # deterministic continuation of the run's training history —
            # loud, logged, and checkpointed afterwards; never silent.
            labels = _training_labels()
            if not labels:
                raise RuntimeError(
                    f"--resume found no checkpoint and no stored labels for "
                    f"{args.run_id!r}; nothing to warm-start from"
                )
            print(
                f"warm start: no {ckpt.name}; fine-tuning on all "
                f"{len(labels)} stored labels before resuming",
                flush=True,
            )
            committee.finetune(labels)
            _save_committee()
        # The conformal window is in-memory; rebuild it from the stored
        # (s, e) stream so calibration continues exactly where the run left off.
        for _, s, e in store.iter_observations(args.run_id):
            switch.observe(s, e)
        runner = Runner.resume(
            store=store,
            run_id=args.run_id,
            surrogate=committee,
            engine=engine,
            switch=switch,
            timestep_fs=args.timestep_fs,
            on_label=updater,
            explore_frac=args.explore_frac,
            explore_rng=explore_rng,
            explore_seed=args.seed,
        )
    else:
        if args.init_committee is not None:
            import torch

            committee.load_state_dict(
                torch.load(args.init_committee, map_location="cpu")
            )
            print(
                f"warm start: committee initialized from {args.init_committee}",
                flush=True,
            )
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
            explore_frac=args.explore_frac,
            explore_rng=explore_rng,
            explore_seed=args.seed,
        )
    del rng  # seed pinned through np.random.seed above (fresh start only)

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
        "n_explore_labels": runner.calc.n_explore,
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
