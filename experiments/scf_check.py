"""Single-point SCF protocol check for a campaign box (M3 prerequisite).

Runs one pw.x SCF through the QeEngine with the frozen §3.1 protocol and
reports convergence, energy, force norms, and wall time.  Use this to
de-risk a new box before committing to a labeling campaign: the interface
cell (474 atoms, metallic Li slab + liquid) is the immediate target.

nbnd is auto-computed from the pseudopotential valences as
ceil(nelec/2) + headroom — the QE default (nelec/2, zero empty bands) is
fragile for Davidson on large cells (design-m3.md §3.1).

Usage:
    python experiments/scf_check.py --atomsXYZ box.xyz --pseudo-dir ... \
        --pw-cmd "mpirun -np 28 pw.x" --metallic --workdir ...
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from ase.io import read

from pyraimd2.engines.qe_engine import QeConfig, QeEngine, valence_from_upf

NBND_HEADROOM = 60  # empty bands beyond nelec/2 (§3.1: occupied + ~50-60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atomsXYZ", required=True, type=Path)
    p.add_argument("--pseudo-dir", required=True, type=Path)
    p.add_argument("--pw-cmd", required=True)
    p.add_argument("--ecutwfc", type=float, default=60.0)
    p.add_argument("--ecutrho", type=float, default=600.0)
    p.add_argument("--conv-thr", type=float, default=1e-6)
    p.add_argument("--metallic", action="store_true")
    p.add_argument("--smearing", default="mv", choices=["mv", "fd", "mp", "gauss"])
    p.add_argument("--degauss", type=float, default=0.01)
    p.add_argument("--nbnd-headroom", type=int, default=NBND_HEADROOM,
                   help="empty bands beyond nelec/2 (metals at the Fermi level "
                        "need many partial bands: 150+ for the interface slab)")
    p.add_argument("--mixing-beta", type=float, default=0.3)
    p.add_argument("--electron-maxstep", type=int, default=200)
    p.add_argument("--timeout-s", type=float, default=10800.0,
                   help="per-attempt pw.x wall timeout (s); raise for slow "
                        "large-cell SCF so converging tails are not killed")
    p.add_argument("--frame-index", type=int, default=None,
                   help="read this frame of a multi-frame extxyz (default: all/last)")
    p.add_argument("--startpot", action="store_true",
                   help="restart from the previous charge density in the same "
                        "workdir (chain labeling of MD frames)")
    p.add_argument("--out-jsonl", type=Path, default=None,
                   help="append a JSON record (energy/forces/wall) per call")
    p.add_argument("--workdir", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.atomsXYZ, index=args.frame_index if args.frame_index is not None else -1)
    if abs(np.linalg.det(atoms.cell)) < 1e-8:
        raise ValueError("input xyz carries no cell; interface boxes must ship theirs")

    cfg0 = QeConfig(pseudo_dir=str(args.pseudo_dir))
    nelec = 0.0
    # Count electrons from the configured pseudopotentials.
    from collections import Counter

    counts = Counter(atoms.get_chemical_symbols())
    for sym, n in sorted(counts.items()):
        zv = valence_from_upf(Path(args.pseudo_dir) / cfg0.pseudos[sym])
        print(f"  {sym}: {n} atoms x z_valence {zv}")
        nelec += n * zv
    nbnd = math.ceil(nelec / 2) + args.nbnd_headroom
    print(f"nelec = {nelec:.0f} -> nbnd = {nbnd}")

    cfg = QeConfig(
        pseudo_dir=str(args.pseudo_dir),
        pw_cmd=tuple(args.pw_cmd.split()),
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        nbnd=nbnd,
        metallic=args.metallic,
        smearing=args.smearing,
        degauss=args.degauss,
        conv_thr=args.conv_thr,
        mixing_beta=args.mixing_beta,
        mixing_ndim=12,
        diago_david_ndim=8,
        diago_full_acc=True,
        electron_maxstep=args.electron_maxstep,
        startpot_file=args.startpot,
        timeout_s=args.timeout_s,
    )
    engine = QeEngine(cfg, run_root=args.workdir / "qe_runs")

    t0 = time.perf_counter()
    result = engine.compute(atoms)
    wall = time.perf_counter() - t0
    fnorm = np.linalg.norm(np.asarray(result.forces), axis=1)
    print(f"SCF CHECK OK: E = {result.energy:.6f} eV, wall = {wall:.0f} s")
    print(f"forces: RMS {np.sqrt((fnorm**2).mean()):.4f} eV/A, max {fnorm.max():.4f} eV/A")
    if args.out_jsonl is not None:
        import json

        record = {
            "atomsXYZ": str(args.atomsXYZ),
            "frame_index": args.frame_index,
            "energy_ev": float(result.energy),
            "forces_ev_a": np.asarray(result.forces).tolist(),
            "nbnd": nbnd,
            "metallic": bool(args.metallic),
            "smearing": args.smearing,
            "degauss": args.degauss,
            "startpot": bool(args.startpot),
            "wall_time_s": wall,
        }
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
