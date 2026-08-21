"""Zero-shot foundation-model shootout on the 31 DFT-labeled smoke configs.

Compares force accuracy of candidate MACE foundations against the stored
QE/PBE labels from smoke run 7615191 (106-atom Li/EC/DMC/LiPF6 mini box,
300 K NVT frames). Forces are the acceptance metric for the adaptive loop
(eps_acc is a force tolerance), so this table directly ranks candidates.

Usage: .venv/bin/python analysis/foundation_shootout.py
"""

from __future__ import annotations

import numpy as np
from ase.db import connect

DB = "analysis/smoke_7615191/loop.db"
OLD_CKPT = "analysis/20231210mace128L0_energy_epoch249model"

CANDIDATES = [
    ("old-20231210-128L0", "file", OLD_CKPT),
    ("mace-mp-0b3-small", "mp", "small-0b3"),
    ("mace-mp-0b3-medium", "mp", "medium-0b3"),
    ("mace-mpa-0-medium", "mp", "medium-mpa-0"),
]


def load_rows():
    db = connect(DB)
    rows = sorted(
        [r for r in db.select(run_id="smoke-7615191")],
        key=lambda r: r.key_value_pairs["step"],
    )
    out = []
    for r in rows:
        eng = r.data.get("engine")
        if eng is None:
            continue
        out.append((r.toatoms(), np.asarray(eng["forces"]), float(eng["energy"])))
    return out


def evaluate(name: str, kind: str, spec: str, rows) -> dict:
    if kind == "file":
        from mace.calculators import MACECalculator

        calc = MACECalculator(
            model_paths=spec, device="cpu", default_dtype="float64"
        )
    else:
        from mace.calculators import mace_mp

        calc = mace_mp(model=spec, device="cpu", default_dtype="float64")

    fmae, fmax, de = [], [], []
    for atoms, f_ref, e_ref in rows:
        a = atoms.copy()
        a.calc = calc
        f = a.get_forces()
        e = a.get_potential_energy()
        d = np.linalg.norm(f - f_ref, axis=1)
        fmae.append(float(d.mean()))
        fmax.append(float(d.max()))
        de.append(e / len(a) - e_ref / len(a))
    de_arr = np.array(de)
    de_arr -= de_arr.mean()  # absorb per-model atomic energy referencing
    return {
        "name": name,
        "fMAE": float(np.mean(fmae)),
        "fMAE_std": float(np.std(fmae)),
        "fMax_mean": float(np.mean(fmax)),
        "fMax_worst": float(np.max(fmax)),
        "dE_atom_mad_meV": float(np.mean(np.abs(de_arr)) * 1000),
    }


def main() -> None:
    rows = load_rows()
    print(f"{len(rows)} labeled configs loaded")
    for name, kind, spec in CANDIDATES:
        try:
            r = evaluate(name, kind, spec, rows)
        except Exception as exc:  # model download/load failure should not kill the table
            print(f"{name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        print(
            f"{r['name']:>22}  fMAE {r['fMAE']:.4f} ± {r['fMAE_std']:.4f}  "
            f"fMax(mean/worst) {r['fMax_mean']:.3f}/{r['fMax_worst']:.3f}  "
            f"|dE/atom| {r['dE_atom_mad_meV']:.2f} meV"
        )


if __name__ == "__main__":
    main()
