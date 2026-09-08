"""Descriptive atom-group residual work from an immutable evaluation snapshot.

No new reference calculations, prediction changes, or acceptance criteria.
"""

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from final_interface_evaluate_20260906 import DATA, require

GROUPS = {
    "slab_Li": (0, 36),
    "initial_EC": (36, 246),
    "initial_DMC": (246, 450),
    "initial_salt": (450, 474),
}


def analyze(snapshot):
    raw_snapshot = snapshot.read_bytes()
    evaluation = json.loads(raw_snapshot)
    hashes = evaluation["input_sha256"]
    for name, digest in hashes.items():
        require(
            hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest,
            f"Changed evaluation input: {name}",
        )

    def read(name, lines=False):
        require(name in hashes, f"Input was not verified in the evaluation: {name}")
        raw = (DATA / name).read_text()
        return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)

    frames, corrections, points = {}, {}, {}
    for p in range(4):
        frames[p] = {
            f["time_fs"]: f
            for f in read(f"dynamics/path_{p}/primary_states.jsonl", True)
        }
        corrections[p] = np.array(
            read(f"dynamics/path_{p}/anchor.json")["correction_ev_A"]
        )
        points[p] = {
            0.0: {
                "positions": np.array(frames[p][0.0]["positions_angstrom"]),
                "residual": np.zeros((474, 3)),
                "endpoint_work_ev": 0.0,
            }
        }

    for kind, rows in (
        ("future", evaluation["future"]),
        ("check", evaluation["checks"]),
    ):
        for row in rows:
            if row["status"] != "complete" or (
                kind == "check" and row["kind"] != "quadrature"
            ):
                continue
            p, t = row["path_index"], row["time_fs"]
            ref = read(f"{kind}_reference_results/case_{row['case']:02d}/result.json")
            frame = frames[p][t]
            require(
                frame["geometry_sha256"] == ref["geometry_sha256"],
                "Wrong path geometry",
            )
            require(len(ref["chemical_symbols"]) == 474, "Wrong material cell")
            for group, formula in (
                ("slab_Li", {"Li": 36}),
                ("initial_EC", {"C": 63, "H": 84, "O": 63}),
                ("initial_DMC", {"C": 51, "H": 102, "O": 51}),
                ("initial_salt", {"Li": 3, "P": 3, "F": 18}),
            ):
                start, end = GROUPS[group]
                elements = ref["chemical_symbols"][start:end]
                require(
                    {e: elements.count(e) for e in set(elements)} == formula,
                    f"Wrong fixed identities in {group}",
                )
            points[p][t] = {
                "positions": np.array(frame["positions_angstrom"]),
                "residual": np.array(frame["base_forces_ev_a"])
                + corrections[p]
                - ref["forces_ev_a"],
                "endpoint_work_ev": row["endpoint_work_ev"],
            }

    results = []
    for p, end, spacing in itertools.product(
        range(4), (0.25, 0.5, 1.0), (0.5, 0.25, 0.125)
    ):
        if end < spacing:
            continue
        times = [i * spacing for i in range(round(end / spacing) + 1)]
        missing = [t for t in times if t not in points[p]]
        result = {
            "path_index": p,
            "end_fs": end,
            "spacing_fs": spacing,
            "status": "pending" if missing else "complete",
            "missing_times_fs": missing,
        }
        if not missing:
            atomic_work = np.zeros(474)
            scalar_work = 0.0
            for a, b in itertools.pairwise(times):
                fa, fb = points[p][a], points[p][b]
                products = (
                    0.5
                    * (fa["residual"] + fb["residual"])
                    * (fb["positions"] - fa["positions"])
                )
                scalar_work += float(products.sum())
                atomic_work += products.sum(axis=1)
            groups = {g: float(atomic_work[a:b].sum()) for g, (a, b) in GROUPS.items()}
            require(
                abs(sum(groups.values()) - scalar_work) < 1e-12,
                "Group additivity failed",
            )
            maximum_atom = int(
                np.linalg.norm(points[p][end]["residual"], axis=1).argmax()
            )
            result.update(
                total_quadrature_ev=scalar_work,
                group_work_ev=groups,
                group_fraction_of_net_work={
                    g: w / scalar_work if scalar_work else None
                    for g, w in groups.items()
                },
                atomic_work_ev=atomic_work.tolist(),
                positive_atomic_work_sum_ev=float(atomic_work[atomic_work > 0].sum()),
                negative_atomic_work_sum_ev=float(atomic_work[atomic_work < 0].sum()),
                endpoint_max_residual_atom_zero_based=maximum_atom,
                endpoint_max_residual_atom_work_ev=float(atomic_work[maximum_atom]),
                endpoint_work_ev=points[p][end]["endpoint_work_ev"],
                quadrature_minus_endpoint_ev=scalar_work
                - points[p][end]["endpoint_work_ev"],
            )
        results.append(result)
    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "purpose": "Post-endpoint descriptive physical interpretation; no acceptance criteria",
        "definition": "Sum over fixed atom identities of trapezoidal residual-force work along the unwrapped primary path",
        "interpretation": "Pathwise force-work partition, not independent atomic DFT energies or a causal molecular-energy decomposition; fractions may be signed",
        "groups_zero_based_half_open": GROUPS,
        "results": results,
        "evaluation_snapshot": str(snapshot),
        "evaluation_sha256": hashlib.sha256(raw_snapshot).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.evaluation)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete_grids": sum(
                    r["status"] == "complete" for r in result["results"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
