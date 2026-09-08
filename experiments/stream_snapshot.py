"""stream_snapshot.py — regenerate the flagship stream snapshot from loop.db.

The figure scripts (fig5, fig8, refusal_curve) read
analysis/flagship_a_prod/stream_snapshot_latest.json: one record per
trajectory step with the supervisor's own logged quantities. Nothing here
is recomputed behind the run's back: s/qhat/W/B are parsed from the stored
decision record (row.data["reason"]); the true error e at engine-labeled
steps is recomputed as max-atom |F_surrogate - F_engine| from the stored
force arrays (bit-stable reconstruction, verified against the previous
snapshot).

Usage:
    python experiments/stream_snapshot.py \
        --db analysis/flagship_a_prod/loop.db --run-id flagship-a-prod \
        --out analysis/flagship_a_prod/stream_snapshot_latest.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RE_Q = re.compile(r"qhat=([\d.]+)")
RE_W = re.compile(r"\|W\|=(\d+)")
RE_B = re.compile(r"B(?:_k)?\(s\)=([\d.]+)")
RE_K = re.compile(r"streak k=(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    import ase.db

    args = parse_args()
    db = ase.db.connect(args.db)
    rows = [r for r in db.select(run_id=args.run_id)]
    rows.sort(key=lambda r: r.key_value_pairs["step"])

    snap = []
    for r in rows:
        kvp = r.key_value_pairs
        reason = r.data.get("reason", "")
        rec = {
            "step": int(kvp["step"]),
            "route": kvp["route"],
            "s": round(float(np.max(np.asarray(r.data["surrogate"]["uncertainty"]))), 5),
        }
        m = RE_Q.search(reason)
        if m:
            rec["qhat"] = float(m.group(1))
        m = RE_W.search(reason)
        if m:
            rec["W"] = int(m.group(1))
        m = RE_B.search(reason)
        if m:
            rec["B"] = float(m.group(1))
        m = RE_K.search(reason)
        if m:
            rec["streak_k"] = int(m.group(1))
        if kvp["route"] != "ml" and "engine" in r.data:
            fs = np.asarray(r.data["surrogate"]["forces"], float)
            fe = np.asarray(r.data["engine"]["forces"], float)
            rec["e"] = float(np.max(np.linalg.norm(fs - fe, axis=-1)))
        snap.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, indent=1) + "\n")
    n_ml = sum(1 for r in snap if r["route"] == "ml")
    print(f"wrote {len(snap)} records ({n_ml} accepted), steps "
          f"{snap[0]['step']}..{snap[-1]['step']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
