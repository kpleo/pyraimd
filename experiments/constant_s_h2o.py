"""Constant-spread ablation on the water validation (referee §5.1):
calibration without the sensor — the missing cell of the 2x2.

The 2x2 over {spread sensor} x {online calibration}:
- tuned threshold (tau=0.0054): sensor, no calibration  -> alpha_hat 0.304
- scheduled (period 4):         neither                 -> alpha_hat 0.0929
- conformal (baseline):         both                    -> alpha_hat 0.000
- THIS RUN:                     calibration, no sensor  -> ?

The ablation fixes the committee spread at s0 (the median spread of the
baseline conformal run, read from its decision log) so the routing bound
degenerates to the per-frame constant B = qhat * (s0 + delta): the sliding
window still calibrates the normalized nonconformity r = e/(s0 + delta)
online (calibration stays), but the bound no longer reacts to the per-frame
spread (sensor gone). Everything else — fresh committee, same seed, same
fine-tune trigger, identical replay code path — matches the baseline
conformal replay (analysis/h2o_streak/decision_log_conformal.json).

Implementation: ConstantSpreadConformalSwitch subclasses ConformalSwitch and
overrides assess() to evaluate the bound on a copy of the prediction whose
uncertainty array is filled with s0; the updater's observe hook is wrapped
to ingest (s0, e) pairs, so the window holds exactly what the switch used.
The ReplayRecord.spread column keeps the TRUE per-frame spread (the replay
harness logs it from the unpatched prediction) — useful for seeing what the
sensor would have said.

Usage:
    uv run python experiments/constant_s_h2o.py \
        --store analysis/h2o_streak/coverage_h2o.db \
        --out analysis/h2o_streak/constant_s_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from ase import Atoms

from pyraimd2.loop import OnlineUpdater
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate
from pyraimd2.surrogate.base import SurrogatePrediction
from pyraimd2.switch import ConformalSwitch, replay
from pyraimd2.switch.base import Decision

COLLECT_RUN_ID = "collect-h2o-nve-300K"
SEED = 20250819
EPS_ACC = 0.25  # the conformal run's budget (violation criterion)
ALPHA = 0.05
WINDOW = 64
W_MIN = 16
DELTA = 1e-3
N_LABEL = 8


class ConstantSpreadConformalSwitch(ConformalSwitch):
    """Conformal calibration with the spread sensor ablated to a constant s0.

    assess() evaluates B = qhat * (s0 + delta) instead of qhat * (s + delta);
    pair it with an updater whose observe ingests (s0, e) so the calibration
    window is self-consistent with what the switch saw.
    """

    def __init__(self, surrogate, s0: float, **kwargs) -> None:
        super().__init__(surrogate, **kwargs)
        if not np.isfinite(s0) or s0 < 0.0:
            raise ValueError(f"s0 must be finite and >= 0, got {s0}")
        self.s0 = float(s0)

    def assess(
        self, atoms: Atoms, step: int, prediction: SurrogatePrediction | None = None
    ) -> Decision:
        if prediction is None:
            prediction = self.surrogate.predict(atoms)
        patched = replace(
            prediction,
            uncertainty=np.full_like(prediction.uncertainty, self.s0),
        )
        decision = super().assess(atoms, step, prediction=patched)
        return Decision(
            route=decision.route,
            score=decision.score,
            reason=f"constant-s(s0={self.s0:.6f}) {decision.reason}",
        )


class _ProgressCommittee:
    """Delegates to CommitteeSurrogate; prints a progress line per fine-tune."""

    def __init__(self, committee: CommitteeSurrogate, tag: str) -> None:
        self._committee = committee
        self._tag = tag

    def predict(self, atoms):
        return self._committee.predict(atoms)

    def finetune(self, labels):
        t0 = time.perf_counter()
        report = self._committee.finetune(labels)
        print(f"  [{self._tag}] finetune on {report.n_labels} labels: loss "
              f"{report.initial_loss:.3f} -> {report.final_loss:.3f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return report


def median_spread(decision_log: Path) -> float:
    """Median committee spread over the baseline conformal replay's records."""
    payload = json.loads(decision_log.read_text())
    spreads = [record["spread"] for record in payload["records"]]
    return float(np.median(spreads))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--decision-log", type=Path,
                   default=Path("analysis/h2o_streak/decision_log_conformal.json"),
                   help="baseline conformal decision log (for s0 = median spread)")
    p.add_argument("--s0", type=float, default=None,
                   help="override the constant spread; default: median spread of "
                        "the baseline conformal decision log")
    p.add_argument("--max-frames", type=int, default=None,
                   help="replay only the first N frames (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames = list(Store(args.store).iter_labels(COLLECT_RUN_ID))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    print(f"{len(frames)} labeled frames", flush=True)

    s0 = args.s0 if args.s0 is not None else median_spread(args.decision_log)
    s0_source = ("--s0 override" if args.s0 is not None
                 else f"median spread of {args.decision_log}")
    print(f"s0 = {s0:.6f} eV/A ({s0_source})", flush=True)

    committee = CommitteeSurrogate(n_members=4, seed=SEED)
    switch = ConstantSpreadConformalSwitch(
        committee, s0, alpha=ALPHA, eps_acc=EPS_ACC,
        window=WINDOW, w_min=W_MIN, delta=DELTA,
    )
    updater = OnlineUpdater(
        _ProgressCommittee(committee, "constant-s"),
        observe=lambda s, e: switch.observe(switch.s0, e),
        n_label=N_LABEL,
    )
    records, summary = replay(frames, committee, switch, updater=updater,
                              eps_acc=EPS_ACC)
    print(f"constant-s: dft_fraction={summary.dft_fraction:.3f} "
          f"n_accepted={summary.n_accepted} alpha_hat={summary.alpha_hat:.4f} "
          f"({summary.wall_time_s:.0f} s)", flush=True)

    report = {
        "task": "T1 constant-spread ablation (calibration, no sensor)",
        "s0": s0,
        "s0_source": s0_source,
        "params": {"alpha": ALPHA, "eps_acc": EPS_ACC, "window": WINDOW,
                   "w_min": W_MIN, "delta": DELTA, "n_label": N_LABEL,
                   "seed": SEED},
        "note": ("record.spread is the TRUE per-frame committee spread; the "
                 "switch saw only s0 (bound B = qhat*(s0+delta)); window "
                 "ingested (s0, e) pairs"),
        "summary": asdict(summary),
        "baseline_conformal_reference": {
            "dft_fraction": 0.25165562913907286, "n_dft": 76,
            "n_accepted": 226, "alpha_hat": 0.0, "n_finetunes": 9,
        },
        "records": [asdict(record) for record in records],
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
