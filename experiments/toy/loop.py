"""Closed surrogate-oracle loop on Lorenz-63 — the PYRAIMD-2 toy port.

Decision layer: :class:`ToyConformalSwitch`, a SEMANTIC MIRROR of
src/pyraimd2/switch/conformal.py's ConformalSwitch.  The MD switch's
``assess(atoms, step, prediction)`` extracts s = max(per-atom spread) from a
SurrogatePrediction; the toy has no ASE Atoms, so ``assess(s, step)`` takes
the scalar spread directly.  Everything else is mirrored line-for-line and
reuses the production code where possible:

- the quantile itself IS the production ``conformal_quantile`` (imported);
- same cold-start rule (route "dft" iff |W| < w_min), same window eviction
  (deque(maxlen=W)), same normalized nonconformity r = e/(s+delta);
- same streak semantics: k = accepted steps since the last label, reset by
  a "dft" route, observe() is replay-safe (never touches the streak);
- same streak-inflated bound B_k = qhat*(s+delta)*(1+rho*k);
- same Decision object and byte-identical reason strings;
- the only addition is a read-only ``streak`` property (MD keeps ``_streak``
  private and reconstructs positions from the route stream in post-hoc
  audits; the toy logs it directly — no semantic change);
- the only wording change is the non-finite-spread ValueError message
  (toy committee, not per-atom surrogate spread); Decision strings are
  untouched.  tests/unit/test_toy_switch.py asserts toy and MD switches
  emit identical routes/scores/reasons on a scripted (s, e) stream.

Loop semantics mirror the MD loop (Runner + OnlineUpdater): the committee
predicts every step (shadow), the switch routes, a "dft" route produces an
oracle label at the current state, feeds (s, e) to the calibration window,
and every N_label=8 new labels fires a fine-tune on the most recent
train_window=16 labels.  The toy's one structural advantage: the oracle is
microseconds, so the true shadow error e is recorded at EVERY step
(accepted violations are measured exactly, not counterfactually).

Drift dial (T3): accepted steps integrate ``macro_mult`` x the base 2
substeps — the unlabeled integration time per accepted step scales, the
label stream (dft steps) stays at the base macro-step.

Records stream to an append-only JSONL store (one row per step, MD
decision-log key names: step/route/spread/error/qhat/bound/streak_k, plus
state/finetune/n_labels/n_sub).  ``replay_decisions`` rebuilds the switch
from the stored (s, e) stream and checks every stored decision — the
decision-layer analog of pyraimd2.switch.replay.

Smoke driver (this task):
    uv run python -m experiments.toy.loop --regime A --n-steps 2000 \
        --out-dir analysis/toy
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.toy.lorenz import (
    DT_SUB,
    N_SUB_BASE,
    RHO_REGIME_A,
    RHO_REGIME_B,
    LorenzEngine,
    rk4_step,
    verify_rk4,
)
from pyraimd2.switch.base import Decision
from pyraimd2.switch.conformal import conformal_quantile

# Mirror note: asserted byte-identical against ConformalSwitch by
# tests/unit/test_toy_switch.py — do not reword the Decision strings.
_NONFINITE_SPREAD_MSG = (
    "ToyConformalSwitch needs a finite committee spread "
    "(got max uncertainty {s}); pair it with a committee, not a "
    "single frozen model"
)


class ToyConformalSwitch:
    """Semantic mirror of ConformalSwitch for scalar states (see module docstring)."""

    def __init__(
        self,
        alpha: float = 0.05,
        eps_acc: float = 0.1,
        window: int = 64,
        w_min: int = 16,
        delta: float = 1e-3,
        streak_rho: float = 0.0,
        initial_streak: int = 0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if eps_acc <= 0.0:
            raise ValueError(f"eps_acc must be > 0, got {eps_acc}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 1 <= w_min <= window:
            raise ValueError(f"w_min must be in [1, window], got {w_min} (window {window})")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if streak_rho < 0.0:
            raise ValueError(f"streak_rho must be >= 0, got {streak_rho}")
        if initial_streak < 0:
            raise ValueError(f"initial_streak must be >= 0, got {initial_streak}")
        self.alpha = alpha
        self.eps_acc = eps_acc
        self.window = window
        self.w_min = w_min
        self.delta = delta
        self.streak_rho = streak_rho
        self._streak = initial_streak  # consecutive ml steps since last label
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)

    @property
    def window_size(self) -> int:
        return len(self._pairs)

    @property
    def streak(self) -> int:
        """Accepted steps since the last label (read-only view of the live
        counter; MD reconstructs this from the route stream in audits)."""
        return self._streak

    def qhat(self) -> float:
        """Current normalized-nonconformity quantile; +∞ on an empty window."""
        if not self._pairs:
            return float("inf")
        return conformal_quantile([e / (s + self.delta) for s, e in self._pairs], self.alpha)

    def bound(self, s: float) -> float:
        """B(s) = q̂ · (s + δ), the predicted field-error bound."""
        return self.qhat() * (s + self.delta)

    def observe(self, s: float, e: float) -> None:
        """Ingest one oracle-labeled observation; replay-safe: the acceptance
        streak is NOT touched here (mirrors ConformalSwitch.observe)."""
        if not np.isfinite(s) or s < 0.0:
            raise ValueError(f"spread s must be finite and >= 0, got {s}")
        if not np.isfinite(e) or e < 0.0:
            raise ValueError(f"error e must be finite and >= 0, got {e}")
        self._pairs.append((float(s), float(e)))

    def assess(self, s: float, step: int) -> Decision:
        """Mirror of ConformalSwitch.assess with the spread given directly."""
        s = float(s)
        if not np.isfinite(s):
            raise ValueError(_NONFINITE_SPREAD_MSG.format(s=s))
        n = len(self._pairs)
        qhat = self.qhat()
        k = self._streak
        bound = qhat * (s + self.delta) * (1.0 + self.streak_rho * k)
        route = "dft" if (n < self.w_min or bound > self.eps_acc) else "ml"
        streak_note = (
            f" streak k={k} rho={self.streak_rho}" if self.streak_rho > 0.0 else ""
        )
        why = (
            f"|W|={n} < w_min={self.w_min} (cold start)"
            if n < self.w_min
            else (
                f"B(s)={bound:.4f} > eps_acc={self.eps_acc} (over budget{streak_note})"
                if bound > self.eps_acc
                else f"B(s)={bound:.4f} <= eps_acc={self.eps_acc} (within budget{streak_note})"
            )
        )
        self._streak = 0 if route == "dft" else k + 1
        return Decision(
            route=route,
            score=bound,
            reason=(
                f"conformal: s={s:.5f} qhat={qhat:.4f} |W|={n} -> {why} "
                f"[alpha={self.alpha}, delta={self.delta}, step={step}]"
            ),
        )


@dataclass
class ToyConfig:
    """One toy run.  Defaults are the pre-registered decision-layer values."""

    rho: float = RHO_REGIME_A
    n_steps: int = 2000
    eps_acc: float = 0.1
    alpha: float = 0.05
    window: int = 64
    w_min: int = 16
    delta: float = 1e-3
    streak_rho: float = 0.0
    n_label: int = 8
    train_window: int = 16
    epochs: int = 30
    macro_mult: int = 1  # T3 drift dial: substeps multiplier on ACCEPTED steps
    seed: int = 20250819
    pretrain_samples: int = 2000
    pretrain_epochs: int = 200


def _finite_or_none(x: float) -> float | None:
    """JSONL sanitation: +inf (empty-window qhat/bound) round-trips as null."""
    return float(x) if math.isfinite(x) else None


def build_prior_committee(cfg: ToyConfig):
    """Build and pre-train the K=4 committee on regime-A (rho=28) attractor
    samples — the MACE-MP-0 foundation-prior analog.  Built ONCE per driver
    invocation and shared by eps calibration and the loop (the calibration
    must measure the committee that actually runs)."""
    from experiments.toy.surrogate import ToyCommittee  # local import: torch is heavy

    committee = ToyCommittee(n_members=4, seed=cfg.seed, epochs=cfg.epochs)
    states, fields = LorenzEngine(RHO_REGIME_A).attractor_samples(
        cfg.pretrain_samples, cfg.seed
    )
    report = committee.pretrain(states, fields, epochs=cfg.pretrain_epochs)
    return committee, report


def run_toy(cfg: ToyConfig, log_path: Path, committee=None, pretrain_report=None) -> dict:
    """Run the closed loop, streaming one JSONL row per step to ``log_path``.

    ``committee`` is the pre-trained prior (built by ``build_prior_committee``
    when not given).  Returns the run summary (counts, timing, fine-tune
    reports); streak and coverage statistics live in the log and are computed
    by ``analyze_rows``.
    """
    if cfg.macro_mult < 1:
        raise ValueError(f"macro_mult must be >= 1, got {cfg.macro_mult}")
    engine = LorenzEngine(cfg.rho)

    t0 = time.perf_counter()
    if committee is None:
        committee, pretrain_report = build_prior_committee(cfg)
        t_pretrain = time.perf_counter() - t0
    else:
        # Built by the caller; the measured wall time rides in the report.
        t_pretrain = pretrain_report.wall_time_s if pretrain_report else 0.0

    switch = ToyConformalSwitch(
        alpha=cfg.alpha,
        eps_acc=cfg.eps_acc,
        window=cfg.window,
        w_min=cfg.w_min,
        delta=cfg.delta,
        streak_rho=cfg.streak_rho,
    )
    state = engine.on_attractor_state(cfg.seed)
    labels: list[tuple[np.ndarray, np.ndarray]] = []
    finetune_reports = []
    n_obs = 0
    timing = {"predict": 0.0, "shadow": 0.0, "oracle": 0.0, "integrate": 0.0, "finetune": 0.0}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t_loop = time.perf_counter()
    with log_path.open("a") as log:
        for step in range(cfg.n_steps):
            t = time.perf_counter()
            mean_field, s = committee.predict(state)
            timing["predict"] += time.perf_counter() - t

            t = time.perf_counter()
            true_field = engine.rhs(state)  # shadow truth; recorded every step
            timing["shadow"] += time.perf_counter() - t
            e = float(np.max(np.abs(mean_field - true_field)))

            streak_k = switch.streak  # position within the current streak
            decision = switch.assess(s, step)
            qhat_at_decision = switch.qhat()  # assess() never touches the window
            state_pre = state  # the state s/e/bound refer to (never mutated in place)
            n_sub = N_SUB_BASE
            finetuned = False
            if decision.route == "dft":
                t = time.perf_counter()
                label = engine.label(state)
                timing["oracle"] += time.perf_counter() - t
                labels.append((state.copy(), label))
                switch.observe(s, e)
                n_obs += 1
                if n_obs % cfg.n_label == 0:  # OnlineUpdater semantics
                    t = time.perf_counter()
                    report = committee.finetune(labels[-cfg.train_window :])
                    timing["finetune"] += time.perf_counter() - t
                    finetune_reports.append(dataclasses.asdict(report))
                    finetuned = True
                t = time.perf_counter()
                state = engine.macro_step(state, n_sub)
                timing["integrate"] += time.perf_counter() - t
            else:
                n_sub = N_SUB_BASE * cfg.macro_mult  # T3 dial: accepted steps only
                t = time.perf_counter()
                for _ in range(n_sub):
                    state = rk4_step(
                        lambda u: committee.predict_members(u).mean(axis=0), state, DT_SUB
                    )
                timing["integrate"] += time.perf_counter() - t

            row = {
                "step": step,
                "route": decision.route,
                "spread": s,
                "error": e,
                "qhat": _finite_or_none(qhat_at_decision),
                "bound": _finite_or_none(decision.score),
                "streak_k": streak_k,
                "state": [float(v) for v in state_pre],
                "finetune": finetuned,
                "n_labels": n_obs,
                "n_sub": n_sub,
            }
            log.write(json.dumps(row) + "\n")
    wall_loop = time.perf_counter() - t_loop

    return {
        "n_steps": cfg.n_steps,
        "n_dft": n_obs,
        "dft_fraction": n_obs / cfg.n_steps,
        "n_finetunes": len(finetune_reports),
        "finetune_reports": finetune_reports,
        "pretrain_report": dataclasses.asdict(pretrain_report) if pretrain_report else None,
        "final_window": switch.window_size,
        "final_qhat": _finite_or_none(switch.qhat()),
        "oracle_label_calls": engine.calls,
        "wall_time_s": wall_loop + t_pretrain,
        "wall_loop_s": wall_loop,
        "wall_pretrain_s": t_pretrain,
        "wall_per_1000_steps_s": 1000.0 * wall_loop / cfg.n_steps,
        "timing_breakdown_s": timing,
        "config": dataclasses.asdict(cfg),
    }


def analyze_rows(rows: list[dict], eps_acc: float, delta: float) -> dict:
    """Streak/coverage statistics from log rows (accepted violations are exact:
    the toy records the true shadow error at every step)."""
    n_ml = sum(r["route"] == "ml" for r in rows)
    violations = [r for r in rows if r["route"] == "ml" and r["error"] > eps_acc]
    # Streaks: maximal runs of consecutive accepted steps.
    streaks: list[list[dict]] = []
    for row in rows:
        if row["route"] == "ml":
            if not streaks or streaks[-1] is None:
                streaks.append([])
            streaks[-1].append(row)
        else:
            if streaks and streaks[-1] is not None:
                streaks.append(None)
    streaks = [s for s in streaks if s is not None]
    # Within-streak drift on the longest streak: r(k) = e/(s+delta) vs k.
    drift = {"slope_per_step": float("nan"), "intercept": float("nan"), "n_points": 0}
    if streaks:
        longest = max(streaks, key=len)
        ks = np.array([r["streak_k"] for r in longest], dtype=float)
        rs = np.array([r["error"] / (r["spread"] + delta) for r in longest])
        if len(longest) > 2:
            slope, intercept = np.polyfit(ks, rs, 1)
            drift = {
                "slope_per_step": float(slope),
                "intercept": float(intercept),
                "n_points": len(longest),
                "streak_start_step": longest[0]["step"],
            }
    return {
        "n_accepted": n_ml,
        "n_violations": len(violations),
        "alpha_hat": len(violations) / max(n_ml, 1),
        "violation_steps": [r["step"] for r in violations],
        "n_streaks": len(streaks),
        "streak_lengths": [len(s) for s in streaks],
        "max_streak": max((len(s) for s in streaks), default=0),
        "drift_longest_streak": drift,
    }


def replay_decisions(log_path: Path, cfg: ToyConfig) -> dict:
    """Decision-layer replay: rebuild the switch from the stored (s, e) stream
    and check every stored route/bound/streak exactly (the toy analog of
    pyraimd2.switch.replay — committee and trajectory are not re-simulated)."""
    switch = ToyConformalSwitch(
        alpha=cfg.alpha,
        eps_acc=cfg.eps_acc,
        window=cfg.window,
        w_min=cfg.w_min,
        delta=cfg.delta,
        streak_rho=cfg.streak_rho,
    )
    n_checked = 0
    mismatches: list[dict] = []
    with log_path.open() as log:
        for line in log:
            row = json.loads(line)
            if switch.streak != row["streak_k"]:
                mismatches.append({"step": row["step"], "field": "streak_k"})
            decision = switch.assess(float(row["spread"]), int(row["step"]))
            expected_bound = row["bound"] if row["bound"] is not None else float("inf")
            if decision.route != row["route"] or decision.score != expected_bound:
                mismatches.append({"step": row["step"], "field": "decision"})
            if decision.route == "dft":
                switch.observe(float(row["spread"]), float(row["error"]))
            n_checked += 1
    return {"n_checked": n_checked, "n_mismatches": len(mismatches), "mismatches": mismatches[:5]}


def calibrate_eps(
    committee, cfg: ToyConfig, n_steps: int, percentile: float
) -> tuple[float, np.ndarray]:
    """eps from the frozen-prior committee's error distribution: roll an
    oracle trajectory on the deployment regime and take the ``percentile``
    of e = max_i |mean_pred_i - true_i| (no labels, no fine-tune — the
    "measurable alpha_hat" operating point is chosen before the run)."""
    engine = LorenzEngine(cfg.rho)
    state = engine.on_attractor_state(cfg.seed + 1)
    errors = np.empty(n_steps)
    for step in range(n_steps):
        mean_field, _ = committee.predict(state)
        errors[step] = float(np.max(np.abs(mean_field - engine.rhs(state))))
        state = engine.macro_step(state)
    return float(np.percentile(errors, percentile)), errors


# -- smoke driver ---------------------------------------------------------------

OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "verm": "#D55E00",
    "ink": "#333333",
    "gray": "#666666",
}


def render_figure(rows: list[dict], eps_acc: float, fig_base: Path) -> None:
    """The fig1(b) analog on real toy data: B(s) and true error vs step with
    the eps budget line, plus a route rug (ML/DFT) with fine-tune markers."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "figure.dpi": 300,
        }
    )
    steps = np.array([r["step"] for r in rows])
    bound = np.array([r["bound"] if r["bound"] is not None else np.nan for r in rows])
    error = np.array([r["error"] for r in rows])
    routes = np.array([r["route"] for r in rows])

    fig, (ax_t, ax_r) = plt.subplots(
        2,
        1,
        figsize=(7.0, 2.9),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.38},
    )
    ax_t.plot(steps, bound, color=OI["blue"], lw=0.9, zorder=3, label="$B(s)$")
    ax_t.plot(
        steps, error, color=OI["verm"], lw=0.5, alpha=0.75, zorder=2, label="true error $e$"
    )
    ax_t.axhline(eps_acc, color=OI["gray"], lw=0.9, ls=(0, (4, 2)), zorder=2)
    over = bound > eps_acc
    ax_t.fill_between(
        steps, eps_acc, bound, where=over, color=OI["orange"], alpha=0.3, lw=0, zorder=1
    )
    viol = (routes == "ml") & (error > eps_acc)
    if viol.any():
        ax_t.plot(
            steps[viol],
            error[viol],
            "x",
            ms=3.5,
            color=OI["verm"],
            zorder=4,
            label="accepted violation",
        )
    finite = np.concatenate([bound[np.isfinite(bound)], error])
    ymax = max(1.3 * eps_acc, 1.15 * float(np.percentile(finite, 99.5)))
    ax_t.set_ylim(0, ymax)
    ax_t.set_ylabel("field error / bound")
    ax_t.text(
        0.995,
        eps_acc - 0.04 * ymax,
        r"accuracy budget $\varepsilon_{\mathrm{acc}}$",
        fontsize=6.5,
        color=OI["ink"],
        ha="right",
        va="top",
        transform=ax_t.get_yaxis_transform(),
    )
    ax_t.legend(loc="upper left")
    ax_t.set_title("(a) conformal bound vs true error (toy smoke, regime A)", loc="left")
    for sp in ("top", "right"):
        ax_t.spines[sp].set_visible(False)

    # Route rug: merged same-route segments, ML blue / DFT orange.
    segments = []
    start = 0
    for i in range(1, len(routes) + 1):
        if i == len(routes) or routes[i] != routes[start]:
            segments.append((start, i - start, routes[start]))
            start = i
    for seg_start, seg_len, seg_route in segments:
        ax_r.broken_barh(
            [(seg_start, seg_len)],
            (0, 1),
            facecolors=OI["blue"] if seg_route == "ml" else OI["orange"],
        )
    finetune_steps = [r["step"] for r in rows if r["finetune"]]
    ax_r.plot(
        finetune_steps,
        [0.5] * len(finetune_steps),
        marker="v",
        ms=3.5,
        color=OI["green"],
        mec="none",
        ls="none",
        zorder=4,
    )
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([])
    ax_r.set_xlabel("toy MD step")
    ax_r.set_ylabel("route", rotation=0, ha="right", va="center")
    for sp in ("top", "right", "left"):
        ax_r.spines[sp].set_visible(False)
    rug_handles = [
        mpl.patches.Patch(color=OI["blue"], label="ML"),
        mpl.patches.Patch(color=OI["orange"], label="DFT"),
        mpl.lines.Line2D(
            [], [], marker="v", ms=3.5, color=OI["green"], mec="none", ls="none",
            label="fine-tune",
        ),
    ]
    ax_r.legend(handles=rug_handles, loc="center left", bbox_to_anchor=(1.005, 0.5))
    ax_r.set_title("(b) route rug", loc="left")

    fig.align_ylabels()
    fig_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(fig_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regime", choices=["A", "B"], default="A",
                   help="A: rho=28 (training distribution); B: rho=35 (deployment shift)")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--eps-acc", type=float, default=None,
                   help="error budget; default: calibrate at --eps-percentile of the "
                        "frozen-prior committee's error distribution")
    p.add_argument("--eps-percentile", type=float, default=60.0)
    p.add_argument("--streak-rho", type=float, default=0.0,
                   help="streak-inflation rate (0 = plain bound, the T1/T2 measurement)")
    p.add_argument("--macro-mult", type=int, default=1,
                   help="T3 drift dial: substep multiplier on accepted steps (1/2/4)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--w-min", type=int, default=16)
    p.add_argument("--n-label", type=int, default=8)
    p.add_argument("--train-window", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain-samples", type=int, default=2000)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=20250819)
    p.add_argument("--run-id", default="smoke")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/toy"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ToyConfig(
        rho=RHO_REGIME_A if args.regime == "A" else RHO_REGIME_B,
        n_steps=args.n_steps,
        eps_acc=args.eps_acc if args.eps_acc is not None else 0.0,  # set below
        alpha=args.alpha,
        window=args.window,
        w_min=args.w_min,
        streak_rho=args.streak_rho,
        n_label=args.n_label,
        train_window=args.train_window,
        epochs=args.epochs,
        macro_mult=args.macro_mult,
        seed=args.seed,
        pretrain_samples=args.pretrain_samples,
        pretrain_epochs=args.pretrain_epochs,
    )

    # Engine gate: one RK4 step at dt=0.005 must match a dt-converged
    # reference to < 1e-6 (spec wording: "over one step" of the integrator;
    # the 2-substep macro-step error is recorded alongside for transparency).
    check_states, _ = LorenzEngine(cfg.rho).attractor_samples(8, cfg.seed + 2, stride=50)
    rk4_max_err = verify_rk4(check_states, rho=cfg.rho, n_sub=1)
    rk4_macro_err = verify_rk4(check_states, rho=cfg.rho, n_sub=N_SUB_BASE)
    print(f"RK4 verification: 1 substep max|err| = {rk4_max_err:.3e} "
          f"(macro-step {rk4_macro_err:.3e})")
    if rk4_max_err >= 1e-6:
        raise RuntimeError(f"RK4 engine failed its convergence gate: {rk4_max_err:.3e} >= 1e-6")

    eps_info = {"mode": "fixed", "eps_acc": args.eps_acc}
    committee, pretrain_report = build_prior_committee(cfg)
    if args.eps_acc is None:
        eps, cal_errors = calibrate_eps(committee, cfg, args.n_steps, args.eps_percentile)
        cfg.eps_acc = eps
        eps_info = {
            "mode": "percentile",
            "percentile": args.eps_percentile,
            "eps_acc": eps,
            "calibration_error_quantiles": {
                str(q): float(np.percentile(cal_errors, q)) for q in (50, 60, 90, 99, 100)
            },
        }
        print(f"eps_acc calibrated at p{args.eps_percentile:.0f} of prior-committee "
              f"errors: {eps:.5f}")

    log_path = args.out_dir / f"toy_log_{args.run_id}.jsonl"
    if log_path.exists():
        log_path.unlink()  # the store is append-only; a fresh run-id starts fresh
    summary = run_toy(cfg, log_path, committee=committee, pretrain_report=pretrain_report)

    rows = [json.loads(line) for line in log_path.open()]
    analysis = analyze_rows(rows, cfg.eps_acc, cfg.delta)
    replay = replay_decisions(log_path, cfg)
    if replay["n_mismatches"] != 0:
        raise RuntimeError(f"decision replay mismatch: {replay['mismatches']}")

    fig_base = args.out_dir / f"fig_toy_{args.run_id}"
    render_figure(rows, cfg.eps_acc, fig_base)

    report = {
        "run_id": args.run_id,
        "regime": args.regime,
        "rk4_verification_max_err": rk4_max_err,
        "rk4_macro_step_max_err": rk4_macro_err,
        "eps": eps_info,
        "replay": replay,
        **analysis,
        **summary,
    }
    out = args.out_dir / (
        "smoke_report.json" if args.run_id == "smoke" else f"smoke_report_{args.run_id}.json"
    )
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "finetune_reports"}, indent=2))
    print(f"log -> {log_path}\nreport -> {out}\nfigure -> {fig_base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
