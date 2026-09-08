"""One preregistered, NumPy-only verification stress study; no MD or training.

Run from the repository root:
    uv run --no-sync python experiments/toy/verification_thinning.py --check-only
    uv run --no-sync python experiments/toy/verification_thinning.py --run

The run refuses to overwrite a started study, including a failed study.
All settings are frozen in the adjacent analysis protocol before sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import signal
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/revision_20260905/verification_thinning"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, obj: object) -> None:
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write("\n")


def exact_checks() -> dict:
    """No RNG: exact multipliers and a full adaptive finite audit tree."""
    checked = 0
    for p in (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)):
        for r in (Fraction(1, 2), Fraction(1, 3), Fraction(3, 4)):
            c = 1 - p + p * r
            for a in (0, 1):
                for y in (0, 1):
                    b = a * y
                    expected = sum(
                        prob * r ** (b * z) / c**b
                        for z, prob in ((0, 1 - p), (1, p))
                    )
                    assert expected == 1, (p, r, a, y, expected)
                    checked += 1
    p, r = Fraction(1, 10), Fraction(1, 2)
    c = 1 - p + p * r
    assert c == Fraction(19, 20)
    # Each leaf contains probability, finite-lambda M, infinity M, last detection.
    leaves = [(Fraction(1), Fraction(1), Fraction(1), 0)]
    depths = []
    for t in range(10):
        next_leaves = []
        for weight, m, m_inf, last_detection in leaves:
            a = int(t % 4 != 0 or last_detection == 1)
            y = int((t % 3 != 1) != bool(last_detection))
            b = a * y
            for z, prob in ((0, 1 - p), (1, p)):
                f = b * z
                next_leaves.append(
                    (
                        weight * prob,
                        m * r**f / c**b,
                        Fraction(0) if f else m_inf / (1 - p) ** b,
                        f,
                    )
                )
        leaves = next_leaves
        assert sum(w for w, _, _, _ in leaves) == 1
        assert sum(w * m for w, m, _, _ in leaves) == 1
        assert sum(w * m_inf for w, _, m_inf, _ in leaves) == 1
        depths.append({"depth": t + 1, "leaves": len(leaves), "both_means": "1"})
    return {
        "status": "passed",
        "exact_binary_multiplier_cases": checked,
        "adaptive_tree": depths,
        "primary_c_exact": str(c),
        "interpretation": "Exact finite checks complement the general conditional proof.",
    }


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Equal-tail exact binomial interval, via stable CDF inversion; no SciPy."""
    if not 0 <= k <= n or n < 1:
        raise ValueError((k, n))
    tail = (1.0 - confidence) / 2.0

    def invert_cdf(last: int, target: float) -> float:
        j = np.arange(last + 1, dtype=np.float64)
        log_choose = np.array(
            [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             for i in range(last + 1)]
        )
        lo, hi = 0.0, 1.0
        for _ in range(64):
            p = (lo + hi) / 2.0
            terms = log_choose + j * math.log(p) + (n - j) * math.log1p(-p)
            peak = float(np.max(terms))
            cdf = math.exp(peak) * float(np.exp(terms - peak).sum())
            if cdf > target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2.0

    if k == 0:
        return [0.0, -math.expm1(math.log(tail) / n)]
    if k == n:
        return [math.exp(math.log(tail) / n), 1.0]
    return [invert_cdf(k - 1, 1.0 - tail), invert_cdf(k, tail)]


def event_summary(values: np.ndarray, comparator: float | None = None) -> dict:
    k, n = int(values.sum()), int(values.size)
    interval = clopper_pearson(k, n)
    result = {
        "events": k,
        "independent_replicates": n,
        "proportion": k / n,
        "ci95_clopper_pearson": interval,
    }
    if comparator is not None:
        result["theoretical_upper_bound"] = comparator
        result["point_estimate_at_or_below_bound"] = k / n <= comparator
        result["mc_interpretation"] = (
            "upper CI below bound: evidence of conservativeness in this process"
            if interval[1] <= comparator
            else "lower CI above bound: discrepancy to investigate; no tuning"
            if interval[0] > comparator
            else "CI contains bound: inconclusive at this Monte Carlo precision"
        )
    return result


def mean_summary(values: np.ndarray) -> dict:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    half = 1.959963984540054 * se
    return {
        "mean": mean,
        "ci95_normal_independent_replicates": [mean - half, mean + half],
        "standard_error": se,
        "independent_replicates": int(values.size),
    }


def simulate(cfg: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    n, steps, batch = cfg["replicates"], cfg["steps"], cfg["batch_size"]
    p, lam, eta = cfg["p"], math.log(2), cfg["eta"]
    log_scale = -math.log1p(p * math.expm1(-lam))
    threshold = math.log(1 / eta)
    v_zero = math.ceil(math.log(eta) / math.log1p(-p))
    integer_names = (
        "N", "V", "D", "audit_labels", "first_crossing_step", "first_zero_event_step"
    )
    data = {name: np.zeros(n, dtype=np.int64) for name in integer_names}
    for name in ("ever_crossed", "ever_capped_bound_failed", "ever_zero_event"):
        data[name] = np.zeros(n, dtype=bool)
    for name in ("max_log_M", "terminal_log_M", "terminal_capped_bound"):
        data[name] = np.zeros(n, dtype=float)
    data["first_crossing_step"].fill(-1)
    data["first_zero_event_step"].fill(-1)
    traces = []
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        size = stop - start
        # Independent replicate streams; columns are unexposed until their turn.
        uniforms = np.empty((size, steps, 3))
        for j, seed_id in enumerate(range(start, stop)):
            seed = np.random.SeedSequence(cfg["seed_entropy"], spawn_key=(seed_id,))
            rng = np.random.Generator(np.random.PCG64(seed))
            uniforms[j] = rng.random((steps, 3))
        q = np.full(size, cfg["initial_q"])
        trust = np.full(size, cfg["initial_trust"])
        cooldown = np.full(size, cfg["initial_cooldown"], dtype=np.int64)
        view = {key: value[start:stop] for key, value in data.items()}
        for t in range(steps):
            risk_prob = np.clip(q + 0.15 * ((t // 64) % 2), 0.01, 0.95)
            accept_prob = np.clip(trust - 0.35 * (cooldown > 0), 0.05, 0.98)
            y = uniforms[:, t, 0] < risk_prob
            a = uniforms[:, t, 1] < accept_prob
            # A, Y, and the current force choice are fixed BEFORE revealing Z.
            z = uniforms[:, t, 2] < p
            h = a & z
            f = h & y
            clean = h & ~y
            view["N"] += a
            view["V"] += a & y  # Current audited steps remain surrogate steps.
            view["D"] += f
            view["audit_labels"] += h
            log_m = log_scale * view["V"] - lam * view["D"]
            capped_bound = np.minimum(view["N"], (lam * view["D"] + threshold) / log_scale)
            crossed = log_m >= threshold
            zero_event = (view["D"] == 0) & (view["V"] >= v_zero)
            first = crossed & ~view["ever_crossed"]
            first_zero = zero_event & ~view["ever_zero_event"]
            view["first_crossing_step"][first] = t + 1
            view["first_zero_event_step"][first_zero] = t + 1
            view["ever_crossed"] |= crossed
            view["ever_capped_bound_failed"] |= view["V"] > capped_bound
            view["ever_zero_event"] |= zero_event
            np.maximum(view["max_log_M"], log_m, out=view["max_log_M"])
            if start == 0:
                for j in range(cfg["trace_replicates"]):
                    traces.append({
                        "replicate": j, "step": t + 1,
                        "q_before": float(q[j]), "trust_before": float(trust[j]),
                        "cooldown_before": int(cooldown[j]),
                        "risk_probability": float(risk_prob[j]),
                        "acceptance_probability": float(accept_prob[j]),
                        "A": int(a[j]), "Y": int(y[j]), "Z": int(z[j]),
                        "used_surrogate": int(a[j]), "detected_failure": int(f[j]),
                        "N": int(view["N"][j]), "V": int(view["V"][j]),
                        "D": int(view["D"][j]), "log_M": float(log_m[j]),
                        "capped_bound": float(capped_bound[j]),
                    })
            # Label feedback affects ONLY future states. No latent unaudited Y here.
            q = np.clip(q + 0.004 + 0.006 * a - 0.20 * f + 0.01 * clean, 0.01, 0.80)
            trust = np.clip(trust + 0.002 + 0.025 * clean - 0.30 * f, 0.10, 0.98)
            cooldown = np.where(f, 8, np.maximum(cooldown - 1, 0))
        view["terminal_log_M"][:] = log_m
        view["terminal_capped_bound"][:] = capped_bound
        assert np.all(view["D"] <= view["V"])
        assert np.all(view["V"] <= view["N"])
        assert np.all(view["D"] <= view["audit_labels"])
        assert np.all(view["N"] > 0)
        print(f"Completed {stop}/{n} independent replicates", flush=True)
    return data, traces


def zero_label_example() -> dict:
    n = 64
    visible_safe = {"acceptance": [1] * n, "surrogate_force": [0.0] * n,
                    "surrogate_score": [0.5] * n, "reference_labels": []}
    visible_unsafe = {"acceptance": [1] * n, "surrogate_force": [0.0] * n,
                      "surrogate_score": [0.5] * n, "reference_labels": []}
    assert visible_safe == visible_unsafe
    return {
        "accepted_steps": n,
        "reference_labels": 0,
        "identical_observable_records": True,
        "world_safe": {"force_budget": 1, "true_force": 0, "V": 0, "V_over_N": 0},
        "world_unsafe": {"force_budget": 1, "true_force": 2, "V": n, "V_over_N": 1},
        "scope": "No identifying physical assumptions: the observation record alone cannot identify risk.",
    }


def save_and_summarize(cfg: dict, data: dict, traces: list[dict]) -> dict:
    p, eta, lam = cfg["p"], cfg["eta"], math.log(2)
    log_scale = -math.log1p(p * math.expm1(-lam))
    v_zero = math.ceil(math.log(eta) / math.log1p(-p))
    with (OUT / "replicates.csv").open("x", newline="") as handle:
        fields = ["replicate", "seed_entropy", "seed_spawn_key", *data]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(cfg["replicates"]):
            writer.writerow({"replicate": i, "seed_entropy": cfg["seed_entropy"],
                             "seed_spawn_key": f"({i},)",
                             **{key: value[i].item() for key, value in data.items()}})
    with (OUT / "traces_first_four.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    n, v, d = data["N"], data["V"], data["D"]
    result = {
        "protocol_id": cfg["protocol_id"],
        "replicates": cfg["replicates"],
        "steps_per_replicate": cfg["steps"],
        "total_simulated_steps": cfg["replicates"] * cfg["steps"],
        "seed_entropy": cfg["seed_entropy"],
        "seed_spawn_key_range": [0, cfg["replicates"] - 1],
        "p": p, "lambda": lam, "eta": eta, "c": math.exp(-log_scale),
        "primary_ever_log_M_crossing": event_summary(data["ever_crossed"], eta),
        "ever_capped_bound_failure": event_summary(data["ever_capped_bound_failed"], eta),
        "zero_detection": {"v_threshold": v_zero,
                           **event_summary(data["ever_zero_event"], (1 - p) ** v_zero)},
        "terminal_true_V_over_N": mean_summary(v / n),
        "terminal_naive_D_over_N": mean_summary(d / n),
        "terminal_adjusted_D_over_pN_diagnostic": mean_summary(d / (p * n)),
        "terminal_paired_undercount_gap": mean_summary((v - d) / n),
        "terminal_capped_bound_over_N": mean_summary(data["terminal_capped_bound"] / n),
        "terminal_naive_strictly_undercounts": event_summary(d < v),
        "terminal_N": mean_summary(n),
        "terminal_V": mean_summary(v),
        "terminal_D_and_toy_retraining_events": mean_summary(d),
        "terminal_audit_labels": mean_summary(data["audit_labels"]),
        "terminal_mean_M_descriptive_not_validity_test": float(np.exp(data["terminal_log_M"]).mean()),
        "zero_label_nonidentifiability": zero_label_example(),
        "limitations": [
            "Independent seeded pseudorandom simulations; time steps are not replicate units.",
            "Clopper-Pearson intervals are separate marginal 95% intervals, not simultaneous.",
            "Finite horizon and one fixed feedback process do not prove an anytime theorem.",
            "Adjusted D/(p*N) is a descriptive diagnostic, not an unbiased-ratio or safety claim.",
            "Only retrospective surrogate force-budget violations are bounded.",
        ],
    }
    write_json(OUT / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    checks = exact_checks()  # A failure stops before ANY Monte Carlo RNG is made.
    # Independent known exact CI values, without a second stochastic study.
    assert np.allclose(clopper_pearson(0, 10), [0, 1 - 0.025**0.1], atol=1e-12)
    assert np.allclose(clopper_pearson(10, 10), [0.025**0.1, 1], atol=1e-12)
    assert np.allclose(clopper_pearson(1, 2), [1 - math.sqrt(0.975), math.sqrt(0.975)], atol=1e-12)
    checks["clopper_pearson_known_values"] = "passed: (0,10), (10,10), (1,2)"
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return
    cfg = json.loads((OUT / "protocol.json").read_text())
    assert cfg["lambda"] == "log(2)"
    files = [Path(__file__).resolve(), OUT / "protocol.json", OUT / "protocol.md"]
    manifest = {
        "status": "started", "started_utc": utc_now(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "command": "uv run --no-sync python experiments/toy/verification_thinning.py --run",
        "sha256_before_sampling": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in files},
        "config": cfg,
    }
    write_json(OUT / "run_started.json", manifest)
    started = time.perf_counter()

    def deadline(_signum: int, _frame: object) -> None:
        raise TimeoutError("Preregistered runtime limit reached; do not rerun/tune.")

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    # Leave 5 seconds within the user ceiling for interpreter/uv startup and exit.
    signal.setitimer(signal.ITIMER_REAL, cfg["runtime_limit_seconds"] - 5)
    try:
        write_json(OUT / "deterministic_checks.json", checks)
        data, traces = simulate(cfg)
        result = save_and_summarize(cfg, data, traces)
        elapsed = time.perf_counter() - started
        write_json(OUT / "run_completed.json", {
            "status": "completed", "completed_utc": utc_now(),
            "wall_seconds": elapsed, "independent_replicates": cfg["replicates"],
            "sha256_after_sampling": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in files},
        })
        print(json.dumps({"wall_seconds": elapsed,
                          "primary": result["primary_ever_log_M_crossing"],
                          "zero_detection": result["zero_detection"],
                          "true_rate": result["terminal_true_V_over_N"],
                          "naive_rate": result["terminal_naive_D_over_N"]}, indent=2))
    except BaseException as exc:
        write_json(OUT / "run_failed.json", {
            "status": "failed", "failed_utc": utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "action": "Report failure; no automatic rerun or parameter change.",
        })
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
