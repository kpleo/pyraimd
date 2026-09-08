"""One bounded audit of frozen samples and saved oscillator intervals.

uv run --no-sync python -B experiments/verify_thinning_independent_20260905.py

Does not import either original experiment, create new seeds, or propagate an MD
trajectory. Outputs are exclusive-create; an existing started manifest stops it.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import csv
import hashlib
import json
import math
import resource
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "analysis/revision_20260905/verification_thinning"
OSC = ROOT / "analysis/revision_20260905/prospective_horizon"
OUT = ROOT / "analysis/execution_20260905/statistics"
REPORT = ROOT / "docs/execution_20260905/statistics_results.md"
CPU_START = time.process_time()
WALL_START = time.monotonic()


def cpu() -> float:
    return time.process_time() - CPU_START


def dump(name: str, data: object) -> None:
    with (OUT / name).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(name: str, records: list[dict]) -> None:
    with (OUT / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pair_sum(values: np.ndarray) -> float:
    """Explicit balanced nonnegative reduction; <= ceil(log2(n)) additions/path."""
    a = np.ravel(values)
    while a.size > 1:
        n = a.size // 2
        b = a[:2*n:2] + a[1:2*n:2]
        a = np.concatenate((b, a[-1:])) if a.size % 2 else b
    return float(a[0]) if a.size else 0.0


def shift_cap(a: np.ndarray, dq: int, dt: int) -> np.ndarray:
    """Integer-grid transition, merging all overshoots into the capped cell."""
    b = np.zeros_like(a)
    b[:, dt:, :] = a[:, :-dt, :]
    b[:, -1, :] += a[:, -dt:, :].sum(axis=1)
    c = np.zeros_like(a)
    c[dq:, :, :] = b[:-dq, :, :]
    c[-1, :, :] += b[-dq:, :, :].sum(axis=0)
    return c


def deterministic_probability() -> dict:
    """Forward absorbed mass recursion in (q/0.002, trust/0.001, V<29)."""
    started = cpu()
    mass = np.zeros((226, 181, 29), dtype=np.float64)
    mass[0, 0, 0] = 1.0
    qi = np.arange(350, 801, 2, dtype=np.int64)[:, None]
    ai = np.arange(800, 981, dtype=np.int64)[None, :]
    success = failure = 0.0
    history = []
    saturated = False
    max_conservation_error = max_weighted_error = 0.0
    for t in range(1024):
        phase = (t // 64) % 2
        if not saturated:
            ri = qi + 150 * phase
            ar = ai * ri
            numerators = ((1000-ai)*10000 + np.zeros_like(ri),
                          ai*(1000-ri)*9, ai*(1000-ri), ar*9, ar)
            assert np.all(sum(numerators) == 10_000_000)
            probs = [a[..., None] / 10_000_000 for a in numerators]
            bad_unchecked = mass * probs[3]
            gain = pair_sum(bad_unchecked[:, :, -1])
            killed = pair_sum(mass * probs[4])
            new = shift_cap(mass * probs[0], 2, 2)
            new += shift_cap(mass * probs[1], 5, 2)
            new += shift_cap(mass * probs[2], 10, 27)
            new[:, :, 1:] += shift_cap(bad_unchecked[:, :, :-1], 5, 2)
            mass = new
            if t == 112:
                # Minimum increments force q=0.80 and trust=0.98 by this point.
                assert np.count_nonzero(mass[:-1]) == 0
                assert np.count_nonzero(mass[-1, :-1]) == 0
                mass = mass[-1, -1].copy()
                saturated = True
        else:
            ar = 980 * (800 + 150 * phase)
            bad_unchecked = mass * (9 * ar / 10_000_000)
            gain = float(bad_unchecked[-1])
            killed = pair_sum(mass * (ar / 10_000_000))
            new = mass * ((1_000_000-ar) / 1_000_000)
            new[1:] += bad_unchecked[:-1]
            mass = new
        success += gain
        failure += killed
        live = pair_sum(mass)
        by_v = mass if saturated else mass.sum(axis=(0, 1))
        weighted = success / .9**29 + pair_sum(by_v / .9**np.arange(29))
        max_conservation_error = max(max_conservation_error, abs(success+failure+live-1))
        max_weighted_error = max(max_weighted_error, abs(weighted-1))
        history.append(dict(step=t+1, success_increment=gain, cumulative_success=success,
                            cumulative_failure=failure, surviving_mass=live,
                            weighted_conservation=weighted, state_slots=int(mass.size)))
        if t in (31, 63, 95, 112, 1023):
            print(f"recursion step={t+1}, P(E0)={success:.16g}, CPU={cpu():.2f}s", flush=True)
    save_csv("absorbed_probability_by_step.csv", history)
    # No interval subtraction at machine precision: express the tiny finite-time
    # gap explicitly using an EXACT rational binomial numerator/denominator.
    with localcontext() as ctx:
        ctx.prec = 90
        n = 1024 - 113
        numerator = sum(math.comb(n, j) * 98**j * 27**(n-j) for j in range(29))
        denominator = 125**n
        missing_q = Decimal(numerator) / Decimal(denominator)
        bound = Decimal(9)**29 / Decimal(10)**29
        gap = bound * missing_q
        analytic = dict(exact_upper=str(bound),
                        finite_horizon_gap_upper_decimal_approx=str(gap),
                        finite_horizon_gap_upper_exact_numerator=str(9**29*numerator),
                        finite_horizon_gap_upper_exact_denominator=str(10**29*denominator),
                        tilted_missing_probability_upper_decimal_approx=str(missing_q),
                        expression="B - delta <= P(E0) <= B; B=(9/10)^29; "
                                   "delta=B*sum_{j=0}^{28} C(911,j)98^j27^(911-j)/125^911")
    # Positive arithmetic: <=128 rounded operations on any contribution per
    # time step, including <=27/10 capped merges and balanced reductions.
    # This deliberately overbounds operation paths rather than inferring error
    # from conservation checks. Absolute subnormal losses are <1e-310 overall.
    u = 2.0**-53
    operations = 128 * 1024
    numerical_bound = operations*u/(1-operations*u) + 1e-310
    # Same-coin coupling: q/trust updates have <=8t*u error each before a
    # differing coin comparison; clip is 1-Lipschitz. Sum the comparison-error
    # bounds over 1024 steps. This is conservative even without saturation.
    source_float_bound = 32 * 1024 * 1025 * u
    result = dict(method="deterministic finite-state absorbed forward probability",
                  p_event_float64=success, max_dense_state_slots=226*181*29,
                  saturated_after_steps=113, finite_horizon=1024,
                  max_mass_conservation_error=max_conservation_error,
                  max_weighted_conservation_error=max_weighted_error,
                  conservative_recursion_absolute_roundoff_bound=numerical_bound,
                  conservative_source_float_vs_integer_absolute_probability_bound=source_float_bound,
                  analytic_finite_horizon_bracket=analytic, cpu_seconds=cpu()-started)
    dump("deterministic_probability.json", result)
    return result


def reconstruct_original_samples() -> dict:
    """All original keys; raw PCG64 bits -> uniforms -> independent scalar event."""
    started = cpu()
    data = rows(OLD / "replicates.csv")
    assert len(data) == 20000
    traces = {(int(r["replicate"]), int(r["step"])): r
              for r in rows(OLD / "traces_first_four.csv")}
    output = []
    mismatches = []
    trace_mismatches = []
    checked_trace_steps = 0
    total_steps = 0
    max_grid_deviation = 0.0
    grid_coin_mismatches = 0
    for i, old in enumerate(data):
        assert int(old["replicate"]) == i and old["seed_spawn_key"] == f"({i},)"
        assert int(old["seed_entropy"]) == 20260905
        bits = np.random.PCG64(np.random.SeedSequence(20260905, spawn_key=(i,))).random_raw(3072)
        q, trust, qint, tint = .35, .8, 350, 800
        accepted = violations = audits = 0
        event_step = detected_step = -1
        decision_hash = hashlib.sha256()
        for t in range(1024):
            risk = min(.95, max(.01, q + .15 * ((t // 64) % 2)))
            ay = int(bits[3*t]) >> 11
            aa = int(bits[3*t+1]) >> 11
            az = int(bits[3*t+2]) >> 11
            y = ay * 2.0**-53 < risk
            a = aa * 2.0**-53 < min(.98, max(.05, trust))
            z = az * 2.0**-53 < .1
            # Exact integer comparisons audit whether source float thresholds
            # would change any original pre-absorption coin decision.
            yi = ay*1000 < (qint+150*((t//64)%2))*2**53
            ai = aa*1000 < tint*2**53
            zi = az*10 < 2**53
            grid_coin_mismatches += int((a, y, z) != (ai, yi, zi))
            max_grid_deviation = max(max_grid_deviation, abs(q-qint/1000), abs(trust-tint/1000))
            accepted += a
            violations += a and y
            audits += a and z
            fail = a and y and z
            clean = a and z and not y
            decision_hash.update(bytes((int(a), int(y), int(z))))
            if i < 4:
                saved = traces[(i, t+1)]
                checked_trace_steps += 1
                actual = dict(A=int(a), Y=int(y), Z=int(z), N=accepted, V=violations,
                              D=int(fail), used_surrogate=int(a), detected_failure=int(fail))
                for key, value in actual.items():
                    if value != int(saved[key]):
                        trace_mismatches.append(dict(replicate=i, step=t+1, field=key,
                                                     original=saved[key], reconstructed=value))
                for key, value in (("q_before", q), ("trust_before", trust),
                                   ("risk_probability", risk)):
                    if value != float(saved[key]):
                        trace_mismatches.append(dict(replicate=i, step=t+1, field=key,
                                                     original=saved[key], reconstructed=value))
            if fail:
                detected_step = t+1
                break
            if violations == 29:
                event_step = t+1
                break
            # Preserve the scalar source expression's floating operation order.
            q = min(.80, max(.01, q + .004 + .006*a - .20*fail + .01*clean))
            trust = min(.98, max(.10, trust + .002 + .025*clean - .30*fail))
            qint = min(800, qint + 4 + 6*a + 10*clean)
            tint = min(980, tint + 2 + 25*clean)
        total_steps += t+1
        matches = ((event_step > 0) == (old["ever_zero_event"] == "True")
                   and event_step == int(old["first_zero_event_step"]))
        record = dict(replicate=i, seed_entropy=20260905, seed_spawn_key=f"({i},)",
                      original_ever_zero_event=old["ever_zero_event"],
                      original_first_zero_event_step=int(old["first_zero_event_step"]),
                      reconstructed_ever_zero_event=event_step > 0,
                      reconstructed_first_zero_event_step=event_step,
                      first_detection_if_before_success=detected_step,
                      inspected_prefix_steps=t+1, N_at_absorption=accepted,
                      V_at_absorption=violations, D_at_absorption=int(detected_step > 0),
                      audit_labels_at_absorption=audits, matches_original=matches,
                      inspected_AYZ_sha256=decision_hash.hexdigest(),
                      original_stream_uint64le_sha256=hashlib.sha256(bits.astype("<u8", copy=False).tobytes()).hexdigest())
        output.append(record)
        if not matches:
            mismatches.append(record)
        if (i+1) % 5000 == 0:
            print(f"original streams audited={i+1}/20000, mismatches={len(mismatches)}, CPU={cpu():.2f}s", flush=True)
    save_csv("original_seed_event_reconstruction.csv", output)
    dump("event_mismatches.json", mismatches)
    dump("retained_trace_prefix_mismatches.json", trace_mismatches)
    k = sum(r["reconstructed_ever_zero_event"] for r in output)
    with localcontext() as ctx:
        ctx.prec = 65
        p = (Decimal(9)/10)**29
        term = Decimal(math.comb(20000, k)) * p**k * (1-p)**(20000-k)
        tail = term
        for j in range(k, 20000):
            term *= Decimal(20000-j)/Decimal(j+1)*p/(1-p)
            tail += term
    result = dict(original_seeds=20000, new_seeds=0, repeated_stochastic_studies=0,
                  inspected_scalar_steps=total_steps, source_generator="NumPy PCG64 / original SeedSequence",
                  independent_uniform_conversion="(uint64 >> 11) * 2**-53; no Generator.random",
                  scope="Every original seed until first detected violation or 29th undetected violation; "
                        "no event can first occur after a cumulative detection. Terminal/main endpoint not replayed.",
                  original_zero_count=sum(r["ever_zero_event"] == "True" for r in data),
                  reconstructed_zero_count=k, mismatch_count=len(mismatches),
                  retained_trace_prefix_steps_checked=checked_trace_steps,
                  retained_trace_prefix_mismatch_count=len(trace_mismatches),
                  integer_grid_coin_decision_mismatches=grid_coin_mismatches,
                  max_original_state_integer_grid_absolute_difference=max_grid_deviation,
                  posthoc_binomial_upper_tail_at_exact_bound=str(tail),
                  unresolved="No mismatch does not establish why the old sample proportion exceeds the bound; "
                             "PRNG/SeedSequence are shared infrastructure, not independently certified.",
                  cpu_seconds=cpu()-started)
    dump("original_sample_audit.json", result)
    return result


def oscillator_diagnostic() -> dict:
    """Endpoint work on saved intervals, including reference-assisted intervals."""
    started = cpu()
    specs = {(r["split"], r["policy"], r["seed"]): r for r in rows(OSC / "trajectories.csv")}
    groups = defaultdict(list)
    for row in rows(OSC / "steps.csv"):
        groups[(row["split"], row["policy"], row["seed"])].append(row)
    summaries, segments, updates = [], [], []
    for key, data in groups.items():
        spec = specs[key]
        assert len(data) == 2000 and [int(r["step"]) for r in data] == list(range(2000))
        k = float(spec["stiffness"])
        x0, v0 = float(spec["x0"]), float(spec["v0"])
        energy0 = .5*(v0*v0+k*x0*x0)
        work, jumps, endpoint_errors, energy_errors = [], [], [], []
        positive_power_updates = negative_power_updates = 0
        contraction_errors = []
        free_work = reference_work = 0.0
        seg_start, seg_work = 0, 0.0
        trajectory_updates = []
        last_bias = float(data[0]["bias_used"])
        for j, row in enumerate(data):
            x, v, b = (float(row[n]) for n in ("x", "v", "bias_used"))
            old_b = float(row["bias_before"])
            assert old_b == (0.0 if j == 0 else float(data[j-1]["bias_used"]))
            if j < 1999:
                xn, vn = float(data[j+1]["x"]), float(data[j+1]["v"])
            else:
                # The final endpoint is stored as errors relative to the exact
                # reference solution. Decode it; do not call an advance routine.
                angle = math.sqrt(k) * float(row["time_end"])
                xn = x0*math.cos(angle)+v0/math.sqrt(k)*math.sin(angle)+float(row["position_error_at_end"])
                vn = -x0*math.sqrt(k)*math.sin(angle)+v0*math.cos(angle)+float(row["velocity_error_at_end"])
            w = .5*(k-1)*(xn*xn-x*x)+b*(xn-x)
            dh = .5*(vn*vn+k*xn*xn-v*v-k*x*x)
            work.append(w)
            endpoint_errors.append(abs(w-dh))
            energy_errors.append(abs((.5*(vn*vn+k*xn*xn)-energy0)/energy0-float(row["relative_energy_change_at_end"])))
            if row["route"] == "unreferenced":
                free_work += w
                assert b == old_b
            else:
                reference_work += w
                pre = v*((k-1)*x+old_b)
                post = v*((k-1)*x+b)
                positive_power_updates += post > 1e-14
                negative_power_updates += post < -1e-14
                full = abs((k-1)*x+b) < 1e-13
                if not full:
                    contraction_errors.append(abs(post-.2*pre))
                rec = dict(split=key[0], policy=key[1], seed=key[2], step=j,
                           time=float(row["time_start"]), route=row["route"], x=x, v=v,
                           bias_before=old_b, bias_after=b, power_before=pre, power_after=post,
                           power_change=post-pre, potential_jump_at_x_zero_gauge=-(b-old_b)*x,
                           included_in_telescoping_sum=j > 0, full_correction=full)
                updates.append(rec)
                trajectory_updates.append(rec)
                if j > 0:
                    jumps.append(-(b-old_b)*x)
                    segments.append(dict(split=key[0], policy=key[1], seed=key[2],
                                         start_step=seg_start, end_step_exclusive=j,
                                         bias=last_bias, work=seg_work, work_over_H0=seg_work/energy0))
                    seg_start, seg_work = j, 0.0
            seg_work += w
            last_bias = b
        segments.append(dict(split=key[0], policy=key[1], seed=key[2], start_step=seg_start,
                             end_step_exclusive=2000, bias=last_bias, work=seg_work,
                             work_over_H0=seg_work/energy0))
        boundary = (.5*(1-k)*x0*x0-float(data[0]["bias_used"])*x0
                    -(.5*(1-k)*xn*xn-last_bias*xn))
        total = math.fsum(work)
        jump_sum = math.fsum(jumps)
        final_energy = .5*(vn*vn+k*xn*xn)-energy0
        summaries.append(dict(split=key[0], policy=key[1], seed=key[2], H0=energy0,
                              n_intervals=2000, n_reference_updates=len(trajectory_updates),
                              W=total, W_over_H0=total/energy0, delta_H=final_energy,
                              original_final_relative_energy_change=float(spec["final_relative_energy_change"]),
                              jump_sum=jump_sum, jump_sum_over_H0=jump_sum/energy0,
                              boundary_term=boundary, boundary_over_H0=boundary/energy0,
                              telescope_error=total-boundary-jump_sum,
                              work_energy_error=total-final_energy,
                              max_interval_work_energy_error=max(endpoint_errors),
                              max_stored_relative_energy_error=max(energy_errors),
                              positive_net_interval_fraction=sum(w > 0 for w in work)/2000,
                              net_over_sum_absolute_interval_work=total/math.fsum(abs(w) for w in work),
                              unreferenced_work_over_H0=free_work/energy0,
                              reference_assisted_work_over_H0=reference_work/energy0,
                              updates_with_positive_post_power=positive_power_updates,
                              updates_with_negative_post_power=negative_power_updates,
                              max_nonfull_power_contraction_error=max(contraction_errors, default=0)))
    save_csv("oscillator_trajectory_work.csv", summaries)
    save_csv("oscillator_model_segments.csv", segments)
    save_csv("oscillator_update_power.csv", updates)
    aggregates = []
    for group in sorted({(r["split"], r["policy"]) for r in summaries}):
        sub = [r for r in summaries if (r["split"], r["policy"]) == group]
        subseg = [r for r in segments if (r["split"], r["policy"]) == group]
        subup = [r for r in updates if (r["split"], r["policy"]) == group]
        avg = lambda field: math.fsum(r[field] for r in sub)/len(sub)
        aggregates.append(dict(split=group[0], policy=group[1], trajectories=len(sub),
                               mean_W_over_H0=avg("W_over_H0"),
                               mean_original_relative_energy_change=avg("original_final_relative_energy_change"),
                               mean_jump_sum_over_H0=avg("jump_sum_over_H0"),
                               mean_boundary_over_H0=avg("boundary_over_H0"),
                               mean_unreferenced_work_over_H0=avg("unreferenced_work_over_H0"),
                               mean_reference_assisted_work_over_H0=avg("reference_assisted_work_over_H0"),
                               mean_positive_net_interval_fraction=avg("positive_net_interval_fraction"),
                               mean_net_over_sum_absolute_interval_work=avg("net_over_sum_absolute_interval_work"),
                               positive_net_model_segments=sum(r["work"] > 0 for r in subseg),
                               model_segments=len(subseg), reference_updates=len(subup),
                               updates_with_positive_post_power=sum(r["power_after"] > 1e-14 for r in subup),
                               updates_with_negative_post_power=sum(r["power_after"] < -1e-14 for r in subup)))
    result = dict(status="saved-trajectory diagnostic only", new_DFT_labels=0,
                  new_trajectories=0, trajectories=len(summaries), intervals=2000*len(summaries),
                  model_segments=len(segments), reference_updates=len(updates),
                  gauge="U_b(0)=U_ref(0)=0 for every bias; delta_U=(1-k)*x^2/2-b*x",
                  endpoint_source="Next saved row, except final endpoint decoded from saved reference-solution errors",
                  residual_work="W_j=(k-1)*(x_end^2-x_start^2)/2+b_j*(x_end-x_start)",
                  initial_update="Boundary uses bias after initial correction; initial correction excluded from jump sum",
                  limits="Descriptive endogenous-path diagnosis, no fixed-bias causal counterfactual. "
                         "Sum absolute interval work is a lower bound on continuous absolute work. "
                         "Reference-assisted intervals still propagate a corrected surrogate. "
                         "Instantaneous switching keeps x,v and H_ref unchanged.",
                  max_total_work_energy_error=max(abs(r["work_energy_error"]) for r in summaries),
                  max_telescope_error=max(abs(r["telescope_error"]) for r in summaries),
                  max_interval_work_energy_error=max(r["max_interval_work_energy_error"] for r in summaries),
                  max_stored_relative_energy_error=max(r["max_stored_relative_energy_error"] for r in summaries),
                  max_nonfull_power_contraction_error=max(r["max_nonfull_power_contraction_error"] for r in summaries),
                  aggregates=aggregates, cpu_seconds=cpu()-started)
    dump("oscillator_work_diagnostic.json", result)
    print(f"saved oscillator diagnosis: {len(summaries)} trajectories, CPU={cpu():.2f}s", flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "run_started.json").exists() or REPORT.exists():
        raise FileExistsError("Existing audit/report retained: no automatic rerun or overwrite")

    def timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("295-second CPU guard reached; retain partial results, no rerun")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, max(1, 295-cpu()))
    used = resource.getrusage(resource.RUSAGE_SELF)
    hard = resource.getrlimit(resource.RLIMIT_CPU)[1]
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(used.ru_utime+used.ru_stime)+298, hard))
    inputs = [p for p in OLD.iterdir() if p.is_file()]
    inputs += [ROOT / "experiments/toy/verification_thinning.py",
               ROOT / "experiments/toy/prospective_horizon.py",
               ROOT / "docs/submission_plan_20260905/scientific_upgrade_options.md"]
    inputs += [OSC / n for n in ("steps.csv", "trajectories.csv", "protocol.json", "manifest.json")]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    frozen = json.loads((OLD / "run_started.json").read_text())["sha256_before_sampling"]
    assert all(hashes[name] == digest for name, digest in frozen.items())
    dump("run_started.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
                                  python=sys.version, numpy=np.__version__,
                                  command="uv run --no-sync python -B experiments/verify_thinning_independent_20260905.py",
                                  script_sha256=sha(Path(__file__)), input_sha256=hashes,
                                  cpu_limit_seconds=300, cpu_soft_guard_seconds=295,
                                  worker_processes=1, numerical_threads=1,
                                  seed_policy="Only frozen entropy 20260905 / keys 0..19999, once",
                                  old_source_protocol_hashes_match=True))
    try:
        original = json.loads((OLD / "results.json").read_text())
        dump("original_statistics_preserved.json", original)
        audit = reconstruct_original_samples()
        probability = deterministic_probability()
        oscillator = oscillator_diagnostic() if cpu() < 230 else dict(status="skipped: remaining CPU reserved")
        unchanged = all(sha(ROOT / name) == digest for name, digest in hashes.items())
        report = dict(status="completed", original_sample_audit=audit,
                      deterministic_probability=probability, oscillator=oscillator,
                      original_inputs_unchanged=unchanged, cpu_seconds=cpu(),
                      wall_seconds=time.monotonic()-WALL_START)
        dump("results.json", report)
        # Detailed interpretation is added to this one authorized report file.
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x") as stream:
            stream.write("# P0 统计核对与已存振子诊断 — 2026-09-05\n\n")
            stream.write(f"- 原样本事件重构：{audit['reconstructed_zero_count']}/20000；差异 {audit['mismatch_count']}。\n")
            stream.write(f"- 确定性有限状态递推：P(E0) ≈ {probability['p_event_float64']:.16g}。\n")
            stream.write(f"- 总 CPU：{cpu():.3f} s；输入哈希保持不变：{unchanged}。\n")
        dump("run_completed.json", dict(status="completed", cpu_seconds=cpu(),
                                         wall_seconds=time.monotonic()-WALL_START,
                                         original_inputs_unchanged=unchanged,
                                         completed_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except BaseException as exc:
        dump("run_failed.json", dict(error=f"{type(exc).__name__}: {exc}", cpu_seconds=cpu(),
                                      action="Preserve all outputs; no automatic rerun"))
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)


def verify_saved_outputs() -> None:
    """Read-only input verification and segment decomposition; no RNG calls.

    This separate stage is reproducible with --verify-saved. Its CPU time is
    added to the completed original audit, not treated as a fresh budget.
    """
    started = cpu()
    previous = json.loads((OUT / "run_completed.json").read_text())
    prior_cpu = previous["cpu_seconds"]

    def stop(_signum: int, _frame: object) -> None:
        raise TimeoutError("Shared 300-second CPU budget exhausted")

    signal.signal(signal.SIGPROF, stop)
    signal.setitimer(signal.ITIMER_PROF, max(.01, 290-prior_cpu-cpu()))
    dump("saved_verification_started.json", dict(prior_cpu_seconds=prior_cpu,
                                                final_script_sha256=sha(Path(__file__)),
                                                command="uv run --no-sync python -B experiments/verify_thinning_independent_20260905.py --verify-saved",
                                                new_random_draws=0))
    try:
        history = rows(OUT / "absorbed_probability_by_step.csv")
        # Exact scalar rational propagation uses a sparse state dictionary,
        # independently of the dense floating-array implementation.
        state = {(350, 800, 0): Fraction(1)}
        failure = Fraction(0)
        exact_checks = []
        for t in range(8):
            future = defaultdict(Fraction)
            for (q, a, v), w in state.items():
                qa = Fraction(q*a, 1_000_000)
                failure += w*qa/10
                for branch, prob in (
                    ((q+4, a+2, v), 1-Fraction(a, 1000)),
                    ((q+10, a+2, v+1), qa*Fraction(9, 10)),
                    ((q+10, a+2, v), Fraction(a, 1000)*(1-Fraction(q, 1000))*Fraction(9, 10)),
                    ((q+20, a+27, v), Fraction(a, 1000)*(1-Fraction(q, 1000))/10),
                ):
                    qn, an, vn = branch
                    future[(min(800, qn), min(980, an), vn)] += w*prob
            state = future
            live = sum(state.values(), Fraction(0))
            assert live+failure == 1
            delta = abs(float(failure)-float(history[t]["cumulative_failure"]))
            assert delta < 2e-15
            exact_checks.append(dict(step=t+1, states=len(state), exact_failure=str(failure),
                                     dense_absolute_difference=delta))
        first_possible = math.prod(Fraction((350+10*t)*(800+2*t)*9, 10_000_000) for t in range(29))
        assert all(float(r["cumulative_success"]) == 0 for r in history[:28])
        earliest_relative_error = abs(float(history[28]["cumulative_success"])/float(first_possible)-1)
        assert earliest_relative_error < 5e-15

        audit = json.loads((OUT / "original_sample_audit.json").read_text())
        reconstructed = rows(OUT / "original_seed_event_reconstruction.csv")
        assert len(reconstructed) == 20000
        assert [int(r["replicate"]) for r in reconstructed] == list(range(20000))
        assert sum(r["reconstructed_ever_zero_event"] == "True" for r in reconstructed) == audit["reconstructed_zero_count"]
        assert sum(int(r["inspected_prefix_steps"]) for r in reconstructed) == audit["inspected_scalar_steps"]
        # Check the saved old confidence interval with a different implementation.
        from scipy.special import betaincinv

        old = json.loads((OLD / "results.json").read_text())["zero_detection"]
        count = old["events"]
        ci = [float(betaincinv(count, 20001-count, .025)),
              float(betaincinv(count+1, 20000-count, .975))]
        assert max(abs(a-b) for a, b in zip(ci, old["ci95_clopper_pearson"])) < 1e-11

        saved_segments = rows(OUT / "oscillator_model_segments.csv")
        wanted = {(r["split"], r["policy"], r["seed"], int(r["start_step"])) for r in saved_segments}
        wanted |= {(r["split"], r["policy"], r["seed"], 1999) for r in saved_segments}
        states = {}
        with (OSC / "steps.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = row["split"], row["policy"], row["seed"], int(row["step"])
                if key in wanted:
                    states[key] = row
        trajectories = {(r["split"], r["policy"], r["seed"]): r for r in rows(OSC / "trajectories.csv")}
        energies = {(r["split"], r["policy"], r["seed"]): float(r["H0"])
                    for r in rows(OUT / "oscillator_trajectory_work.csv")}
        decomposition, accum = [], defaultdict(lambda: [0.0, 0.0])
        max_decomposition_error = 0.0
        for seg in saved_segments:
            key = seg["split"], seg["policy"], seg["seed"]
            first = states[(*key, int(seg["start_step"]))]
            x = float(first["x"])
            k = float(first["stiffness"])
            end = int(seg["end_step_exclusive"])
            if end < 2000:
                xn = float(states[(*key, end)]["x"])
            else:
                last, spec = states[(*key, 1999)], trajectories[key]
                x0, v0 = float(spec["x0"]), float(spec["v0"])
                angle = math.sqrt(k)*float(last["time_end"])
                xn = x0*math.cos(angle)+v0/math.sqrt(k)*math.sin(angle)+float(last["position_error_at_end"])
            dx = xn-x
            r_start = (k-1)*x+float(seg["bias"])
            anchor_work = r_start*dx
            growth_work = .5*(k-1)*dx*dx
            error = anchor_work+growth_work-float(seg["work"])
            max_decomposition_error = max(max_decomposition_error, abs(error))
            accum[key][0] += anchor_work/energies[key]
            accum[key][1] += growth_work/energies[key]
            decomposition.append(dict(**seg, x_start=x, x_end=xn, residual_at_segment_start=r_start,
                                      anchor_residual_work=anchor_work, curvature_growth_work=growth_work,
                                      decomposition_error=error))
        assert max_decomposition_error < 1e-12
        save_csv("oscillator_segment_mechanism.csv", decomposition)
        mechanism_groups = []
        for group in sorted({key[:2] for key in accum}):
            subset = [value for key, value in accum.items() if key[:2] == group]
            mechanism_groups.append(dict(split=group[0], policy=group[1], trajectories=len(subset),
                                         mean_anchor_residual_work_over_H0=math.fsum(x[0] for x in subset)/len(subset),
                                         mean_curvature_growth_work_over_H0=math.fsum(x[1] for x in subset)/len(subset)))
        dump("oscillator_segment_mechanism.json", dict(
            identity="W_segment = r_start*Delta_x + (k-1)*(Delta_x)^2/2",
            interpretation="For this k>1 harmonic mismatch, each fixed-bias segment has a nonnegative "
                           "curvature-growth contribution. Updates contract residuals and re-anchor segments; "
                           "this decomposition is not a fixed-bias trajectory counterfactual.",
            max_decomposition_error=max_decomposition_error, aggregates=mechanism_groups))
        reference_path = OSC / "reference_events.csv"
        authoritative_full = {(r["split"], r["policy"], r["seed"], r["step"]): bool(int(r["full_correction_used"]))
                              for r in rows(reference_path)}
        update_rows = rows(OUT / "oscillator_update_power.csv")
        flag_differences = sum((r["full_correction"] == "True") != authoritative_full[
            (r["split"], r["policy"], r["seed"], r["step"])] for r in update_rows)
        original_manifest = json.loads((OUT / "run_started.json").read_text())
        unchanged = all(sha(ROOT / name) == digest for name, digest in original_manifest["input_sha256"].items())
        assert unchanged
        checks = dict(status="completed", exact_sparse_rational_checks=exact_checks,
                      exact_first_possible_success_probability=str(first_possible),
                      earliest_success_relative_error=earliest_relative_error,
                      independent_scipy_clopper_pearson_95=ci,
                      source_full_correction_flag_differences=flag_differences,
                      additional_input_reference_events_sha256=sha(reference_path),
                      original_inputs_unchanged=unchanged,
                      source_scope="Main audit source SHA is preserved in run_started.json; "
                                   "this verification stage was appended later and never reruns old streams.",
                      final_script_sha256=sha(Path(__file__)),
                      verification_cpu_seconds=cpu()-started,
                      total_analysis_cpu_seconds=prior_cpu+cpu(),
                      mechanism_aggregates=mechanism_groups)
        dump("saved_output_verification.json", checks)
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    except BaseException as exc:
        dump("saved_verification_failed.json", dict(error=f"{type(exc).__name__}: {exc}",
                                                   total_analysis_cpu_seconds=prior_cpu+cpu()))
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify-saved"]:
        verify_saved_outputs()
    else:
        main()
