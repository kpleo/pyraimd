"""Analyze saved P2a records and draw a four-panel figure; zero model/DFT calls.

uv run --no-sync python -B experiments/analyze_forward_motion_20260905.py --analyze-only
uv run --no-sync python -B experiments/analyze_forward_motion_20260905.py --draw-only
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/execution_20260905/h2o"
DERIVED = DATA / "motion_derived"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = str(DERIVED / "runtime")

import argparse
import ast
import csv
import hashlib
import json
import math
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
from ase import units
from ase.data import atomic_masses

EPS = .1
FACTORS = (.5, 1., 2.)
SEEDS = (2026090501, 2026090502)
CONTRACT = {
    "conclusion": "Two molecular anchors show distinct observed crossing times under velocity scaling; "
                  "short empirical forecasts and local derivative comparisons are diagnostics, while the "
                  "stored oscillator work separates anchor residual and curvature contributions.",
    "archetype": "quantitative grid; two top trajectory panels are the primary evidence",
    "backend": "Python/matplotlib, explicitly selected by the requested Python workflow",
    "panels": {"a": "anchor A: three recorded force-error time series and preset forecast bars",
               "b": "anchor B: same axes and encodings; censoring explicit",
               "c": "directional residual derivative at two h, observed path secants, and stored empirical kappa",
               "d": "paired additive work contributions for all 16 stored heldout class-horizon oscillators"},
    "exports": {"width_mm": 183, "height_mm": 151, "formats": ["pdf", "png", "svg"],
                "png_dpi": 600, "font_pt": 7, "editable_vector_text": True},
    "statistics": "Two anchors and six deterministic interventions, no inferred lifetime for four censored cases. "
                  "Panel d contains 16 original trajectories; means weight trajectories equally. "
                  "No confidence interval, significance test, or fit to an inverse-speed law.",
    "risks": "No continuous-time certificate, no exact prediction claim, no new inference/DFT/training; "
             "saved timestamps cannot authenticate a before-reference prediction commitment.",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def write_csv(name, records):
    with (DERIVED / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def norm_inf2(a):
    return float(np.sqrt(np.sum(np.asarray(a)**2, axis=1)).max())


def geometry_key(x, numbers):
    return hashlib.sha256((str(numbers.tolist())+"|"+";".join(
        f"{a:.10f},{b:.10f},{c:.10f}" for a, b, c in x/units.Bohr)).encode()).hexdigest()


def read_anchors():
    path = ROOT / "analysis/h2o_streak/coverage_h2o.db"
    connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    labels = []
    for numbers, positions, masses, constraints, kv in connection.execute(
            "SELECT numbers, positions, masses, constraints, key_value_pairs FROM systems"):
        meta = json.loads(kv)
        if meta.get("run_id") != "collect-h2o-nve-300K":
            continue
        numbers = np.frombuffer(numbers, dtype="<i4").copy()
        mass = (np.frombuffer(masses, dtype="<f8").copy() if masses else atomic_masses[numbers])
        labels.append(dict(step=meta["step"], numbers=numbers, masses=mass,
                           x=np.frombuffer(positions, dtype="<f8").reshape(-1, 3).copy(),
                           constraints=constraints))
    connection.close()
    labels.sort(key=lambda r: r["step"])
    return {SEEDS[0]: labels[0], SEEDS[1]: labels[151]}


def analyze():
    started = time.process_time()
    DERIVED.mkdir(parents=True, exist_ok=True)
    summary = read_json(DATA / "velocity_summary.json")
    protocol = read_json(DATA / "locked_protocol.json")
    inputs = sorted(DATA.glob("velocity_*.json"))
    inputs += [DATA / "velocity_reference_attempts.jsonl", DATA / "locked_protocol.json",
               DATA / "development_source.py", ROOT / "experiments/forward_h2o_20260905.py",
               ROOT / "analysis/h2o_streak/coverage_h2o.db",
               ROOT / "analysis/execution_20260905/statistics/bregman_closed_form_results.json"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    ledger = [json.loads(line) for line in (DATA / "velocity_reference_attempts.jsonl").read_text().splitlines()]
    completed = {r["tag"]: r for r in ledger if r["status"] == "complete"}
    attempts = [r for r in ledger if r["status"] == "started"]
    checks = []

    def check(name, condition, details=None):
        checks.append(dict(name=name, passed=bool(condition), details=details))

    check("122 unique attempts and completions", len(attempts) == len(completed) == summary["attempts"] == 122
          and {r["attempt"] for r in attempts} == set(range(1, 123)))
    check("114 trajectory and eight directional-probe completions",
          sum(k.startswith("velocity_") for k in completed) == 114
          and sum(k.startswith("probe:") for k in completed) == 8)
    check("saved reference wall time total", abs(sum(r["seconds"] for r in completed.values())-summary["reference_seconds"]) < 1e-8)
    dt_fs = protocol["dt_fs"]
    dt = dt_fs*units.fs
    anchors = read_anchors()
    case_rows, point_rows, component_rows, forecast_rows = [], [], [], []
    trajectories = {}
    max_position_error = max_momentum_error = max_norm_error = max_ledger_error = 0.
    max_forecast_length_error = max_forecast_bound_error = 0.
    for case in summary["cases"]:
        seed, factor = case["seed"], case["factor"]
        prefix = f"velocity_{seed}_{factor:g}"
        saved = read_json(DATA / f"{prefix}_steps.json")
        prediction = read_json(DATA / f"{prefix}_prediction.json")
        anchor = anchors[seed]
        masses, numbers = anchor["masses"][:, None], anchor["numbers"]
        x, p, fs, fr = (np.array([r[k] for r in saved]) for k in
                        ("positions", "momenta_full", "surrogate_forces", "reference_forces"))
        t = np.array([r["time_fs"] for r in saved])
        residual = fs-fr
        e = np.array([norm_inf2(r) for r in residual])
        check(prefix+" fixed grid and finite arrays", len(saved) == 20
              and [r["step"] for r in saved] == list(range(20))
              and np.array_equal(t, np.arange(20)*dt_fs)
              and all(np.isfinite(v).all() for v in (x, p, fs, fr)))
        check(prefix+" saved initial geometry", np.array_equal(x[0], anchor["x"]) and anchor["constraints"] is None)
        check(prefix+" prediction anchor identity", geometry_key(x[0], numbers) == prediction["coordinates_sha256"])
        position_error = np.max(np.abs(x[1:] - (x[:-1]+dt*(p[:-1]+.5*dt*fs[:-1])/masses)))
        momentum_error = np.max(np.abs(p[1:] - (p[:-1]+.5*dt*(fs[:-1]+fs[1:]))))
        norm_error = np.max(np.abs(e-np.array([r["error"] for r in saved])))
        max_position_error = max(max_position_error, float(position_error))
        max_momentum_error = max(max_momentum_error, float(momentum_error))
        max_norm_error = max(max_norm_error, float(norm_error))
        check(prefix+" complete velocity-Verlet", position_error < 1e-12 and momentum_error < 1e-12)
        check(prefix+" independently recomputed max-atom vector force error", norm_error < 1e-12)
        length_steps = np.linalg.norm(np.diff(x, axis=0).reshape(19, -1), axis=1)
        path = np.r_[0., np.cumsum(length_steps)]
        changes = np.array([norm_inf2(r) for r in np.diff(residual, axis=0)])
        secants = changes/length_steps
        anchor_change = np.array([norm_inf2(r-residual[0]) for r in residual])
        kap, e0 = prediction["kappa"], prediction["e0"]
        empirical_bound = e0+kap*path
        forecast = prediction["forecast"]
        last_safe = 0.
        for time_fs, length, bound in forecast:
            i = int(round(time_fs/dt_fs))
            max_forecast_length_error = max(max_forecast_length_error, abs(length-path[i]))
            max_forecast_bound_error = max(max_forecast_bound_error, abs(bound-(e0+kap*length)))
            forecast_rows.append(dict(seed=seed, factor=factor, time_fs=time_fs,
                                      saved_forecast_length_A=length, observed_path_length_A=path[i],
                                      saved_forecast_bound_ev_A=bound, inside_budget=bound <= EPS))
            if bound <= EPS:
                last_safe = time_fs
        check(prefix+" forecast finite-step inversion", last_safe == prediction["horizon_fs"] == case["horizon_fs"])
        check(prefix+" initial error and kappa agree", abs(e[0]-e0) < 1e-12
              and e0 == case["e0"] and kap == case["kappa"])
        crossings = np.flatnonzero(e > EPS)
        crossing = float(t[crossings[0]]) if crossings.size else None
        check(prefix+" first discrete exceedance and censoring", crossing == case["first_crossing_fs"]
              and bool(not crossings.size) == case["right_censored"])
        for i, row in enumerate(saved):
            if i:
                ref = completed[f"{prefix}:{i}"]
                check(f"{prefix}:{i} reference geometry", ref["geometry"] == geometry_key(x[i], numbers))
                max_ledger_error = max(max_ledger_error, float(np.max(np.abs(fr[i]-np.array(ref["forces"])))))
            point_rows.append(dict(seed=seed, anchor="A" if seed == SEEDS[0] else "B", factor=factor,
                                   step=i, time_fs=float(t[i]), error_ev_A=float(e[i]),
                                   path_length_A=float(path[i]), residual_change_from_anchor_ev_A=float(anchor_change[i]),
                                   measured_vector_secant_ev_A2=float(secants[i-1]) if i else None,
                                   empirical_kappa_ev_A2=kap, empirical_bound_along_saved_path_ev_A=float(empirical_bound[i]),
                                   horizon_fs=last_safe, first_observed_crossing_fs=crossing,
                                   right_censored=not bool(crossings.size)))
            for atom in range(3):
                component_rows.append(dict(seed=seed, factor=factor, step=i, atom=atom,
                                          rx_ev_A=float(residual[i, atom, 0]), ry_ev_A=float(residual[i, atom, 1]),
                                          rz_ev_A=float(residual[i, atom, 2]), atom_norm_ev_A=float(np.linalg.norm(residual[i, atom]))))
        case_rows.append(dict(seed=seed, anchor="A" if seed == SEEDS[0] else "B", factor=factor,
                              initial_error_ev_A=float(e[0]), horizon_fs=last_safe,
                              first_observed_crossing_fs=crossing, right_censored=not bool(crossings.size),
                              observation_end_fs=float(t[-1]), max_error_ev_A=float(e.max()),
                              exceeding_force_points=int(crossings.size), initial_speed_full_norm_A_fs=float(np.linalg.norm(p[0]/masses)*units.fs),
                              kappa_ev_A2=kap, max_observed_vector_secant_ev_A2=float(secants.max()),
                              max_observed_secant_over_kappa=float(secants.max()/kap),
                              max_anchor_change_over_kappa_path=float(np.max(anchor_change[1:]/(kap*path[1:]))),
                              empirical_bound_exceedances=int(np.sum(e > empirical_bound+1e-12)),
                              max_position_identity_error_A=float(position_error),
                              max_momentum_identity_error_ASE=float(momentum_error)))
        trajectories[(seed, factor)] = dict(x=x, p=p, fs=fs, fr=fr, residual=residual)
    check("all stored reference forces match completed ledger", max_ledger_error < 1e-12)
    check("saved surrogate forecasts follow same later recorded path", max_forecast_length_error < 1e-12
          and max_forecast_bound_error < 1e-12)
    for seed in SEEDS:
        base = trajectories[(seed, 1.)]
        for factor in FACTORS:
            tr = trajectories[(seed, factor)]
            check(f"{seed} factor {factor:g} paired velocity intervention",
                  np.array_equal(tr["x"][0], base["x"][0])
                  and np.max(np.abs(tr["p"][0]-factor*base["p"][0])) < 1e-12
                  and np.max(np.abs(tr["fs"][0]-base["fs"][0])) < 1e-12
                  and np.max(np.abs(tr["fr"][0]-base["fr"][0])) < 1e-12)
    probe_rows, probe_components, probe_summary = [], [], []
    for seed, start_index in zip(SEEDS, (0, 151)):
        probes = read_json(DATA / f"velocity_{seed}_directional_probe.json")
        d = [np.array(row["directional_residual_derivative"]) for row in probes]
        base = trajectories[(seed, 1.)]
        vel = base["p"][0]/anchors[seed]["masses"][:, None]
        direction = vel/np.linalg.norm(vel)
        norms = [norm_inf2(a) for a in d]
        for row, derivative, magnitude in zip(probes, d, norms):
            h = row["h_A"]
            check(f"{seed} probe h={h} vector norm", abs(magnitude-row["norm"]) < 1e-12)
            for sign in (-1, 1):
                xprobe = base["x"][0]+sign*h*direction
                check(f"{seed} probe h={h} sign={sign} normalized-direction geometry",
                      geometry_key(xprobe, anchors[seed]["numbers"]) == completed[f"probe:{start_index}:{h}:{sign}"]["geometry"])
            probe_rows.append(dict(seed=seed, anchor="A" if seed == SEEDS[0] else "B", h_A=h,
                                   derivative_norm_ev_A2=magnitude, empirical_kappa_ev_A2=summary["cases"][0]["kappa"],
                                   max_observed_path_secant_ev_A2=max(r["max_observed_vector_secant_ev_A2"] for r in case_rows if r["seed"] == seed)))
            for atom, row_vec in enumerate(derivative):
                probe_components.append(dict(seed=seed, h_A=h, atom=atom,
                                             Dx_ev_A2=float(row_vec[0]), Dy_ev_A2=float(row_vec[1]), Dz_ev_A2=float(row_vec[2])))
        probe_summary.append(dict(seed=seed, norm_h001_ev_A2=norms[0], norm_h002_ev_A2=norms[1],
                                  relative_vector_difference=norm_inf2(d[0]-d[1])/norms[0],
                                  relative_norm_difference=abs(norms[0]-norms[1])/norms[0],
                                  empirical_kappa_over_local_derivative=summary["cases"][0]["kappa"]/norms[0]))
    source = (ROOT / "experiments/forward_h2o_20260905.py").read_text()
    tree = ast.parse(source)
    velocity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "velocity")
    predicts = [n.lineno for n in ast.walk(velocity) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "forecast"]
    evaluate_lines = [n.lineno for n in ast.walk(velocity) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "evaluate"]
    check("source forecast before future reference loop", max(predicts) < sorted(evaluate_lines)[1])
    bregman = read_json(ROOT / "analysis/execution_20260905/statistics/bregman_closed_form_results.json")
    oscillator = [dict(trajectory_index=i+1, **r) for i, r in enumerate(bregman["heldout_class_horizon_per_seed"])]
    check("16 old oscillator decompositions close", len(oscillator) == 16
          and max(abs(r["anchor_work_over_H0"]+r["curvature_work_over_H0"]-r["original_W_over_H0"]) for r in oscillator) < 1e-12)
    for name, records in (("motion_points.csv", point_rows), ("motion_residual_vectors.csv", component_rows),
                          ("motion_cases.csv", case_rows), ("motion_forecasts.csv", forecast_rows),
                          ("motion_directional_probes.csv", probe_rows), ("motion_probe_vectors.csv", probe_components),
                          ("oscillator_work_components.csv", oscillator)):
        write_csv(name, records)
    unchanged = all(sha(ROOT / p) == digest for p, digest in before.items())
    check("read-only inputs unchanged", unchanged)
    result = dict(status="passed" if all(r["passed"] for r in checks) else "failed",
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), analysis_script_sha256=sha(Path(__file__)),
                  source_sha256=before, source_input_mtimes_ns={str(p.relative_to(ROOT)): p.stat().st_mtime_ns for p in inputs},
                  old_reference_attempts=122, new_DFT=0, new_inference=0, new_training=0, new_trajectories=0,
                  cases=case_rows, directional_probes=probe_summary, checks=checks,
                  max_position_identity_error_A=max_position_error, max_momentum_identity_error_ASE=max_momentum_error,
                  max_force_norm_error_ev_A=max_norm_error, max_reference_ledger_error_ev_A=max_ledger_error,
                  max_forecast_path_length_error_A=max_forecast_length_error,
                  max_forecast_bound_arithmetic_error_ev_A=max_forecast_bound_error,
                  source_order=dict(forecast_line=predicts[0], reference_evaluation_lines=sorted(evaluate_lines),
                                    reference_record_timestamps_available=False,
                                    independent_timestamp_commitment=False,
                                    interpretation="Source ordering and later path consistency support the declared protocol; "
                                                   "self-reported flags and filesystem mtimes do not authenticate pre-reference commitment."),
                  derivative_limit="Saved probe vectors allow norm and two-spacing stability checks. Individual "
                                   "surrogate forces at +/-h were not saved, so a full raw-force difference cannot be independently rebuilt.",
                  kappa_limit="Historical same-model residual vectors used to construct kappa were not retained; "
                              "kappa is a recorded empirical comparator, not independently re-estimated or certified.",
                  relation_limit="Finite path secants and e0+kappa*observed_length use saved points only; "
                                 "neither bounds unseen curvature or between-point continuous errors.",
                  figure_contract=CONTRACT, cpu_seconds=time.process_time()-started)
    write_json(DATA / "motion_verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "cases", "directional_probes", "max_position_identity_error_A",
                                            "max_momentum_identity_error_ASE", "max_force_norm_error_ev_A",
                                            "max_forecast_path_length_error_A", "cpu_seconds")}, indent=2))
    return result


def draw(output=None, write_report=True):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from xml.etree import ElementTree

    result = read_json(DATA / "motion_verification.json")
    if result["status"] != "passed":
        raise RuntimeError("Retain failed audit; inspect before making a manuscript figure")
    for name, digest in result["source_sha256"].items():
        # A figure-only export reads saved numbers, not trajectory-generation code.
        # Updating the analysis report still requires the complete original inputs.
        if not write_report and Path(name).suffix == ".py":
            continue
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Input changed since independent audit: {name}")
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
                         "legend.fontsize": 6.7, "legend.frameon": False, "axes.linewidth": .65,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "forward-motion-20260905",
                         "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
                         "savefig.facecolor": "white"})
    width, height = 183., 151.
    fig = plt.figure(figsize=(width/25.4, height/25.4), dpi=150, facecolor="white")
    color = {.5: "#487FA5", 1.: "#368C87", 2.: "#B86F45"}
    linestyle = {.5: (0, (3, 2)), 1.: "-", 2.: "-."}
    marker = {.5: "o", 1.: "s", 2.: "^"}
    ink, muted = "#242B31", "#707A83"

    def text(x, y, value, **kwargs):
        return fig.text(x/width, y/height, value, color=ink, **kwargs)

    def axes(x, y, w, h):
        ax = fig.add_axes([x/width, y/height, w/width, h/height])
        ax.tick_params(direction="out", length=2.4, width=.65, pad=2)
        return ax

    def title(x, y, letter, value):
        text(x, y, letter, fontsize=9, fontweight="bold", va="baseline")
        text(x+6, y, value, fontsize=7.8, fontweight="bold", va="baseline")

    handles = [Line2D([], [], color=color[f], linestyle=linestyle[f], marker=marker[f],
                      markersize=3, linewidth=1.1, label=f"{f:g}× initial velocity") for f in FACTORS]
    handles += [Line2D([], [], color=ink, linestyle="--", linewidth=.8, label="0.10 eV Å$^{-1}$ budget"),
                Line2D([], [], color=muted, linestyle="none", marker=">", markerfacecolor="white",
                       markersize=4, label="Right-censored at 9.5 fs")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.515, .928), ncol=5,
               columnspacing=1.0, handlelength=1.7, handletextpad=.45)
    all_main_axes = []
    for seed, left, letter in zip(SEEDS, (16., 108.), ("a", "b")):
        title(left-11, 146, letter, f"Water anchor {'A' if seed == SEEDS[0] else 'B'}")
        ax = axes(left, 98, 69, 37)
        ax.set(xlim=(0, 10), ylim=(0, .177))
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yticks([0, .05, .10, .15], labels=["0", "0.05", "0.10", "0.15"])
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("Force error (eV Å$^{-1}$)")
        ax.axhline(EPS, color=ink, lw=.8, ls=(0, (4, 3)), zorder=1)
        rug = axes(left, 78.5, 69, 11.5)
        rug.set(xlim=(0, 10), ylim=(-.65, 2.65), xlabel="Physical time (fs)")
        rug.set_xticks([0, 2, 4, 6, 8, 10])
        rug.set_yticks([])
        rug.spines["left"].set_visible(False)
        text(left, 94.0, "Preset horizon", fontsize=6.8, fontweight="bold")
        for j, f in enumerate(FACTORS):
            rows_saved = read_json(DATA / f"velocity_{seed}_{f:g}_steps.json")
            case = next(r for r in result["cases"] if r["seed"] == seed and r["factor"] == f)
            t = np.array([r["time_fs"] for r in rows_saved])
            e = np.array([norm_inf2(np.array(r["surrogate_forces"])-np.array(r["reference_forces"])) for r in rows_saved])
            ax.plot(t, e, color=color[f], ls=linestyle[f], marker=marker[f],
                    ms=2.5, mew=.35, lw=1.1, zorder=3)
            if case["right_censored"]:
                ax.plot(t[-1], e[-1], marker=">", ms=5.2, mfc="white", mec=color[f], mew=1., zorder=6)
            else:
                crossing = case["first_observed_crossing_fs"]
                i = int(round(crossing/.5))
                ax.plot(t[i], e[i], marker="o", ms=5., mfc="white", mec=color[f], mew=1.1, zorder=7)
                xytext = (1.5, .147) if seed == SEEDS[0] else (crossing-.7, .137)
                ax.annotate(f"First > budget\n{crossing:g} fs", xy=(t[i], e[i]), xytext=xytext,
                            fontsize=6.2, ha="center", va="center", color=ink,
                            bbox=dict(facecolor="white", edgecolor="none", pad=.5),
                            arrowprops=dict(arrowstyle="-", lw=.65, color=muted), zorder=8)
            y = 2-j
            horizon = case["horizon_fs"]
            rug.text(-.42, y, f"{f:g}×", ha="right", va="center", fontsize=6.2, color=color[f])
            rug.plot([0, horizon], [y, y], color=color[f], lw=1.7, solid_capstyle="butt")
            rug.plot(horizon, y, marker="|" if horizon else "x", ms=4.5, mew=1., color=color[f], clip_on=False)
            label = f"{horizon:g} fs" if horizon else "0 fs (<1 step)"
            rug.text(horizon+.23, y, label, va="center", fontsize=6.2, color=color[f])
        all_main_axes.append(ax)

    title(5, 60.8, "c", "Residual slopes versus empirical κ")
    ax = axes(16, 15, 69, 36.5)
    ax.set(xlim=(0, 2.15), ylim=(-.25, 1.9), xlabel="Residual slope (eV Å$^{-2}$)")
    ax.set_xticks([0, .5, 1., 1.5, 2.])
    ax.set_yticks([1.25, .25], labels=["Anchor A", "Anchor B"])
    for probe, y in zip(result["directional_probes"], (1.25, .25)):
        seed = probe["seed"]
        maximum = max(r["max_observed_vector_secant_ev_A2"] for r in result["cases"] if r["seed"] == seed)
        ax.plot(probe["norm_h001_ev_A2"], y+.14, "o", color=ink, ms=4, mfc=ink, mew=.7)
        ax.plot(probe["norm_h002_ev_A2"], y, "s", color=ink, ms=4, mfc="white", mew=.8)
        ax.plot(maximum, y-.14, "D", color="#69848F", ms=4, mfc="#69848F", mew=.7)
    kap = result["cases"][0]["kappa_ev_A2"]
    ax.axvline(kap, color=muted, lw=.9, ls=(0, (4, 3)))
    ax.text(kap-.06, 1.74, "κ = 1.950", ha="right", va="center", fontsize=6.5, color=muted)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ink, ms=3.4, label="h=0.001 Å"),
                       Line2D([], [], marker="s", ls="", color=ink, mfc="white", ms=3.4, label="h=0.002 Å"),
                       Line2D([], [], marker="D", ls="", color="#69848F", ms=3.4, label="Max. path secant")],
              loc="lower left", bbox_to_anchor=(-.05, 1.015), ncol=3,
              fontsize=5.9, handletextpad=.15, columnspacing=.65, borderaxespad=0.)

    title(97, 60.8, "d", "Oscillator residual work (n=16)")
    ax = axes(108, 15, 69, 36.5)
    with (DERIVED / "oscillator_work_components.csv").open() as stream:
        osc = list(csv.DictReader(stream))
    x = np.arange(1, 17)
    anchor = 100*np.array([float(r["anchor_work_over_H0"]) for r in osc])
    curvature = 100*np.array([float(r["curvature_work_over_H0"]) for r in osc])
    ax.bar(x, anchor, width=.72, color="#B9C7D3", edgecolor="white", linewidth=.3,
           label=f"Anchor residual: {anchor.mean():.2f}%")
    ax.bar(x, curvature, bottom=anchor, width=.72, color="#6A9F9C", edgecolor="white", linewidth=.3,
           label=f"Curvature: {curvature.mean():.2f}%")
    total_mean = float(np.mean(anchor+curvature))
    ax.axhline(total_mean, color=muted, lw=.7, ls=(0, (3, 3)))
    ax.set(xlim=(.2, 16.8), ylim=(0, 108), xlabel="Original trajectory order", ylabel="Residual work / $H_0$ (%)")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.text(.02, .94, f"Mean total\n{total_mean:.2f}%", transform=ax.transAxes,
            fontsize=6.3, va="top", color=ink)
    ax.legend(loc="upper right", fontsize=6.2, borderaxespad=.25,
              handlelength=1.0, handletextpad=.4, labelspacing=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5:
            outside.append(item.get_text())
    output = Path(output) if output is not None else ROOT / "manuscript/figures/forward_motion"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix("."+ext), dpi=600)
    plt.close(fig)
    svg = ElementTree.parse(output.with_suffix(".svg")).getroot()
    editable_text_count = sum(node.tag.endswith("}text") for node in svg.iter())
    pdf_bytes = output.with_suffix(".pdf").read_bytes()
    qa = dict(renderer="Python/matplotlib Agg; same figure canvas exported to PDF/SVG/PNG",
              final_size_mm=[width, height], png_dpi=600, text_outside_canvas=outside,
              svg_editable_text_elements=editable_text_count,
              pdf_embedded_truetype_font_reference=b"/FontFile2" in pdf_bytes,
              svg_embedded_raster_elements=sum(node.tag.endswith("}image") for node in svg.iter()),
              matplotlib_version=mpl.__version__, final_script_sha256=sha(Path(__file__)),
              figure_sha256={ext: sha(output.with_suffix("."+ext)) for ext in ("pdf", "png", "svg")},
              visual_review_status="pending direct inspection of the rendered PNG",
              statistics="Each line is one intervention; no smoothed fit or uncertainty bands. "
                         "Panel c is descriptive and vertically staggers h solely for visibility. "
                         "Panel d shows all 16 observations and equal-trajectory mean components.")
    if write_report:
        result["figure_qa"] = qa
        write_json(DATA / "motion_verification.json", result)
        write_reports(result)
    print(json.dumps(qa, indent=2))


def write_reports(result):
    report = ROOT / "docs/execution_20260905/molecular_motion_results.md"
    cases = result["cases"]
    paragraphs = ["# 两锚点速度干预：独立已存数据核对与作图\n",
                  "六条记录通过本次定向核对。两条 2× 速度轨迹首次离散超差分别为 2.5、8.0 fs；其余四条在 9.5 fs 右删失。预设时限明显短于这些观测，不能称准确寿命预测或逆速度定律验证。此任务只做保存数据算术和作图，新增 DFT／推理／训练／轨迹均为 0。\n",
                  "原 P2a 的 122 次新增参考由 114 个未来轨迹位置和 8 个方向探针构成；两初始锚点复用旧标签。6×20=120 个保存状态，0.5 fs 步长，物理窗口 0–9.5 fs；统计设计为两个锚点的六个速度干预，不是六个独立随机初态。\n",
                  "每组按 0.5×、1×、2× 顺序："]
    for seed in SEEDS:
        group = [r for r in cases if r["seed"] == seed]
        horizons = ", ".join(f"{r['horizon_fs']:g}" for r in group)
        maxima = ", ".join(f"{r['max_error_ev_A']:.6f}" for r in group)
        crossings = ", ".join("9.5 fs 右删失" if r["right_censored"] else f"{r['first_observed_crossing_fs']:g} fs 首次离散超差" for r in group)
        slopes = ", ".join(f"{r['max_observed_vector_secant_ev_A2']:.6f}" for r in group)
        paragraphs.append(f"- 锚点 {group[0]['anchor']}，seed={seed}：时限 [{horizons}] fs；结果 [{crossings}]；最大误差 [{maxima}] eV/Å；各轨迹最大残差向量割线 [{slopes}] eV/Å²。")
    paragraphs += ["\n0 fs 时限表示第一个 0.5 fs 预演点已不能由经验规则接受，不表示初始误差已超差，也不是实际可用寿命为零。四条右删失记录仅说明所有保存力评价点截至 9.5 fs 未超差，不排除点间超差，不能赋予无穷寿命。\n",
                   "独立算术检查：",
                   f"- 从原数据库只读取得质量与初始构型，逐一核对配对坐标、0.5/1/2 动量缩放、冻结初始代理力与参考力。114 个区间使用保存代理力验证完整 velocity-Verlet；最大位置恒等式差 {result['max_position_identity_error_A']:.3g} Å，最大完整动量差 {result['max_momentum_identity_error_ASE']:.3g} ASE 动量单位。",
                   f"- 独立重算每个原子的三分量残差，再取最大原子二范数，120 个误差值最大差 {result['max_force_norm_error_ev_A']:.3g} eV/Å；首次超差、删失标记与原 summary 一致。114 个未来参考力及坐标指纹均与完成账本一致。",
                   f"- 保存 forecast 的路径长度与同一冻结模型随后记录的路径一致，最大差 {result['max_forecast_path_length_error_A']:.3g} Å；e0+κℓ 算术和时限离散反演一致。这是对保存数据的一致性检查，没有重复预演或调用模型。",
                   "- 原源码先写 prediction 再进入未来参考循环，预测 JSON 也有相应声明；但账本没有逐次可靠时间戳或独立预承诺。文件 mtime 与自报布尔值不能证明真实的事前提交，本核对不伪造这种证据。\n",
                   "方向导数与历史 κ：以完整构型速度的欧氏范数归一化方向，导数范数与 κ 均用 eV/Å²；输出力向量范数始终是最大原子二范数。"]
    for r in result["directional_probes"]:
        paragraphs.append(f"- seed={r['seed']}：h=0.001/0.002 Å 的方向导数范数分别为 {r['norm_h001_ev_A2']:.9f}/{r['norm_h002_ev_A2']:.9f}；两估计向量的相对差 {100*r['relative_vector_difference']:.6f}%；κ/局部导数={r['empirical_kappa_over_local_derivative']:.5f}。")
    paragraphs += ["- 8 个探针坐标指纹可由保存的归一化初始速度重建，并与参考账本一致。但 ±h 的各次代理力未保存，只能独立核对已存差分向量的范数与两间距稳定性，不能宣称从完整原始力重做了差分。历史标签的同版本代理残差向量也未保留，κ=1.950382716 是保存的经验比较量，没有重新估计。",
                   "- 对所有保存相邻状态，计算 ‖rᵢ−rᵢ₋₁‖∞,₂ / ‖Xᵢ−Xᵢ₋₁‖₂，不用标量误差之差冒充向量变化。六条轨迹的最大割线均小于 κ，比例范围 0.40555–0.58388；沿保存折线路径的 e0+κℓ 没有点上超差。它可说明已采样路径上的保守性，不证明未见曲率、连续路径上界或迁移有效性。\n",
                   "图的证据与导出约定：183×151 mm 四联 quantitative grid，Python/matplotlib 唯一绘图后端，PDF/SVG 可编辑字体，PNG 600 dpi。a/b 为主证据，展示误差、预算、首次离散超差、右删失及独立的预设时限条带；c 比较两探针间距、每锚点三条路径中的最大割线和同一历史 κ；d 展示旧 16 条振子的逐轨迹加性分解。d 的等权均值为起始残差 15.42094% H0、曲率 36.51314% H0，总计 51.93408% H0；没有新振子轨迹或因果消融。无拟合、显著性检验或置信区间。\n",
                   f"渲染自动检查：画布外文字 {len(result['figure_qa']['text_outside_canvas'])} 项，SVG 可编辑文字 {result['figure_qa']['svg_editable_text_elements']} 项，PDF TrueType 字体嵌入={result['figure_qa']['pdf_embedded_truetype_font_reference']}；视觉复核状态：{result['figure_qa']['visual_review_status']}。\n",
                   "产物：",
                   "- [核对 JSON](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/analysis/execution_20260905/h2o/motion_verification.json)：原文件哈希、逐项检查、误差、限制、作图约定与导出核验。",
                   "- [derived CSV](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/analysis/execution_20260905/h2o/motion_derived/motion_cases.csv)：另含 motion_points、motion_residual_vectors、motion_forecasts、motion_directional_probes、motion_probe_vectors、oscillator_work_components 七份源数据表。",
                   "- [作图及分析脚本](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/experiments/analyze_forward_motion_20260905.py)、[PDF](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/manuscript/figures/forward_motion.pdf)、[PNG](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/manuscript/figures/forward_motion.png)、[SVG](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/manuscript/figures/forward_motion.svg)。",
                   "- [独立 LaTeX 片段](/Users/pengkang/Research/Project/Project_PYRAIMD/pyraimd2/manuscript/forward_motion_results.tex)：两段 P2a 结果及新图 caption；未修改其他 tex/refs。"]
    report.write_text("\n".join(paragraphs)+"\n")
    tex = r"""% Independent saved-data analysis; insert from the manuscript directory.
\subsection{Velocity interventions at two molecular anchors}
We compared six forward interventions obtained by scaling the same initial velocity at each of two water configurations by factors of $0.5$, $1$, and $2$, while keeping the surrogate fixed. Each record contains 20 force-evaluation states at 0.5\,fs spacing, spanning 0--9.5\,fs; the original experiment used 122 new reference evaluations, comprising 114 future trajectory states and eight directional probes. At the $0.10\,\mathrm{eV\,\mathring{A}^{-1}}$ force budget, the two $2\times$ trajectories first exceeded the budget at recorded times of 2.5 and 8.0\,fs, respectively, whereas the other four records were right-censored at 9.5\,fs (Fig.~\ref{fig:forward-motion}a,b). The preset empirical horizons were $(2.0,0.5,0)$\,fs for anchor A and $(1.0,0.5,0)$\,fs for anchor B, in increasing velocity-factor order. A zero horizon denotes failure to admit even the first 0.5\,fs forecast step, not an initial force-budget violation. Thus the forecasts were shorter than the observed discrete crossing times; these two-anchor observations do not establish an inverse-velocity lifetime law or accurate failure-time prediction. Saved forecast paths agree with the corresponding later trajectory positions. Source ordering supports the stated prospective protocol, but the retained records do not supply an independently timestamped pre-reference commitment.

Central directional differences at the initial anchors, with displacement magnitudes of 0.001 and 0.002\,\AA\ along the normalized full-configuration velocity, were stable: the relative differences between the two residual-derivative vectors were 0.00705\% and 0.00969\%. At the smaller displacement, the derivative norms were 0.775 for anchor A and 0.801 for anchor B, in units of $\mathrm{eV\,\mathring{A}^{-2}}$. The recorded empirical value was $\widehat\kappa=1.950$ in the same units (Fig.~\ref{fig:forward-motion}c). The maximum adjacent-state residual-vector secants on the six recorded paths ranged from $0.791$ to $1.139\,\mathrm{eV\,\mathring{A}^{-2}}$; all recorded points also satisfied the corresponding empirical $e_0+\widehat\kappa\ell$ comparison along the observed polygonal path. These measurements indicate conservativeness on the sampled paths, while neither local finite differences nor finite path secants certify unseen curvature or continuous-time force error. The molecular comparisons and the separately recorded oscillator work decomposition in Fig.~\ref{fig:forward-motion}d address distinct aspects of motion-dependent residual growth and accumulated work.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/forward_motion.pdf}
\caption{\textbf{Velocity interventions, empirical residual growth, and stored oscillator work.}
\textbf{a,b,} Maximum-atom Euclidean force error for three velocity factors at each of two water anchors. Lines connect the 20 saved states; the dashed horizontal line is the $0.10\,\mathrm{eV\,\mathring{A}^{-1}}$ budget. Open right-pointing markers identify the four records with no observed exceedance through 9.5\,fs; they do not denote infinite lifetimes or exclude between-point violations. Labels mark the first observed exceedances in the $2\times$ interventions. Lower strips give the preset horizons separately from the observed trajectories; 0\,fs means less than one admissible 0.5\,fs forecast step.
\textbf{c,} Norms of the saved central directional residual derivatives at $h=0.001$ and $0.002$\,\AA\ (staggered vertically for visibility), the largest adjacent-state residual-vector secant among the three paths at each anchor, and the recorded historical $\widehat\kappa$ (dashed). The direction has unit full-configuration Euclidean norm; derivative and secant outputs use the maximum-atom Euclidean norm, giving units of $\mathrm{eV\,\mathring{A}^{-2}}$. These are descriptive comparisons, not certified derivative bounds.
\textbf{d,} Additive residual-work contributions for all 16 previously stored, held-out class-horizon oscillator trajectories, normalized by each trajectory's initial reference energy $H_0$. The mean anchor-residual and curvature contributions are 15.42\% and 36.51\%, respectively; the dashed line marks their mean sum, 51.93\%. Bars show individual trajectories in original order, with no confidence intervals or significance tests. No new molecular or oscillator propagation, model inference, training, or reference calculation was performed for this analysis.}
\label{fig:forward-motion}
\end{figure*}
"""
    (ROOT / "manuscript/forward_motion_results.tex").write_text(tex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--draw-only", action="store_true")
    args = parser.parse_args()
    if not args.draw_only:
        analyze()
    if not args.analyze_only:
        draw()
