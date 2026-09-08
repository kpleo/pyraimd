"""Budgeted prospective molecular experiment; no production-controller changes.

prepare uses only archived development labels. pilot and velocity make new
reference calls, with attempts counted before execution. Proposal logs precede
independent audit draws and reference evaluation. Hidden labels never enter
controller state. All units are ASE units, except reported physical time in fs.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
from ase import units
from pyraimd2.engines import PyscfEngine
from pyraimd2.engines.base import EngineResult
from pyraimd2.store import Store
from pyraimd2.surrogate import CommitteeSurrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/execution_20260905/h2o"
DB = ROOT / "analysis/h2o_streak/coverage_h2o.db"
MODEL = Path.home() / ".cache/mace/20231210mace128L0_energy_epoch249model"
EPS, DT_FS, AUDIT_P = .10, .5, .10
DT = DT_FS * units.fs
SEEDS = [2026090501, 2026090502]
START_INDICES = [0, 151]
ARMS = ["reference", "periodic", "calibrated", "horizon"]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def force_norm(x):
    return float(np.linalg.norm(x, axis=1).max())


def key(atoms):
    # Identical to the coordinate precision presented to the reference engine.
    return hashlib.sha256((str(atoms.numbers.tolist()) + "|" + ";".join(
        f"{x:.10f},{y:.10f},{z:.10f}" for x, y, z in atoms.positions / units.Bohr
    )).encode()).hexdigest()


def archived():
    return list(Store(DB).iter_labels("collect-h2o-nve-300K"))


def model():
    return CommitteeSurrogate(model=str(MODEL), n_members=4, seed=20250819,
                              epochs=50, device="cpu")


def fit(m, labels, tag):
    with (OUT / "training.log").open("a") as f, contextlib.redirect_stdout(f):
        print(tag, flush=True)
        report = m.finetune(labels[-64:])
    print(f"TRAIN {tag}: {report.wall_time_s:.1f}s", flush=True)
    return asdict(report)


def qhat(pairs):
    vals = sorted(e / (s + .001) for s, e in pairs[-64:])
    if not vals:
        return math.inf
    return vals[min(math.ceil((len(vals) + 1) * .95), len(vals)) - 1]


def initial(frame, seed, factor=1.):
    a = frame.copy()
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(3, 3)) * np.sqrt(a.get_masses()[:, None])
    p -= a.get_masses()[:, None] * p.sum(axis=0) / a.get_masses().sum()
    kinetic = .5 * np.sum(p * p / a.get_masses()[:, None])
    p *= math.sqrt((6 * .5 * units.kB * 300) / kinetic) * factor
    a.set_momenta(p)
    return a


def prepare():
    import torch
    if (OUT / "locked_protocol.json").exists():
        raise RuntimeError("Existing locked protocol: refuse to overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "development_spec.json", {
        "source_sha256": sha(DB), "model_sha256": sha(MODEL),
        "reference": "RKS PBE/def2-SVP, conv_tol=1e-9; neutral singlet",
        "epsilon_ev_A": EPS, "dt_fs": DT_FS, "audit_p": AUDIT_P,
        "seeds": SEEDS, "start_indices": START_INDICES,
        "fit_frames": list(range(64)), "calibration_frames": list(range(64, 96)),
        "development_scoring_frames": list(range(96, 302)),
        "period_candidates": [1, 2, 4, 8, 16], "q_multiplier_candidates": [1, 1.5, 2],
        "selection": "frozen-checkpoint archived replay; minimum reference fraction among <=5% observed accepted violations; no held-out forward labels",
        "growth_gamma": 2., "h_env_fs": 8., "recent_growth_labels": 4,
        "growth_status": "empirical vector secants, not certified DFT derivative bound",
        "new_reference_attempt_limits": {"pilot": 900, "velocity": 128},
        "pilot_evaluation_states": 100, "pilot_wall_limit_seconds": 3600,
        "online_updates": "every 8 newly revealed labels; current force stays frozen; last64 labels; original 50-epoch recipe",
        "verification": {"lambda": math.log(2), "eta": .05,
                         "target": "accepted frozen force record, all-time bound per arm"},
        "velocity": {"factors": [.5, 1, 2], "states": 20,
                     "directional_difference_h_A": [.001, .002]},
        "discretization": "velocity Verlet; on-step momenta stored; discrete force-point validation only",
    })
    frames = archived()
    m = model()
    report = fit(m, frames[:64], "common development64")
    torch.save(m.state_dict(), OUT / "common_checkpoint.pt")
    preds = [m.predict(a) for a, _ in frames]
    pairs = [(float(p.uncertainty.max()), force_norm(p.forces-r.forces))
             for p, (_, r) in zip(preds, frames)]
    calibration = pairs[64:96]
    candidates = []
    for kind, settings in [("periodic", [1, 2, 4, 8, 16]),
                           ("calibrated", [1, 1.5, 2])]:
        for setting in settings:
            memory = calibration.copy()
            accepted = errors = refs = 0
            for j, (s, e) in enumerate(pairs[96:]):
                accept = (j % setting != 0) if kind == "periodic" else qhat(memory)*(s+.001)*setting <= EPS
                accepted += int(accept)
                errors += int(accept and e > EPS)
                if not accept:
                    refs += 1
                    memory.append((s, e))
            candidates.append({"kind": kind, "setting": setting, "accepted": accepted,
                               "errors": errors, "references": refs,
                               "risk": errors/accepted if accepted else None})
    selected = {}
    for kind in ["periodic", "calibrated"]:
        eligible = [r for r in candidates if r["kind"] == kind and
                    (r["risk"] is None or r["risk"] <= .05)]
        if not eligible:
            eligible = [r for r in candidates if r["kind"] == kind]
            choice = min(eligible, key=lambda r: (r["risk"], r["references"]))
        else:
            choice = min(eligible, key=lambda r: r["references"])
        selected[kind] = choice["setting"]
    protocol = json.loads((OUT / "development_spec.json").read_text())
    protocol.update({"selected": selected, "checkpoint_sha256": sha(OUT / "common_checkpoint.pt"),
                     "calibration_pairs": calibration, "development_candidates": candidates,
                     "common_training": report, "development_new_reference_attempts": 0,
                     "source_script_sha256": sha(__file__)})
    write(OUT / "locked_protocol.json", protocol)
    write(OUT / "development_predictions.json", {"pairs": pairs})
    print(json.dumps({"prepared": True, "selected": selected, "candidates": candidates}), flush=True)


class ReferenceLedger:
    def __init__(self, phase, limit, wall_seconds):
        self.path = OUT / f"{phase}_reference_attempts.jsonl"
        if self.path.exists():
            raise RuntimeError("Refuse rerun over an existing reference ledger")
        self.path.touch()
        self.cache = {key(a): r for a, r in archived()}
        self.n = 0
        self.limit = limit
        self.deadline = time.monotonic() + wall_seconds
        self.engine = PyscfEngine()
        self.reference_seconds = 0.

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Phase wall-clock budget exhausted")

    def evaluate(self, a, tag):
        self.check()
        k = key(a)
        if k in self.cache:
            return self.cache[k], False
        if self.n >= self.limit:
            raise RuntimeError("Reference attempt budget exhausted")
        self.n += 1
        with self.path.open("a") as f:
            f.write(json.dumps({"attempt": self.n, "tag": tag, "geometry": k,
                               "status": "started"}) + "\n")
        signal.alarm(60)
        try:
            result = self.engine.compute(a)
        finally:
            signal.alarm(0)
        if not np.isfinite(result.forces).all() or not np.isfinite(result.energy):
            raise RuntimeError("Nonfinite reference")
        self.reference_seconds += result.wall_time_s
        self.cache[k] = result
        with self.path.open("a") as f:
            f.write(json.dumps({"attempt": self.n, "tag": tag, "geometry": k,
                               "status": "complete", "energy": result.energy,
                               "forces": result.forces.tolist(), "seconds": result.wall_time_s}) + "\n")
        return result, True


def kappa(m, refs):
    chosen = refs[-4:]
    xs = [a.positions-a.get_center_of_mass() for a, _ in chosen]
    rs = [m.predict(a).forces-r.forces for a, r in chosen]
    slopes = [force_norm(b-a)/np.linalg.norm(y-x) for a, b, x, y in
              zip(rs[:-1], rs[1:], xs[:-1], xs[1:]) if np.linalg.norm(y-x) > 1e-4]
    return 2 * max(slopes) if slopes else math.inf


def forecast(m, atoms, f_current, e0, kap):
    if not np.isfinite(kap) or e0 > EPS:
        return 0., []
    trial = atoms.copy()
    p = trial.get_momenta()
    masses = trial.get_masses()[:, None]
    f = f_current.copy()
    length, last_safe, trace = 0., 0., []
    for j in range(1, 17):
        old = trial.positions.copy()
        p += .5 * DT * f
        trial.positions += DT * p / masses
        f = m.predict(trial).forces
        p += .5 * DT * f
        length += float(np.linalg.norm(trial.positions-old))
        b = e0+kap*length
        trace.append([j*DT_FS, length, b])
        if b > EPS:
            break
        last_safe = j*DT_FS
    return last_safe, trace


def restored():
    import torch
    m = model()
    m.load_state_dict(torch.load(OUT / "common_checkpoint.pt", weights_only=False))
    return m


def trajectory(arm, seed, start, ledger, protocol, n_states=100, update_model=True,
               velocity_factor=1., save_final_model=True):
    frames = archived()
    a = initial(frames[start][0], seed, velocity_factor)
    m = None if arm == "reference" else restored()
    refs = frames[:64].copy()
    growth_refs = frames[92:96].copy()
    pairs = [tuple(x) for x in protocol["calibration_pairs"]]
    rng = np.random.default_rng([seed, 919])
    records, updates = [], []
    nvisible = 0
    anchor_x = None
    age, horizon, path_length, kap, anchor_e = 0., 0., 0., math.inf, math.inf
    p = a.get_momenta()
    previous_f = None
    counter_before = ledger.n
    t0 = time.monotonic()
    proposal_file = OUT / f"{arm}_{seed}_proposals.jsonl"
    data_file = OUT / f"{arm}_{seed}_steps.jsonl"
    with proposal_file.open("x") as proposals, data_file.open("x") as data:
        for step in range(n_states):
            ledger.check()
            if step:
                p += .5*DT*previous_f
                old_x = a.positions.copy()
                a.positions += DT*p/a.get_masses()[:, None]
                if anchor_x is not None:
                    path_length += float(np.linalg.norm(a.positions-old_x))
                    age += DT_FS
            pred = m.predict(a) if m else None
            s = float(pred.uncertainty.max()) if pred else 0.
            bound = qhat(pairs)*(s+.001)*protocol["selected"]["calibrated"] if pred else None
            motion_bound = anchor_e+kap*path_length if np.isfinite(kap) else math.inf
            if arm == "reference" or step == 0:
                accepted = False
            elif arm == "periodic":
                accepted = step % protocol["selected"]["periodic"] != 0
            elif arm == "calibrated":
                accepted = bound <= EPS
            else:
                accepted = bound <= EPS and age <= horizon+1e-10 and motion_bound <= EPS
            proposals.write(json.dumps({"step": step, "geometry": key(a), "accepted": bool(accepted),
                "surrogate_forces": pred.forces.tolist() if pred else None,
                "calibrated_bound": bound, "announced_horizon_fs": horizon,
                "anchor_age_fs": age, "motion_bound": motion_bound if np.isfinite(motion_bound) else None})+"\n")
            proposals.flush()
            audit = bool(rng.random() < AUDIT_P) if accepted else False
            truth, new = ledger.evaluate(a, f"{arm}:{seed}:{step}")
            drive = pred.forces.copy() if accepted else truth.forces.copy()
            if step:
                p += .5*DT*drive
            a.set_momenta(p)
            error = force_norm(pred.forces-truth.forces) if pred else 0.
            revealed = not accepted or audit
            rec = {"step": step, "time_fs": step*DT_FS, "accepted": bool(accepted),
                   "audit": audit, "revealed": bool(revealed), "new_reference": new,
                   "error_ev_A": error, "violation": bool(accepted and error > EPS),
                   "spread": s, "bound": bound, "positions": a.positions.tolist(),
                   "momenta_full": p.tolist(), "surrogate_forces": pred.forces.tolist() if pred else None,
                   "reference_forces": truth.forces.tolist(), "driving_forces": drive.tolist(),
                   "reference_energy_ev": truth.energy, "kinetic_energy_ev": a.get_kinetic_energy(),
                   "residual_power_ase": float(np.sum(a.get_velocities()*(drive-truth.forces))),
                   "anchor_age_fs": age, "horizon_fs": horizon,
                   "motion_bound": motion_bound if np.isfinite(motion_bound) else None,
                   "model_updates_before": len(updates)}
            records.append(rec)
            data.write(json.dumps(rec, allow_nan=False)+"\n")
            data.flush()
            if revealed and m:
                pairs.append((s, error))
                refs.append((a.copy(), truth))
                growth_refs.append((a.copy(), truth))
                nvisible += 1
                if update_model and nvisible % 8 == 0:
                    updates.append(fit(m, refs, f"{arm}:{seed}:{step}"))
                if arm == "horizon":
                    post = m.predict(a)
                    anchor_e = force_norm(post.forces-truth.forces)
                    kap = kappa(m, growth_refs)
                    horizon, forecast_trace = forecast(m, a, drive, anchor_e, kap)
                    anchor_x, age, path_length = a.positions.copy(), 0., 0.
                    with (OUT / f"horizon_{seed}_anchors.jsonl").open("a") as f:
                        f.write(json.dumps({"step": step, "e0": anchor_e,
                            "kappa": kap if np.isfinite(kap) else None,
                            "horizon_fs": horizon, "forecast": forecast_trace})+"\n")
            previous_f = drive
            if step % 25 == 0:
                print(f"PILOT {arm} seed={seed} step={step} calls={ledger.n} e={error:.4f}", flush=True)
    energies = np.array([r["reference_energy_ev"]+r["kinetic_energy_ev"] for r in records])
    power = np.array([r["residual_power_ase"] for r in records])
    work = np.r_[0., np.cumsum(.5*DT*(power[1:]+power[:-1]))]
    n = sum(r["accepted"] for r in records)
    v = sum(r["violation"] for r in records)
    d = sum(r["violation"] and r["audit"] for r in records)
    result = {"arm": arm, "seed": seed, "start_index": start, "states": len(records),
              "accepted": n, "violations": v, "audit_detections": d,
              "online_references": sum(r["revealed"] for r in records),
              "audits": sum(r["audit"] for r in records),
              "new_reference_attempts": ledger.n-counter_before,
              "risk": v/n if n else None,
              "sequential_upper": min(1., (math.log(2)*d+math.log(20))/(n*(-math.log(.95)))) if n else None,
              "max_Href_drift_ev": float(np.max(np.abs(energies-energies[0]))),
              "end_Href_drift_ev": float(energies[-1]-energies[0]),
              "max_work_balance_residual_ev": float(np.max(np.abs(energies-energies[0]-work))),
              "max_accepted_error_ev_A": max((r["error_ev_A"] for r in records if r["accepted"]), default=None),
              "training_updates": updates, "wall_seconds": time.monotonic()-t0}
    if m and save_final_model:
        import torch
        torch.save(m.state_dict(), OUT / f"{arm}_{seed}_final.pt")
    write(OUT / f"{arm}_{seed}_summary.json", result)
    print(json.dumps(result | {"training_updates": len(updates)}), flush=True)
    return result


def pilot():
    protocol = json.loads((OUT / "locked_protocol.json").read_text())
    ledger = ReferenceLedger("pilot", 900, 3600)
    summaries = []
    for seed, start in zip(SEEDS, START_INDICES):
        for arm in ARMS:
            summaries.append(trajectory(arm, seed, start, ledger, protocol))
            write(OUT / "pilot_summary.json", {"status": "partial", "arms": summaries,
                  "attempts": ledger.n, "reference_seconds": ledger.reference_seconds})
    write(OUT / "pilot_summary.json", {"status": "complete", "arms": summaries,
          "attempts": ledger.n, "reference_seconds": ledger.reference_seconds})


def velocity():
    ledger = ReferenceLedger("velocity", 128, 3600)
    frames = archived()
    m = restored()
    results = []
    for seed, start in zip(SEEDS, START_INDICES):
        base = initial(frames[start][0], seed)
        truth0, _ = ledger.evaluate(base, f"velocity:{start}:anchor")
        pred0 = m.predict(base)
        kap = kappa(m, frames[92:96]+[(base.copy(), truth0)])
        e0 = force_norm(pred0.forces-truth0.forces)
        for factor in [.5, 1., 2.]:
            a = initial(frames[start][0], seed, factor)
            pred = m.predict(a)
            horizon, forecast_trace = forecast(m, a, pred.forces, e0, kap)
            tag = f"velocity_{seed}_{factor:g}"
            write(OUT / f"{tag}_prediction.json", {"e0": e0, "kappa": kap,
                "horizon_fs": horizon, "forecast": forecast_trace,
                "coordinates_sha256": key(a), "factor": factor,
                "prediction_precedes_future_references": True})
            records = []
            p = a.get_momenta()
            for step in range(20):
                if step:
                    p += .5*DT*pred.forces
                    a.positions += DT*p/a.get_masses()[:, None]
                    pred = m.predict(a)
                    p += .5*DT*pred.forces
                a.set_momenta(p)
                truth, _ = ledger.evaluate(a, f"{tag}:{step}")
                error = force_norm(pred.forces-truth.forces)
                records.append({"step": step, "time_fs": step*DT_FS, "error": error,
                    "positions": a.positions.tolist(), "momenta_full": p.tolist(),
                    "surrogate_forces": pred.forces.tolist(), "reference_forces": truth.forces.tolist()})
            first = next((r["time_fs"] for r in records if r["error"] > EPS), None)
            results.append({"seed": seed, "factor": factor, "horizon_fs": horizon,
                            "first_crossing_fs": first, "right_censored": first is None,
                            "e0": e0, "kappa": kap})
            write(OUT / f"{tag}_steps.json", records)
            print(f"VELOCITY {tag}: horizon={horizon} first={first}", flush=True)
        direction = base.get_velocities()
        direction /= np.linalg.norm(direction)
        probes = []
        for h in [.001, .002]:
            residuals = []
            for sign in [-1, 1]:
                a = base.copy()
                a.positions += sign*h*direction
                truth, _ = ledger.evaluate(a, f"probe:{start}:{h}:{sign}")
                residuals.append(m.predict(a).forces-truth.forces)
            derivative = (residuals[1]-residuals[0])/(2*h)
            probes.append({"h_A": h, "directional_residual_derivative": derivative.tolist(),
                           "norm": force_norm(derivative)})
        write(OUT / f"velocity_{seed}_directional_probe.json", probes)
    write(OUT / "velocity_summary.json", {"status": "complete", "cases": results,
          "attempts": ledger.n, "reference_seconds": ledger.reference_seconds})


def main():
    import torch
    from pyscf import lib
    torch.set_num_threads(1)
    lib.num_threads(1)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60s reference timeout")))
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "pilot", "velocity"])
    args = parser.parse_args()
    {"prepare": prepare, "pilot": pilot, "velocity": velocity}[args.phase]()


if __name__ == "__main__":
    main()
