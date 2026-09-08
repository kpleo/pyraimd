"""Fixed origin/50-fs W stock-model control; no DFT, fitting, or downloads.

Run once: UV_OFFLINE=1 uv run --no-sync python experiments/w_stock_fairness_20260906.py
Only writes analysis/final_campaign_20260905/w_fairness_stock_20260906.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import warnings
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "analysis/execution_20260905/tungsten"
OUT = ROOT / "analysis/final_campaign_20260905/w_fairness_stock_20260906"
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def array_hash(arrays):
    h = hashlib.sha256()
    for k, v in sorted(arrays.items()):
        a = np.ascontiguousarray(v)
        h.update(json.dumps([k, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def model_hash(m):
    return array_hash({k: v.detach().cpu().numpy() for k, v in m.state_dict().items()})


def deny_network(event, args):
    if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
        raise RuntimeError("No network in the fixed stock comparison")


if __name__ == "__main__":
    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "plan.json").exists():
        raise RuntimeError("Existing experiment; do not overwrite or silently rerun")
    for k, folder in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "mplconfig"),
                      ("XDG_CACHE_HOME", "cache")):
        p = OUT / "runtime" / folder
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p)
    sys.addaudithook(deny_network)
    import numpy as np
    from ase import Atoms
    from ase.io import read
    old_plan = json.loads((OLD / "frozen_plan.json").read_text())
    selected = [0, 20]  # Shared origin and existing manuscript spike 50 fs, frame 38.
    files = [OLD / f for f in ("frozen_plan.json", "geometries_24.extxyz",
                               "labels_27_original.jsonl", "predictions/W24_00.npz",
                               "predictions/W24_20.npz")]
    files += [Path(x["path"]) for x in old_plan["model_files"]]
    files += [ROOT / "src/pyraimd2/surrogate/committee.py", Path(__file__)]
    for x in old_plan["model_files"]:
        assert sha(x["path"]) == x["sha256"]
    config = dict(old_plan["committee_constructor"], perturbation=0.0)
    plan = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Pre-fixed shared origin (canonical frame 0) and manuscript spike 50 fs (frame 38); no outcome-based selection",
        "geometry_rows": selected,
        "geometry_records": [old_plan["geometry_manifest"][i] for i in selected],
        "input_files": [{"path": str(p), "sha256": sha(p)} for p in files],
        "stock_constructor": config,
        "original_constructor": old_plan["committee_constructor"],
        "primary_reference": "Archived QE PBE-D3; D3 not removed in this experiment",
        "force_budget_ev_A": 0.2,
        "metrics": "Max atom norm and vector RMS, fixed original 40 core/392 matrix IDs; each member and equal-weight mean",
        "energy_metric": "Only fixed-pair energy increments; no absolute energy-offset fit",
        "cost_policy": "2 geometries x 2 members, one attempt, CPU 2 threads; no fitting/DFT/downloads",
    }
    write_json("plan.json", plan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        from pyraimd2.surrogate import CommitteeSurrogate

        class CapturedStock(CommitteeSurrogate):
            def _forward(self, member, batch, training):
                assert training is False
                result, batch_dict = super()._forward(member, batch, training)
                self.captured.append({"forces": result["forces"].detach().cpu().numpy().copy(),
                                      "energy": float(result["energy"].detach().cpu().reshape(-1)[0])})
                return result, batch_dict

            def finetune(self, labels):
                raise RuntimeError("No fitting")

        committee = CapturedStock(**config)
        committee._ensure_loaded()
        for m in committee._models:
            for p in m.parameters():
                p.requires_grad_(False)
        hashes_before = [model_hash(m) for m in committee._models]
        assert hashes_before[0] == old_plan["effective_model_identity"]["member_tensor_sha256"][0]
        # File checkpoints are locally trusted and hash-locked; validate zero-perturbation
        # models against each original file, not merely the constructor argument.
        for i, f in enumerate(old_plan["model_files"]):
            raw = torch.load(f["path"], map_location="cpu", weights_only=False).double()
            assert model_hash(raw) == hashes_before[i]
            del raw
        frames = read(OLD / "geometries_24.extxyz", ":")
        labels = {x["frame_index"]: x for x in map(json.loads, (OLD / "labels_27_original.jsonl").read_text().splitlines())}
        core = np.isin(np.arange(432), old_plan["initial_core_atom_indices_0based"])

        def metrics(f, ref):
            out = {}
            for k, mask in (("all", np.ones(432, bool)), ("core", core), ("matrix", ~core)):
                e = np.linalg.norm((f-ref)[mask], axis=1)
                out[k] = {"max_ev_A": float(e.max()), "vector_rms_ev_A": float(np.sqrt(np.mean(e**2))),
                          "atoms_above_0p2": int(np.count_nonzero(e > .2))}
            return out

        results, energies = [], []
        for gi in selected:
            meta = old_plan["geometry_manifest"][gi]
            a = frames[gi]
            assert array_hash({"numbers": np.asarray(a.numbers, dtype="<i8"),
                               "positions": np.asarray(a.positions, dtype="<f8"),
                               "cell": np.asarray(a.cell, dtype="<f8"),
                               "pbc": np.asarray(a.pbc, dtype="u1")}) == meta["coordinate_sha256"]
            clean = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
            committee.captured = []
            ti = time.perf_counter()
            pred = committee.predict(clean)
            wall = time.perf_counter()-ti
            fs = np.stack([x["forces"] for x in committee.captured])
            es = np.array([x["energy"] for x in committee.captured])
            assert fs.shape == (2, 432, 3) and np.isfinite(fs).all() and np.isfinite(es).all()
            assert np.allclose(fs.mean(axis=0), pred.forces, atol=1e-12, rtol=1e-12)
            old = np.load(OLD / "predictions" / (meta["geometry_id"] + ".npz"))
            unchanged = float(np.max(np.abs(fs[0] - old["member_forces"][0])))
            assert unchanged < 1e-10  # Shared member is a complete inference-path control.
            ref = np.asarray(labels[meta["representative_frame_index"]]["forces_ev_a"])
            np.savez_compressed(OUT / (meta["geometry_id"] + "_stock.npz"), stock_forces=fs,
                                stock_energies=es, stock_mean_forces=pred.forces,
                                stock_spread=pred.uncertainty, reference_forces=ref)
            row = {"geometry": meta, "inference_wall_s": wall,
                   "member0_reproduction_max_component_ev_A": unchanged,
                   "stock_MP0b3": metrics(fs[0], ref), "stock_MPA0": metrics(fs[1], ref),
                   "stock_mean": metrics(pred.forces, ref),
                   "old_perturbed_MPA0": metrics(old["member_forces"][1], ref),
                   "old_mean": metrics(old["mean_forces"], ref),
                   "perturbation_force_change": metrics(old["member_forces"][1], fs[1]),
                   "stock_spread_max_ev_A": float(pred.uncertainty.max()),
                   "old_spread_max_ev_A": float(old["spread"].max())}
            results.append(row)
            energies.append({"reference": labels[meta["representative_frame_index"]]["energy_ev"],
                             "stock_MP0b3": float(es[0]), "stock_MPA0": float(es[1]),
                             "stock_mean": float(es.mean()),
                             "old_perturbed_MPA0": float(old["member_energies"][1]),
                             "old_mean": float(old["mean_energy"])})
            print(json.dumps(row), flush=True)
        assert hashes_before == [model_hash(m) for m in committee._models]
        assert all(sha(x["path"]) == x["sha256"] for x in plan["input_files"])
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    delta = {k: energies[1][k]-energies[0][k] for k in energies[0]}
    summary = {"status": "complete", "plan_sha256": sha(OUT / "plan.json"),
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "model_tensor_sha256": hashes_before, "models_equal_original_file_tensors": True,
               "inputs_and_model_tensors_unchanged": True,
               "geometry_results": results, "raw_energies_ev": energies,
               "pair_energy_increment_ev": delta,
               "pair_increment_error_vs_PBE_D3_ev": {k: v-delta["reference"] for k, v in delta.items() if k != "reference"},
               "cost": {"geometries": 2, "member_forward_calls": 4, "new_DFT": 0,
                        "cloud_jobs": 0, "downloads": 0, "installed_packages": 0,
                        "elapsed_wall_s": time.perf_counter()-t0,
                        "inference_wall_s": sum(x["inference_wall_s"] for x in results),
                        "cpu_user_s": r1.ru_utime-r0.ru_utime, "cpu_system_s": r1.ru_stime-r0.ru_stime,
                        "max_rss_bytes_macos": r1.ru_maxrss, "threads": 2},
               "versions": {k: importlib.metadata.version(k) for k in ("torch", "mace-torch", "numpy", "ase")},
               "warnings": [{"category": x.category.__name__, "message": str(x.message)} for x in caught]}
    write_json("summary.json", summary)
    print(json.dumps({"complete": True, "cost": summary["cost"]}), flush=True)
