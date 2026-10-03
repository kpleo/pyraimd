"""export-pairs: saved MTS labels -> pairs.npz -> calibrate-scale.

Real chain on the harmonic uncalibrated MTS template (analytic backends,
zero external programs) plus the refusal classes.  Nothing here is a
material validation — it is software verification on toy potentials.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.config import load_config
from pyraimd2.store import Store
from pyraimd2.workflows import (
    ExportPairsError,
    export_pairs,
    run_workflow,
)
from pyraimd2.workflows.templates import write_template


def _tree_state(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _mts_run(tmp_path: Path, *, arm: str = "run_mts.toml") -> Path:
    demo = tmp_path / ("demo dir" if arm == "run_mts.toml"
                       else "demo scaled dir")  # space-carrying paths
    write_template("harmonic-compare", demo)
    run_workflow(load_config(demo / arm), verbose=False, handle_sigint=False)
    return demo / "runs" / ("mts" if arm == "run_mts.toml"
                            else "mts-scaled")


def _row_at_step(run_dir: Path, step: int):
    with Store(run_dir / "trajectory.db", read_only=True) as store:
        for row in store._db.select():
            if int(row.key_value_pairs["step"]) == step:
                return row
    raise AssertionError(f"no row at step {step}")


def _mutate_row_data(run_dir: Path, step: int, mutate) -> None:
    """Rewrite one row's data payload on a COPY of the run directory."""
    with Store(run_dir / "trajectory.db") as store:
        row = _row_at_step(run_dir, step)
        store._db.update(row.id, data=mutate(dict(row.data)))
        store._db.connection.commit()


def _forbid_backend_construction(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("export must never construct a backend")

    from pyraimd2.backends import registry
    from pyraimd2.workflows import setup
    monkeypatch.setattr(registry, "create_backend", forbidden)
    monkeypatch.setattr(registry, "backend_factory", forbidden)
    monkeypatch.setattr(setup, "build_backends", forbidden)
    monkeypatch.setattr(setup, "create_configured_backend", forbidden)


def test_full_chain_saved_labels_to_scale(tmp_path, monkeypatch, capsys):
    run_dir = _mts_run(tmp_path)
    before = _tree_state(run_dir)
    _forbid_backend_construction(monkeypatch)
    output = tmp_path / "pairs.npz"
    report = export_pairs(run_dir, evaluation_ids=[1, 2, 3],
                          output=output)
    assert report["frames"] == 3
    assert report["n_atoms"] == 2
    assert report["evaluation_ids"] == [1, 2, 3]
    # the source run directory is byte-identical afterwards
    assert _tree_state(run_dir) == before
    # NPZ forces are EXACTLY the rows' stored arrays, element-wise
    z = np.load(output, allow_pickle=False)
    assert all(z[key].dtype.kind != "O" for key in z.files)
    for index, step in enumerate((-1, 4, 8)):
        row = _row_at_step(run_dir, step)
        np.testing.assert_array_equal(
            z["reference_forces_eV_A"][index],
            np.asarray(row.data["engine"]["forces"], dtype=np.float64))
        np.testing.assert_array_equal(
            z["fast_forces_eV_A"][index],
            np.asarray(row.data["surrogate"]["forces"], dtype=np.float64))
        np.testing.assert_array_equal(
            z["positions_A"][index],
            np.asarray(row.toatoms().positions, dtype=np.float64))
    assert z["force_unit"] == "eV/angstrom"
    assert len(set(z["frame_ids"].tolist())) == 3
    assert str(z["reference_id"]).startswith("pyramid-reference-sha256:")
    assert str(z["fast_model_id"]).startswith("pyramid-fast-sha256:")
    provenance = json.loads(str(z["provenance_json"]))
    assert provenance["schema_version"] == 1
    assert provenance["driver"] == "mts-nve-respa"
    assert set(provenance["source_files"]) == {
        "trajectory.db", "events.jsonl", "manifest.json",
        "resolved_config.json"}
    assert [frame["evaluation_id"] for frame in provenance["frames"]] == \
        [1, 2, 3]
    assert [frame["step"] for frame in provenance["frames"]] == [-1, 4, 8]
    assert provenance["reference_backend"] == "harmonic-reference"
    assert provenance["fast_backend"] == "harmonic-surrogate"
    # calibrate-scale consumes the export UNMODIFIED and returns the
    # analytic coefficient (reference k=1.0 over fast k=0.8)
    scale_json = tmp_path / "scale.json"
    code = cli_main(["calibrate-scale", "--pairs", str(output),
                     "--output", str(scale_json)])
    assert code == 0
    capsys.readouterr()
    fit = json.loads(scale_json.read_text())
    assert fit["scale"] == pytest.approx(1.25, rel=1e-12)
    assert fit["frame_ids"] == z["frame_ids"].tolist()
    assert fit["reference_id"] == str(z["reference_id"])
    assert fit["fast_model_id"] == str(z["fast_model_id"])
    # the exported archive is accepted by the existing reader as-is
    from pyraimd2.surrogate.calibration import read_pairs_npz
    reference, fast, _provenance = read_pairs_npz(output)
    np.testing.assert_array_equal(reference, z["reference_forces_eV_A"])
    np.testing.assert_array_equal(fast, z["fast_forces_eV_A"])


def test_works_from_another_cwd_and_json_cli(tmp_path, monkeypatch,
                                             capsys):
    run_dir = _mts_run(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "pairs.npz"
    code = cli_main(["export-pairs", str(run_dir), "--evaluation-ids",
                     "1", "2", "3", "--output", str(output), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["frames"] == 3
    code = cli_main(["export-pairs", str(run_dir), "--evaluation-ids",
                     "99", "--output", str(tmp_path / "other.npz"),
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False
    assert "99" in payload["error"]["message"]
    assert not (tmp_path / "other.npz").exists()


def test_selection_and_output_refusals(tmp_path, monkeypatch):
    run_dir = _mts_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ExportPairsError, match="no committed record"):
        export_pairs(run_dir, evaluation_ids=[1, 99],
                     output=tmp_path / "x.npz")
    with pytest.raises(ExportPairsError, match="duplicate"):
        export_pairs(run_dir, evaluation_ids=[1, 1],
                     output=tmp_path / "x.npz")
    with pytest.raises(ExportPairsError, match=">= 1"):
        export_pairs(run_dir, evaluation_ids=[0],
                     output=tmp_path / "x.npz")
    with pytest.raises(ExportPairsError, match="at least one"):
        export_pairs(run_dir, evaluation_ids=[],
                     output=tmp_path / "x.npz")
    with pytest.raises(ExportPairsError, match="OUTSIDE"):
        export_pairs(run_dir, evaluation_ids=[1],
                     output=run_dir / "inside.npz")
    assert not list(run_dir.glob("*.npz"))          # nothing half-written
    # an existing output is kept by default; --force replaces only it
    output = tmp_path / "pairs.npz"
    output.write_bytes(b"pre-existing\n")
    with pytest.raises(ExportPairsError, match="--force"):
        export_pairs(run_dir, evaluation_ids=[1], output=output)
    assert output.read_bytes() == b"pre-existing\n"
    export_pairs(run_dir, evaluation_ids=[1], output=output, force=True)
    assert output.read_bytes() != b"pre-existing\n"
    # a symlinked output is refused; an alias to a source file doubly so
    symlink = tmp_path / "linked.npz"
    symlink.symlink_to(run_dir / "trajectory.db")
    with pytest.raises(ExportPairsError, match="symlink|source file"):
        export_pairs(run_dir, evaluation_ids=[1], output=symlink,
                     force=True)
    assert (run_dir / "trajectory.db").is_file()
    assert symlink.is_symlink()


def test_incomplete_boundary_is_not_a_pairing_frame(tmp_path):
    run_dir = _mts_run(tmp_path)
    broken = tmp_path / "broken"
    shutil.copytree(run_dir, broken)
    events_path = broken / "events.jsonl"
    lines = events_path.read_text().splitlines()
    items = [json.loads(line) for line in lines]
    boundary = max(i for i, e in enumerate(items)
                   if e.get("type") == "step_completed")
    del lines[boundary]                       # drop the last boundary only
    events_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ExportPairsError,
                       match="STEP_COMPLETED boundary"):
        export_pairs(broken, evaluation_ids=[5],
                     output=tmp_path / "x.npz")
    assert not (tmp_path / "x.npz").exists()


def test_missing_malformed_or_contradictory_labels_refused(tmp_path):
    run_dir = _mts_run(tmp_path)

    def drop_surrogate(data):
        data["surrogate"] = None
        return data

    def bad_shape(data):
        data["engine"] = dict(data["engine"],
                              forces=[[1.0, 2.0]])
        return data

    def non_finite(data):
        forces = np.array(data["surrogate"]["forces"], dtype=float)
        forces[0, 0] = np.nan
        data["surrogate"] = dict(data["surrogate"],
                                 forces=forces.tolist())
        return data

    for step, mutate, needle in ((-1, drop_surrogate, "fast"),
                                 (-1, bad_shape, "strictly"),
                                 (-1, non_finite, "non-finite")):
        broken = tmp_path / f"broken-{needle}"
        shutil.copytree(run_dir, broken)
        _mutate_row_data(broken, step, mutate)
        with pytest.raises(ExportPairsError, match=needle):
            export_pairs(broken, evaluation_ids=[1],
                         output=tmp_path / "x.npz")
        assert not (tmp_path / "x.npz").exists()


def test_identity_refusals(tmp_path):
    run_dir = _mts_run(tmp_path)

    # a correction wrapper run: stored labels are NOT raw base forces
    scaled_dir = _mts_run(tmp_path, arm="run_mts_scaled.toml")
    with pytest.raises(ExportPairsError, match="correction wrapper"):
        export_pairs(scaled_dir, evaluation_ids=[1],
                     output=tmp_path / "x.npz")

    # tampered manifest fingerprint disagrees with the event log
    tampered = tmp_path / "tampered"
    shutil.copytree(run_dir, tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["surrogate"]["fingerprint"] = "forged-fingerprint"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ExportPairsError, match="fingerprint"):
        export_pairs(tampered, evaluation_ids=[1],
                     output=tmp_path / "x.npz")

    # a second run_start with a different identity is never paired
    mixed = tmp_path / "mixed"
    shutil.copytree(run_dir, mixed)
    events_path = mixed / "events.jsonl"
    first = json.loads(events_path.read_text().splitlines()[0])
    first["model_id"] = "other-model#g0"
    with events_path.open("a") as handle:
        handle.write(json.dumps(first) + "\n")
    with pytest.raises(ExportPairsError, match="identit"):
        export_pairs(mixed, evaluation_ids=[1], output=tmp_path / "x.npz")

    # a commit re-pointed at the wrong row fails its digest binding
    rebound = tmp_path / "rebound"
    shutil.copytree(run_dir, rebound)
    events_path = rebound / "events.jsonl"
    lines = events_path.read_text().splitlines()
    items = [json.loads(line) for line in lines]
    target = next(i for i, e in enumerate(items)
                  if e.get("type") == "evaluation_committed"
                  and e.get("row_id") == 2)
    items[target]["row_id"] = 3
    lines[target] = json.dumps(items[target])
    events_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ExportPairsError, match="bindings"):
        export_pairs(rebound, evaluation_ids=[1, 2],
                     output=tmp_path / "x.npz")


def test_source_change_during_read_refused(tmp_path, monkeypatch):
    run_dir = _mts_run(tmp_path)
    import importlib
    ep_module = importlib.import_module("pyraimd2.workflows.export_pairs")
    original = ep_module._sha256_file
    calls = {"events.jsonl": 0}

    def racing(path):
        if Path(path).name == "events.jsonl":
            calls["events.jsonl"] += 1
            if calls["events.jsonl"] > 1:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(ep_module, "_sha256_file", racing)
    with pytest.raises(ExportPairsError, match="changed while being read"):
        export_pairs(run_dir, evaluation_ids=[1],
                     output=tmp_path / "x.npz")
    assert not (tmp_path / "x.npz").exists()
