"""Unified scratch lifecycle: managed tmp root, exclusive
attempts, durable-result archival and idempotent reclaim — fake pw.x
only, no real QE.

State chain: allocate → run → parse-verified → archived → cleaned;
failures stay kept; archived + cleanup failure is safely retryable.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from ase import Atoms

from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import (
    DENSITY_MANIFEST,
    QeConfig,
    QeEngine,
    QeEngineError,
    parse_qe_output,
)
from pyraimd2.runtime import scratch as scratch_mod

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"


def _si() -> Atoms:
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                 cell=[5.43] * 3, pbc=True)


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                 | stat.S_IXOTH)
    return ("bash", str(script))


def _success_script(tmp_path: Path) -> tuple[str, ...]:
    return _fake_pwx(
        tmp_path,
        "#!/bin/bash\n"
        "mkdir -p tmp/pyraimd2.save && echo fake-density > "
        "tmp/pyraimd2.save/charge-density.dat\n"
        f"cat {FIXTURE.resolve()}\n")


def _records(run_root: Path) -> list[dict]:
    root = run_root / scratch_mod.SCRATCH_RECORD_DIR
    return [json.loads(p.read_text())
            for p in sorted(root.rglob("*.json"))] if root.is_dir() else []


def test_legacy_default_unchanged(tmp_path):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path)),
        run_root=tmp_path / "runs")
    result = engine.compute(_si(), label="si")
    attempt = Path(engine.last_attempt_records[-1]["directory"])
    assert result.forces is not None
    assert (attempt / "tmp" / "pyraimd2.save").is_dir()
    assert engine._last_density_dir is not None
    assert _records(tmp_path / "runs") == []


def test_results_lifecycle_archives_then_reclaims(tmp_path):
    root = tmp_path / "shared tmp"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(root), retention="results"),
        run_root=tmp_path / "runs")
    result = engine.compute(_si(), label="si")
    archive_dir = tmp_path / "runs" / "si-000000" / "attempt-1"
    records = _records(tmp_path / "runs")
    assert len(records) == 1
    record = records[0]
    assert record["state"] == "cleaned"
    # the scratch subtree is gone; the shared root and owner marker stay
    assert not Path(record["scratch_dir"]).exists()
    assert (root / record["run_uuid"] / "owner.json").is_file()
    assert root.is_dir()
    # the durable result is archived OUTSIDE scratch, re-readable
    assert (archive_dir / "pw.in").is_file()
    assert (archive_dir / "pw.out").is_file()
    manifest = json.loads((archive_dir / DENSITY_MANIFEST).read_text())
    assert manifest["scratch_removed"] is True
    reparsed = parse_qe_output((archive_dir / "pw.out").read_text())
    assert reparsed.energy == pytest.approx(result.energy, rel=0, abs=1e-12)
    assert reparsed.forces == pytest.approx(result.forces, rel=0, abs=1e-12)
    assert engine._last_density_dir is None
    assert engine.last_attempt_records[-1]["scratch_cleanup"]["status"] == "cleaned"


def test_all_mode_keeps_scratch_and_density_usable(tmp_path):
    root = tmp_path / "tmp"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(root), retention="all"),
        run_root=tmp_path / "runs")
    engine.compute(_si(), label="si")
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "kept"
    scratch_save = Path(record["scratch_dir"]) / "tmp" / "pyraimd2.save"
    assert scratch_save.is_dir()
    manifest = json.loads((tmp_path / "runs" / "si-000000" / "attempt-1"
                           / DENSITY_MANIFEST).read_text())
    assert "scratch_removed" not in manifest
    assert Path(manifest["save_dir"]) == scratch_save
    assert engine._last_density_dir is not None


def test_concurrent_engines_never_collide(tmp_path):
    root = tmp_path / "tmp"
    engines = [QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_success_script(tmp_path / name),
                 scratch_root=str(root), retention="results"),
        run_root=tmp_path / f"runs-{name}")
        for name in ("a", "b")]
    results = [engine.compute(_si(), label="si") for engine in engines]
    assert all(r.forces is not None for r in results)
    run_uuids = {engine._scratch_run_uuid for engine in engines}
    assert len(run_uuids) == 2
    for engine in engines:
        assert _records(engine.run_root)[0]["state"] == "cleaned"
    assert sorted(p.name for p in root.iterdir()) == sorted(run_uuids)


def test_failed_attempt_kept_retry_archived_and_cleaned(tmp_path):
    marker = tmp_path / "marker"
    body = ("#!/bin/bash\n"
            "mkdir -p tmp/pyraimd2.save && echo fake > "
            "tmp/pyraimd2.save/charge-density.dat\n"
            f"if [ ! -f {marker} ]; then touch {marker}; "
            "echo total garbage; exit 1; fi\n"
            f"cat {FIXTURE.resolve()}\n")
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo",
                 pw_cmd=_fake_pwx(tmp_path, body), max_retries=1,
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    states = sorted(r["state"] for r in _records(tmp_path / "runs"))
    assert states == ["cleaned", "failed_kept"]
    failed_dir = next(r for r in _records(tmp_path / "runs")
                      if r["state"] == "failed_kept")["scratch_dir"]
    assert Path(failed_dir).is_dir()  # failed attempts are never cleaned


def test_unsupported_combinations_and_modes_refused(tmp_path):
    root = tmp_path / "tmp"
    for kwargs in ({"startpot_file": True},
                   {"density_source": str(tmp_path / "elsewhere")}):
        with pytest.raises(ValueError, match="does not compose"):
            QeEngine(QeConfig(pseudo_dir="/pseudo", scratch_root=str(root),
                              retention="results", **kwargs),
                     run_root=tmp_path / "runs")
        with pytest.raises(ValueError, match="does not compose"):
            AseQeEngine(QeConfig(pseudo_dir="/pseudo",
                                 scratch_root=str(root),
                                 retention="results", **kwargs),
                        run_root=tmp_path / "runs-ase")
    with pytest.raises(ValueError, match="retention"):
        QeEngine(QeConfig(pseudo_dir="/pseudo", scratch_root=str(root),
                          retention="everything"),
                 run_root=tmp_path / "runs")


def test_cleanup_states_idempotent_and_guarded(tmp_path):
    root = tmp_path / "tmp"
    archive_dir = tmp_path / "runs" / "case-000000" / "attempt-1"
    archive_dir.mkdir(parents=True)
    handle = scratch_mod.allocate(
        run_root=tmp_path / "runs", scratch_root=root,
        run_uuid="run-deadbeef01", backend_role="reference",
        request_id="req-1", attempt_id="attempt-1",
        archive_dir=archive_dir, retention="results")
    (handle.scratch_dir / "pw.in").write_text("in")
    (handle.scratch_dir / "pw.out").write_text("out")
    # not archived → refused, never deleted
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert handle.scratch_dir.is_dir()
    # archive manually, then cleanup twice: cleaned → already_cleaned
    scratch_mod.archive(handle, ["pw.in", "pw.out"])
    first = scratch_mod.cleanup(handle)
    assert first["status"] == "cleaned"
    assert not handle.scratch_dir.exists()
    second = scratch_mod.cleanup(handle)
    assert second["status"] == "already_cleaned"
    assert handle.load_record()["state"] == "cleaned"
    # the shared root itself is never touched
    assert root.is_dir()


def test_archive_failure_keeps_scratch_and_fails_delivery(tmp_path,
                                                          monkeypatch):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")

    def boom(handle, files):
        raise scratch_mod.ScratchError("injected archive failure")

    monkeypatch.setattr(scratch_mod, "archive", boom)
    with pytest.raises(QeEngineError, match="archive"):
        engine.compute(_si(), label="si")
    assert engine.last_attempt_records[-1]["status"] == \
        "post_processing_failed"


def test_cleanup_failure_retryable_without_dft_rerun(tmp_path, monkeypatch):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")

    def boom(*args, **kwargs):
        raise OSError("injected deletion failure")

    monkeypatch.setattr(scratch_mod, "_delete_tree_via_fd", boom)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the label always delivers
    assert len(engine.last_attempt_records) == 1  # no DFT rerun
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "cleanup_pending"
    assert engine.last_attempt_records[-1]["scratch_cleanup"]["status"] == "failed"
    monkeypatch.undo()
    view = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=True)
    assert len(view["reclaimable"]) == 1
    done = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=False)
    assert done["receipts"][0]["status"] in {"cleaned", "already_cleaned"}
    assert _records(tmp_path / "runs")[0]["state"] == "cleaned"


def test_inspect_lists_managed_and_unknown(tmp_path):
    root = tmp_path / "tmp"
    (root / "stranger").mkdir(parents=True)
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(root), retention="all"),
        run_root=tmp_path / "runs")
    engine.compute(_si(), label="si")
    view = scratch_mod.inspect_root(root)
    assert view["unknown"] == ["stranger"]
    assert len(view["runs"]) == 1
    attempt = view["runs"][0]["attempts"][0]
    assert attempt["state"] == "kept"
    assert attempt["present_bytes"] > 0
    (root / "stranger" / "keep.txt").write_text("x")
    assert (root / "stranger" / "keep.txt").is_file()


def test_missing_scratch_dir_is_already_cleaned(tmp_path):
    root = tmp_path / "tmp"
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(root), retention="results"),
        run_root=tmp_path / "runs")
    engine.compute(_si(), label="si")
    record_path = next((tmp_path / "runs" / scratch_mod.SCRATCH_RECORD_DIR)
                       .rglob("*.json"))
    record = json.loads(record_path.read_text())
    handle = scratch_mod.AttemptScratch(
        run_uuid=record["run_uuid"], backend_role=record["backend_role"],
        request_id=record["request_id"], attempt_id=record["attempt_id"],
        scratch_dir=Path(record["scratch_dir"]),
        archive_dir=Path(record["archive_dir"]), record_path=record_path)
    again = scratch_mod.cleanup(handle)
    assert again["status"] == "already_cleaned"


def test_both_qe_paths_share_the_lifecycle(tmp_path):
    root = tmp_path / "tmp"
    outcomes = {}
    for name, engine in (
            ("qe", QeEngine(
                QeConfig(pseudo_dir="/pseudo",
                         pw_cmd=_success_script(tmp_path / "a"),
                         scratch_root=str(root), retention="results"),
                run_root=tmp_path / "runs-qe")),
            ("ase-qe", AseQeEngine(
                QeConfig(pseudo_dir="/pseudo",
                         pw_cmd=_success_script(tmp_path / "b"),
                         scratch_root=str(root), retention="results"),
                run_root=tmp_path / "runs-ase"))):
        result = engine.compute(_si(), label="si")
        record = _records(engine.run_root)[0]
        outcomes[name] = (engine, result, record)
    for name, (engine, result, record) in outcomes.items():
        assert record["state"] == "cleaned", name
        assert not Path(record["scratch_dir"]).exists(), name
        assert engine._last_density_dir is None, name
    assert outcomes["qe"][1].energy == pytest.approx(
        outcomes["ase-qe"][1].energy, rel=1e-7)


def test_scratch_root_normalized_against_construction_cwd(tmp_path,
                                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root="./tmp", retention="results"),
        run_root=tmp_path / "runs")
    assert engine.config.scratch_root == str((tmp_path / "tmp").resolve())
    engine.compute(_si(), label="si")
    assert _records(tmp_path / "runs")[0]["state"] == "cleaned"
    assert (tmp_path / "tmp").is_dir()
