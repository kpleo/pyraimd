"""Scratch lifecycle hardening: archive content verification, identity
guards, receipt-failure semantics, the all-mode density chain, the
archived window and corrupt-record tolerance — fake pw.x only."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest
from ase import Atoms

from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import (
    DENSITY_MANIFEST,
    QeConfig,
    QeEngine,
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


def _make_handle(tmp_path, *, run_uuid="run-aaa0000001", request_id="req-1",
                 attempt_id="attempt-1", retention="results"):
    archive_dir = tmp_path / "runs" / "case-000000" / attempt_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    return scratch_mod.allocate(
        run_root=tmp_path / "runs", scratch_root=tmp_path / "tmp",
        run_uuid=run_uuid, backend_role="reference", request_id=request_id,
        attempt_id=attempt_id, archive_dir=archive_dir, retention=retention)


def _records(run_root: Path) -> list[dict]:
    root = run_root / scratch_mod.SCRATCH_RECORD_DIR
    return [json.loads(p.read_text())
            for p in sorted(root.rglob("*.json"))] if root.is_dir() else []


# --- F1: archive content verification -------------------------------------------

def test_archive_records_sha256_and_cleanup_reverifies(tmp_path):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "cleaned"
    assert all(len(entry["sha256"]) == 64 for entry in record["archived"])
    archive_dir = Path(record["archive_dir"])
    reparsed = parse_qe_output((archive_dir / "pw.out").read_text())
    assert reparsed.energy == pytest.approx(result.energy, rel=0, abs=1e-12)


def test_deleted_or_tampered_archive_keeps_scratch(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.in").write_text("in")
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.in", "pw.out"])
    record = handle.load_record()
    archived_bytes = (Path(record["archive_dir"]) / "pw.out").read_bytes()
    # archive content deleted: cleanup refuses, scratch stays
    (Path(record["archive_dir"]) / "pw.out").unlink()
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert "no longer verifies" in receipt["reason"]
    assert handle.scratch_dir.is_dir()
    # restored with the SAME SIZE but different content: caught by SHA-256
    tampered = archived_bytes[:1] + b"X" + archived_bytes[2:]
    assert len(tampered) == len(archived_bytes)
    (Path(record["archive_dir"]) / "pw.out").write_bytes(tampered)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert handle.scratch_dir.is_dir()
    # restored correctly: the retry cleans (cleanup_pending is retryable)
    (Path(record["archive_dir"]) / "pw.out").write_bytes(archived_bytes)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "cleaned"
    assert not handle.scratch_dir.exists()


def test_archive_rejects_escaping_names(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    for bad in ("/etc/passwd", "../escape", "a/../../b", ".."):
        with pytest.raises(scratch_mod.ScratchError):
            scratch_mod.archive(handle, [bad])
    assert handle.load_record()["state"] == "archive_failed"


def test_archive_dir_inside_scratch_is_refused(tmp_path):
    with pytest.raises(scratch_mod.ScratchError, match="independent"):
        scratch_mod.allocate(
            run_root=tmp_path / "runs", scratch_root=tmp_path / "tmp",
            run_uuid="run-bb", backend_role="reference", request_id="req-1",
            attempt_id="attempt-1",
            archive_dir=tmp_path / "tmp" / "run-bb" / "reference" / "req-1"
            / "attempt-1" / "nested",
            retention="results")


# --- F2: identity guards ----------------------------------------------------------

def test_parent_symlink_replacement_never_deletes_sibling(tmp_path):
    engine_b = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path / "b"),
                 scratch_root=str(tmp_path / "tmp"), retention="all"),
        run_root=tmp_path / "runs-b")
    handle_a = _make_handle(tmp_path, request_id="req-a")
    (handle_a.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle_a, ["pw.out"])
    engine_b.compute(_si(), label="si")  # the active sibling stays around
    handle_b = engine_b._last_scratch_handle
    record_b = handle_b.load_record()
    dir_b = Path(record_b["scratch_dir"])
    # replace A's request parent with a symlink to B's request parent
    import shutil
    parent_a = handle_a.scratch_dir.parent
    shutil.rmtree(parent_a)
    parent_a.symlink_to(dir_b.parent)
    receipt = scratch_mod.cleanup(handle_a)
    assert receipt["status"] in {"refused", "failed"}
    assert dir_b.is_dir()  # the active sibling is never deleted
    assert (dir_b / "pw.out").is_file()


def test_recreated_directory_identity_mismatch_refused(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    import shutil
    shutil.rmtree(handle.scratch_dir)
    handle.scratch_dir.mkdir()
    (handle.scratch_dir / "pw.out").write_text("out")
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert "identity changed" in receipt["reason"]
    # the replacement directory's original bytes are never touched
    assert (handle.scratch_dir / "pw.out").read_text() == "out"


def _replace_dir_simulating_number_reuse(handle):
    """Delete the attempt directory and recreate it empty at the same
    path, then hand-set the record's pinned dev/ino to the REPLACEMENT's
    actual values — a simulation of an OS recycling the numeric identity
    (the Linux CI condition), not claimed as a real Linux experiment."""
    import shutil
    shutil.rmtree(handle.scratch_dir)
    handle.scratch_dir.mkdir()
    (handle.scratch_dir / "pw.out").write_text("out")
    replacement = handle.scratch_dir.stat()
    record = json.loads(handle.record_path.read_text())
    record["identity"] = {"dev": replacement.st_dev,
                          "ino": replacement.st_ino}
    handle.record_path.write_text(json.dumps(record))
    return record


@pytest.mark.parametrize("case", ["missing", "wrong-token", "other-attempt",
                                  "symlink", "not-json", "unknown-schema",
                                  "different-attempt-fields"])
def test_ownership_marker_decides_despite_matching_numbers(tmp_path, case):
    """Even with every numeric identity check forced to pass (simulated
    number reuse), a directory without the valid per-attempt ownership
    marker is never reclaimed: missing / wrong-token / another attempt's
    marker / link / non-JSON / unknown schema / mismatched association
    are all refused with the reason, and the bytes are kept."""
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    record = _replace_dir_simulating_number_reuse(handle)
    marker = handle.scratch_dir / scratch_mod.SCRATCH_OWNERSHIP_MARKER
    if case == "wrong-token":
        marker.write_text(json.dumps({
            "record": scratch_mod.SCRATCH_OWNERSHIP_SCHEMA,
            "ownership": "0" * 32,
            "run_uuid": record["run_uuid"],
            "backend_role": record["backend_role"],
            "request_id": record["request_id"],
            "attempt_id": record["attempt_id"]}))
    elif case == "other-attempt":
        other = _make_handle(tmp_path, request_id="req-other")
        marker.write_text((other.scratch_dir
                           / scratch_mod.SCRATCH_OWNERSHIP_MARKER).read_text())
    elif case == "symlink":
        genuine = tmp_path / "genuine-marker"
        genuine.write_text(json.dumps({
            "record": scratch_mod.SCRATCH_OWNERSHIP_SCHEMA,
            "ownership": record["ownership"],
            "run_uuid": record["run_uuid"],
            "backend_role": record["backend_role"],
            "request_id": record["request_id"],
            "attempt_id": record["attempt_id"]}))
        marker.symlink_to(genuine)
    elif case == "not-json":
        marker.write_text("{ not json")
    elif case == "unknown-schema":
        marker.write_text(json.dumps({
            "record": "scratch-attempt-ownership-v0",
            "ownership": record["ownership"],
            "run_uuid": record["run_uuid"],
            "backend_role": record["backend_role"],
            "request_id": record["request_id"],
            "attempt_id": record["attempt_id"]}))
    elif case == "different-attempt-fields":
        marker.write_text(json.dumps({
            "record": scratch_mod.SCRATCH_OWNERSHIP_SCHEMA,
            "ownership": record["ownership"],
            "run_uuid": record["run_uuid"],
            "backend_role": record["backend_role"],
            "request_id": "req-someone-else",
            "attempt_id": record["attempt_id"]}))
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert "identity changed" in receipt["reason"]
    assert (handle.scratch_dir / "pw.out").read_text() == "out"


def test_ownership_rechecked_on_the_open_fd(tmp_path, monkeypatch):
    """The marker is verified again relative to the held target fd inside
    the deletion critical section: a swap after the path-level pre-check
    is still refused before anything is removed."""
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    original_verify = scratch_mod._verify_archived_content
    marker = handle.scratch_dir / scratch_mod.SCRATCH_OWNERSHIP_MARKER

    def swap_marker(record):
        marker.write_text(json.dumps({
            "record": scratch_mod.SCRATCH_OWNERSHIP_SCHEMA,
            "ownership": "0" * 32,
            "run_uuid": record["run_uuid"],
            "backend_role": record["backend_role"],
            "request_id": record["request_id"],
            "attempt_id": record["attempt_id"]}))
        return original_verify(record)

    monkeypatch.setattr(scratch_mod, "_verify_archived_content", swap_marker)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "failed"
    assert "identity changed" in receipt["error"]
    assert (handle.scratch_dir / "pw.out").read_text() == "out"


def test_record_without_ownership_is_refused_not_adopted(tmp_path):
    """A record without the per-attempt ownership token (written before
    it existed, or stripped) is refused conservatively — the manager
    never mints a new identity to adopt an unknown directory."""
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    record = json.loads(handle.record_path.read_text())
    record.pop("ownership")
    handle.record_path.write_text(json.dumps(record))
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert "ownership" in receipt["reason"]
    assert handle.scratch_dir.is_dir()


def test_dry_run_lists_bad_ownership_marker_as_kept(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    (handle.scratch_dir / scratch_mod.SCRATCH_OWNERSHIP_MARKER).unlink()
    dry = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=True)
    assert dry["reclaimable"] == []
    assert len(dry["kept"]) == 1
    assert "identity changed" in dry["kept"][0]["reason"]
    actual = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=False)
    assert actual["reclaimable"] == [] and actual["receipts"] == []
    assert handle.scratch_dir.is_dir()


def test_unsafe_identity_components_refused(tmp_path):
    for bad in ("../escape", "a/b", "/abs", "..", ""):
        with pytest.raises(scratch_mod.ScratchError):
            _make_handle(tmp_path, request_id=bad)


def test_concurrent_allocation_same_run_uuid_no_clobber(tmp_path):
    errors, handles = [], []

    def claim(n):
        try:
            handles.append(_make_handle(
                tmp_path, request_id=f"req-{n}"))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=claim, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(handles) == 6
    # every allocation's record exists and owner.json is a valid single doc
    assert len(_records(tmp_path / "runs")) == 6
    owner = json.loads((tmp_path / "tmp" / "run-aaa0000001"
                        / "owner.json").read_text())
    assert owner["run_uuid"] == "run-aaa0000001"


def test_flock_blocks_competitor_and_recovers_after_exit(tmp_path):
    """An active flock holder blocks a competing cleanup; once the holder
    exits (or is killed) the kernel releases the lock and the same object
    is safely reclaimable — a stale lock file alone never blocks."""
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    import subprocess
    import sys as _sys

    holder = subprocess.Popen([
        _sys.executable, "-c",
        ("import fcntl, os, time\n"
         f"fd = os.open({str(handle.record_path.with_suffix('.lock'))!r}, "
         "os.O_CREAT | os.O_WRONLY, 0o644)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('locked', flush=True)\n"
         "time.sleep(30)\n")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # wait until the holder actually holds the lock (no startup race)
        assert holder.stdout.readline().strip() == "locked"
        receipt = scratch_mod.cleanup(handle)
        assert receipt["status"] != "cleaned"
        assert handle.scratch_dir.is_dir()
    finally:
        holder.kill()
        holder.wait(timeout=10)
        # close the pipes: an unclosed holder stdout/stderr is a
        # ResourceWarning under -W error once the process is reaped
        holder.stdout.close()
        holder.stderr.close()
    # after the holder died the kernel released the lock: the retry works
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "cleaned"
    assert not handle.scratch_dir.exists()
    # and a stale lock FILE alone never blocks a new reclaimer
    handle2 = _make_handle(tmp_path, request_id="req-2")
    handle2.record_path.with_suffix(".lock").write_text("stale")
    (handle2.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle2, ["pw.out"])
    receipt = scratch_mod.cleanup(handle2)
    assert receipt["status"] == "cleaned"


# --- F3: receipt failure never loses the result -----------------------------------

def test_unpersistable_pre_delete_state_blocks_deletion(tmp_path, monkeypatch):
    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")

    original = scratch_mod.AttemptScratch.update_record

    def fail_on_pending(self, **fields):
        if fields.get("state") == "cleanup_pending":
            raise OSError(28, "No space left on device")
        return original(self, **fields)

    monkeypatch.setattr(scratch_mod.AttemptScratch, "update_record",
                        fail_on_pending)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the label always delivers
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "archived"  # nothing deleted
    assert Path(record["scratch_dir"]).is_dir()
    assert engine.last_attempt_records[-1]["scratch_cleanup"]["status"] == "failed"
    # the archived result is still fully readable
    reparsed = parse_qe_output(
        (Path(record["archive_dir"]) / "pw.out").read_text())
    assert reparsed.energy == pytest.approx(result.energy, rel=0, abs=1e-12)


def test_cleaned_persist_error_reported_not_raised(tmp_path, monkeypatch):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    original = scratch_mod.AttemptScratch.update_record

    def fail_on_cleaned(self, **fields):
        if fields.get("state") == "cleaned":
            raise OSError(28, "No space left on device")
        return original(self, **fields)

    monkeypatch.setattr(scratch_mod.AttemptScratch, "update_record",
                        fail_on_cleaned)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "cleaned"
    assert "persist_error" in receipt
    assert not handle.scratch_dir.exists()


def test_missing_record_refuses_without_deleting(tmp_path):
    handle = _make_handle(tmp_path)
    handle.record_path.unlink()
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert handle.scratch_dir.is_dir()


def test_manifest_mark_failure_never_raises(tmp_path):
    """The real marker on an unwritable manifest: best-effort, no raise,
    and it is only ever written after a confirmed reclaim."""
    import pyraimd2.engines.qe_engine as qe_module

    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    record = _records(tmp_path / "runs")[0]
    manifest = Path(record["archive_dir"]) / DENSITY_MANIFEST
    manifest.chmod(0o444)
    try:
        qe_module._mark_manifest_scratch_removed(
            Path(record["archive_dir"]))  # must not raise
    finally:
        manifest.chmod(0o644)


# --- F4: the all-mode density chain -------------------------------------------------

@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_all_mode_second_compute_is_a_density_start(tmp_path, engine_kind):
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 startpot_file=True, scratch_root=str(tmp_path / "tmp"),
                 retention="all"),
        run_root=tmp_path / "runs")
    engine.compute(_si(), label="first")
    engine.compute(_si(), label="second")
    assert engine.last_density_decision["start"] == "density"
    assert engine.last_attempt_records[-1]["start"] == "density"


# --- F5: archived window and corrupt records -----------------------------------------

def test_archived_window_reclaimed_by_clean_pending(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    # simulate a crash between archive and cleanup: no cleanup ran
    view = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=True)
    assert len(view["reclaimable"]) == 1
    done = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=False)
    assert done["receipts"][0]["status"] == "cleaned"
    assert not handle.scratch_dir.exists()


def test_all_mode_archived_is_never_auto_reclaimed(tmp_path):
    handle = _make_handle(tmp_path, retention="all")
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    view = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=True)
    assert view["reclaimable"] == []
    assert handle.scratch_dir.is_dir()


def test_corrupt_record_is_unreadable_never_crashing(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    handle.record_path.write_text("{ not json")
    view = scratch_mod.inspect_root(tmp_path / "tmp")
    assert view["runs"][0]["attempts"][0]["state"] == "unreadable"
    done = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=False)
    assert done["receipts"] == []
    assert handle.scratch_dir.is_dir()
    # wrong JSON type: a list is not a record either
    handle.record_path.write_text("[1, 2]")
    view = scratch_mod.inspect_root(tmp_path / "tmp")
    assert view["runs"][0]["attempts"][0]["state"] == "unreadable"


# --- C1/C2: user entries --------------------------------------------------------------

def test_cli_accepts_relative_root_from_caller_cwd(tmp_path, monkeypatch,
                                                   capsys):
    from pyraimd2.cli import main as cli_main

    engine = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    engine.compute(_si(), label="si")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["scratch", "inspect", "--root", "./tmp"]) == 0
    assert "cleaned" in capsys.readouterr().out
    assert cli_main(["scratch", "clean", "--root", "./tmp",
                     "--dry-run"]) == 0


def test_label_template_second_label_associates_correctly(tmp_path):
    import subprocess
    import sys as _sys

    work = tmp_path / "work"
    script_dir = tmp_path / "scripts"
    (work / "pseudos").mkdir(parents=True)
    fake = _success_script(script_dir)
    structure = tmp_path / "si2.extxyz"
    from ase.io import write as ase_write
    ase_write(structure, _si())
    label_py = Path(__file__).parents[2] / "examples" / "standalone_qe_label" / "label.py"
    env = dict(os.environ)
    for label in ("case-00", "case-01"):
        completed = subprocess.run(
            [_sys.executable, str(label_py), "--structure", str(structure),
             "--run-root", str(work / "runs"), "--scratch-root",
             str(work / "tmp"), "--pw-cmd", " ".join(fake),
             "--pseudo-dir", str(work / "pseudos"), "--label", label],
            capture_output=True, text=True, env=env, check=False)
        assert completed.returncode == 0, completed.stderr[-400:]
    first = json.loads((work / "runs" / "label-case-00.json").read_text())
    second = json.loads((work / "runs" / "label-case-01.json").read_text())
    assert first["archive_dir"] != second["archive_dir"]
    assert Path(second["archive_dir"]).is_dir()
    assert "case-01" in second["archive_dir"]
    assert (Path(second["archive_dir"]) / "pw.out").is_file()
    assert second["scratch_record"] != first["scratch_record"]
    second_record = json.loads(Path(second["scratch_record"]).read_text())
    assert second_record["archive_dir"] == second["archive_dir"]
