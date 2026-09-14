"""Deletion critical-section and post-processing failure boundaries —
the review's R1–R5 scenarios as behavior tests, fake pw.x only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import QeConfig, QeEngine
from pyraimd2.runtime import scratch as scratch_mod

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"


def _si() -> Atoms:
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                 cell=[5.43] * 3, pbc=True)


def _fake_pwx(tmp_path: Path, body: str) -> tuple[str, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_pwx.sh"
    script.write_text(body)
    import stat
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


def _make_handle(tmp_path, *, retention="results", request_id="req-1"):
    archive_dir = tmp_path / "runs" / "case-000000" / "attempt-1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return scratch_mod.allocate(
        run_root=tmp_path / "runs", scratch_root=tmp_path / "tmp",
        run_uuid="run-ccc0000001", backend_role="reference",
        request_id=request_id, attempt_id="attempt-1",
        archive_dir=archive_dir, retention=retention)


def _records(run_root: Path) -> list[dict]:
    root = run_root / scratch_mod.SCRATCH_RECORD_DIR
    return [json.loads(p.read_text())
            for p in sorted(root.rglob("*.json"))] if root.is_dir() else []


# --- A: the full maintenance boundary -------------------------------------------------

@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_lock_open_enospc_delivers_result_with_reason(tmp_path, monkeypatch,
                                                      engine_kind):
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    original_open = scratch_mod.os.open

    def enospc_on_lock(path, *args, **kwargs):
        if str(path).endswith(".lock"):
            raise OSError(28, "No space left on device")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(scratch_mod.os, "open", enospc_on_lock)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the label always delivers
    assert len(engine.last_attempt_records) == 1  # exactly one fake call
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "failed"
    assert "lock" in receipt["error"]
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "archived"  # nothing deleted
    assert Path(record["scratch_dir"]).is_dir()
    for entry in record["archived"]:
        assert (Path(record["archive_dir"]) / entry["file"]).is_file()


@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_archive_read_eio_delivers_result_with_reason(tmp_path, monkeypatch,
                                                      engine_kind):
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")

    def eio(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(scratch_mod, "_verify_archived_content", eio)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    assert len(engine.last_attempt_records) == 1
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "failed"
    assert "re-verified" in receipt["error"]
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "cleanup_pending"  # retryable, not deleted
    assert Path(record["scratch_dir"]).is_dir()
    for entry in record["archived"]:
        assert (Path(record["archive_dir"]) / entry["file"]).is_file()


@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_all_kept_enospc_delivers_result_with_reason(tmp_path, monkeypatch,
                                                     engine_kind):
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="all"),
        run_root=tmp_path / "runs")
    original = scratch_mod.AttemptScratch.update_record

    def enospc_on_kept(self, **fields):
        if fields.get("state") == "kept":
            raise OSError(28, "No space left on device")
        return original(self, **fields)

    monkeypatch.setattr(scratch_mod.AttemptScratch, "update_record",
                        enospc_on_kept)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    assert len(engine.last_attempt_records) == 1
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "keep_persist_failed"
    assert "No space left" in receipt["error"]
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "archived"  # unmarked, but never lost
    assert Path(record["scratch_dir"]).is_dir()


# --- B: archive compares against the source -------------------------------------------

def test_archive_rejects_copy_that_returns_success_but_breaks(tmp_path,
                                                              monkeypatch):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("correct-output")
    original_copy = scratch_mod.shutil.copyfileobj

    def broken_copy(src, dst, *args, **kwargs):
        original_copy(src, dst, *args, **kwargs)
        dst.seek(0)
        dst.write(b"broken!")

    monkeypatch.setattr(scratch_mod.shutil, "copyfileobj", broken_copy)
    with pytest.raises(scratch_mod.ScratchError, match="does not match"):
        scratch_mod.archive(handle, ["pw.out"])
    assert handle.load_record()["state"] == "archive_failed"
    # the source was never archived nor deleted
    assert (handle.scratch_dir / "pw.out").read_text() == "correct-output"
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"


def test_archive_destination_cannot_land_in_scratch(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    nested = handle.scratch_dir / "nested"
    nested.mkdir()
    # an archive_dir whose child path resolves back into the scratch tree
    link = handle.archive_dir / "link"
    link.symlink_to(handle.scratch_dir)
    with pytest.raises(scratch_mod.ScratchError):
        scratch_mod.archive(handle, ["link/pw.out"])


# --- C: the deletion anchors on the verified directory object -------------------------

def test_parent_swap_during_hash_never_deletes_sibling(tmp_path, monkeypatch):
    handle_a = _make_handle(tmp_path, request_id="req-a")
    (handle_a.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle_a, ["pw.out"])
    engine_b = QeEngine(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path / "b"),
                 scratch_root=str(tmp_path / "tmp"), retention="all"),
        run_root=tmp_path / "runs-b")
    engine_b.compute(_si(), label="si")
    handle_b = engine_b._last_scratch_handle
    dir_b = handle_b.scratch_dir

    original_verify = scratch_mod._verify_archived_content

    def swap_during_verify(record):
        import shutil
        parent_a = handle_a.scratch_dir.parent
        shutil.rmtree(parent_a)
        parent_a.symlink_to(dir_b.parent)
        return original_verify(record)

    monkeypatch.setattr(scratch_mod, "_verify_archived_content",
                        swap_during_verify)
    receipt = scratch_mod.cleanup(handle_a)
    # the fd-anchored boundary re-check refuses: A never claims a cleanup,
    # and B is still allocated and readable
    assert receipt["status"] in {"refused", "failed"}
    assert "identity changed" in receipt.get(
        "error", receipt.get("reason", ""))
    assert dir_b.is_dir()
    assert (dir_b / "pw.out").is_file()
    assert handle_b.load_record()["state"] == "kept"


# --- D: strict record validation --------------------------------------------------------

def _archive_then_mangle(tmp_path, mangle):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    path = handle.record_path
    record = json.loads(path.read_text())
    mangle(record)
    path.write_text(json.dumps(record))
    return handle


@pytest.mark.parametrize("case", ["no-archived", "empty-archived", "no-state",
                                  "no-identity", "wrong-schema",
                                  "archived-wrong-type"])
def test_incomplete_records_refuse_without_raising(tmp_path, case):
    def mangle(record):
        if case == "no-archived":
            record.pop("archived")
        elif case == "empty-archived":
            record["archived"] = []
        elif case == "no-state":
            record.pop("state")
        elif case == "no-identity":
            record.pop("identity")
        elif case == "wrong-schema":
            record["record"] = "scratch-attempt-v1"
        elif case == "archived-wrong-type":
            record["archived"] = {"pw.out": "out"}

    handle = _archive_then_mangle(tmp_path, mangle)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "refused"
    assert handle.scratch_dir.is_dir()
    view = scratch_mod.inspect_root(tmp_path / "tmp")
    assert view["runs"][0]["attempts"][0]["state"] == "unreadable"


def test_valid_record_still_cleans(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(handle, ["pw.out"])
    assert scratch_mod.cleanup(handle)["status"] == "cleaned"


# --- N024-R1: the lock lifecycle is inside the maintenance boundary --------------------

@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_lock_write_enospc_never_leaks_lock_or_loses_result(
        tmp_path, monkeypatch, engine_kind):
    """The lock file carries no payload — a failing os.write changes
    nothing: the compute delivers, the cleanup proceeds, no fd or active
    lock is leaked, and the same process can finish the reclaim."""
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")

    def enospc(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(scratch_mod.os, "write", enospc)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the label always delivers
    assert len(engine.last_attempt_records) == 1  # exactly one fake call
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "cleaned"
    handle = engine._last_scratch_handle
    # no payload is ever written to the lock file — the kernel flock is
    # the whole protocol
    assert handle.record_path.with_suffix(".lock").read_bytes() == b""
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "cleaned"
    for entry in record["archived"]:
        assert (Path(record["archive_dir"]) / entry["file"]).is_file()
    # no active lock remains in this process: the retry is a clean no-op
    assert scratch_mod.cleanup(handle)["status"] == "already_cleaned"


@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_unlock_eio_keeps_the_cleaned_fact(tmp_path, monkeypatch,
                                           engine_kind):
    """A LOCK_UN that reports EIO after a completed reclaim: the closed
    fd still freed the kernel lock, the receipt keeps the actual cleaned
    fact plus the maintenance error, and the result is delivered."""
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    real_flock = scratch_mod.fcntl.flock

    def eio_on_unlock(fd, operation):
        if operation == scratch_mod.fcntl.LOCK_UN:
            raise OSError(5, "Input/output error")
        return real_flock(fd, operation)

    monkeypatch.setattr(scratch_mod.fcntl, "flock", eio_on_unlock)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    assert len(engine.last_attempt_records) == 1
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "cleaned"  # the actual fact is kept
    assert "Input/output error" in receipt["maintenance_error"]
    handle = engine._last_scratch_handle
    record = _records(tmp_path / "runs")[0]
    assert record["state"] == "cleaned"
    assert not Path(record["scratch_dir"]).exists()
    for entry in record["archived"]:
        assert (Path(record["archive_dir"]) / entry["file"]).is_file()
    # the fd was closed, so the kernel lock is gone: retry is a no-op
    assert scratch_mod.cleanup(handle)["status"] == "already_cleaned"


# --- N024-R2: every descent keeps the verified directory object -------------------------

def test_nested_directory_swap_at_open_never_enters_sibling(tmp_path,
                                                            monkeypatch):
    """The reviewer's nested-directory-swap-at-open: a child directory
    replaced by a link to a live sibling between the stat and the open
    is refused at the open (O_NOFOLLOW), never descended into — the
    allocated sibling keeps its original bytes."""
    handle_a = _make_handle(tmp_path, request_id="req-a")
    (handle_a.scratch_dir / "pw.out").write_text("a-out")
    scratch_mod.archive(handle_a, ["pw.out"])
    handle_b = _make_handle(tmp_path, request_id="req-b")
    (handle_b.scratch_dir / "pw.out").write_text("b-original-bytes")
    child = handle_a.scratch_dir / "subtree"
    child.mkdir()
    (child / "own-scratch").write_text("a")
    original_open = scratch_mod.os.open
    fired = {"swap": False}

    def child_swap(path, flags, *args, **kwargs):
        if str(path) == "subtree" and kwargs.get("dir_fd") is not None \
                and not fired["swap"]:
            fired["swap"] = True
            child.rename(child.with_name("original-subtree"))
            child.symlink_to(handle_b.scratch_dir,
                             target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(scratch_mod.os, "open", child_swap)
    receipt = scratch_mod.cleanup(handle_a)
    assert fired["swap"]  # the injection really fired at the critical gap
    assert receipt["status"] != "cleaned"
    assert receipt.get("error") or receipt.get("reason")  # explicable
    # the live sibling was never entered: state and original bytes intact
    assert handle_b.load_record()["state"] == "allocated"
    assert (handle_b.scratch_dir / "pw.out").read_bytes() \
        == b"b-original-bytes"


# --- N024-R3: the dry run shares the reclaimability judgment ----------------------------

def test_dry_run_lists_corrupt_archive_as_kept_with_reason(tmp_path):
    handle = _make_handle(tmp_path)
    (handle.scratch_dir / "pw.out").write_text("correct-output")
    scratch_mod.archive(handle, ["pw.out"])
    # same-length tampering: 14 bytes for 14 bytes
    (handle.archive_dir / "pw.out").write_text("damaged-output")
    root = tmp_path / "tmp"
    dry = scratch_mod.clean_pending(root, dry_run=True)
    assert dry["reclaimable"] == []
    assert len(dry["kept"]) == 1
    assert "no longer verifies" in dry["kept"][0]["reason"]
    actual = scratch_mod.clean_pending(root, dry_run=False)
    assert actual["reclaimable"] == [] and actual["receipts"] == []
    assert len(actual["kept"]) == 1
    assert handle.scratch_dir.is_dir()  # the scratch is kept


def test_dry_run_lists_identity_change_as_kept_and_cleans_valid(tmp_path):
    good = _make_handle(tmp_path, request_id="req-good")
    (good.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(good, ["pw.out"])
    bad = _make_handle(tmp_path, request_id="req-bad")
    (bad.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(bad, ["pw.out"])
    # recreate the attempt directory: the object identity no longer
    # matches the dev/ino pinned at allocation
    import shutil
    shutil.rmtree(bad.scratch_dir)
    bad.scratch_dir.mkdir()
    (bad.scratch_dir / "pw.out").write_text("out")
    root = tmp_path / "tmp"
    dry = scratch_mod.clean_pending(root, dry_run=True)
    assert [r["request_id"] for r in dry["reclaimable"]] == ["req-good"]
    kept = {k["request_id"]: k["reason"] for k in dry["kept"]}
    assert "identity changed" in kept["req-bad"]
    actual = scratch_mod.clean_pending(root, dry_run=False)
    assert [r["request_id"] for r in actual["reclaimable"]] == ["req-good"]
    assert actual["receipts"][0]["status"] == "cleaned"
    assert not good.scratch_dir.exists()
    assert bad.scratch_dir.is_dir()  # the changed object is never touched


# --- entry protection: a state-query error is never "absent" ----------------------------

@pytest.mark.parametrize("engine_kind", ["qe", "ase-qe"])
def test_exists_eio_at_cleanup_entry_delivers_result_with_reason(
        tmp_path, monkeypatch, engine_kind):
    """An EIO from the entry's scratch_dir.exists() query (an I/O error
    is not "the directory is absent"): the successful compute still
    delivers, the receipt is failed with the reason, the record stays
    archived (never already_cleaned), and scratch + archive survive."""
    build = {"qe": QeEngine, "ase-qe": AseQeEngine}[engine_kind]
    engine = build(
        QeConfig(pseudo_dir="/pseudo", pw_cmd=_success_script(tmp_path),
                 scratch_root=str(tmp_path / "tmp"), retention="results"),
        run_root=tmp_path / "runs")
    original_cleanup = scratch_mod.cleanup
    original_exists = Path.exists

    def cleanup_with_eio(handle):
        def fail_exists(path):
            if path == handle.scratch_dir:
                raise OSError(5, "Input/output error")
            return original_exists(path)
        with patch.object(Path, "exists", fail_exists):
            return original_cleanup(handle)

    monkeypatch.setattr(scratch_mod, "cleanup", cleanup_with_eio)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the label always delivers
    assert len(engine.last_attempt_records) == 1  # exactly one fake call
    receipt = engine.last_attempt_records[-1]["scratch_cleanup"]
    assert receipt["status"] == "failed"
    assert "Input/output error" in receipt["error"]
    handle = engine._last_scratch_handle
    record = _records(tmp_path / "runs")[0]
    # a query failure is never misclassified as "directory absent"
    assert record["state"] == "archived"
    assert handle.scratch_dir.is_dir()
    for entry in record["archived"]:
        assert (handle.archive_dir / entry["file"]).is_file()


def test_dry_run_lists_exists_query_error_as_kept(tmp_path, monkeypatch):
    """The side-effect-free eligibility query reports the same EIO as a
    kept reason; the batch still completes for every other entry."""
    bad = _make_handle(tmp_path, request_id="req-bad")
    (bad.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(bad, ["pw.out"])
    good = _make_handle(tmp_path, request_id="req-good")
    (good.scratch_dir / "pw.out").write_text("out")
    scratch_mod.archive(good, ["pw.out"])
    original_exists = Path.exists

    def fail_exists(path):
        if path == bad.scratch_dir:
            raise OSError(5, "Input/output error")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)
    dry = scratch_mod.clean_pending(tmp_path / "tmp", dry_run=True)
    assert [r["request_id"] for r in dry["reclaimable"]] == ["req-good"]
    kept = {k["request_id"]: k["reason"] for k in dry["kept"]}
    assert "could not be queried" in kept["req-bad"]
    assert bad.scratch_dir.is_dir()


# --- F: the demo never deletes user data ----------------------------------------------

def test_demo_refuses_existing_non_empty_directory(tmp_path):
    demo = Path(__file__).parents[2] / "examples" / "standalone_qe_label" \
        / "demo_fake.sh"
    work = tmp_path / "existing dir"
    work.mkdir()
    sentinel = work / "keep.txt"
    sentinel.write_text("sentinel")
    completed = subprocess.run(
        ["sh", str(demo), str(work)], capture_output=True, text=True,
        check=False)
    assert completed.returncode == 2
    assert "refusing" in completed.stderr
    assert sentinel.read_text() == "sentinel"  # nothing was touched


def test_demo_runs_in_a_new_path_with_spaces(tmp_path):
    demo = Path(__file__).parents[2] / "examples" / "standalone_qe_label" \
        / "demo_fake.sh"
    work = tmp_path / "new demo dir"
    env = dict(os.environ, PYTHON=sys.executable,
               PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    completed = subprocess.run(
        ["sh", str(demo), str(work)], capture_output=True, text=True,
        check=False, env=env)
    assert completed.returncode == 0, completed.stderr[-400:]
    assert "cleaned" in completed.stdout
