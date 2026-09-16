"""Density restart registry: publish transaction recovery, validity closure
and ownership boundaries.  Synthetic role files and real
CheckpointManager/scratch.allocate fixtures only — never real QE validation.
Deletion exists only through execute_density_reclaim: lock-held,
reference-re-reading, tombstoned first.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.runtime import restart as rr
from pyraimd2.runtime import scratch as scratch_mod
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.events import EventLog
from pyraimd2.runtime.restart import (
    RestartError,
    compute_references,
    execute_density_reclaim,
    inspect_density_registry,
    latest_density_generation,
    plan_density_reclaim,
    publish_density_generation,
    publish_density_generation_from_attempt,
    resolve_density_for_resume,
)


def _source(run_dir: Path, *, density_name="charge-density.dat",
            content: bytes = b"density-v1", name="eval-000000") -> Path:
    """A run-owned attempt directory holding a fake .save tree."""
    attempt = run_dir / "calculations" / name
    save = attempt / "tmp" / "pyraimd2.save"
    save.mkdir(parents=True, exist_ok=True)
    (save / density_name).write_bytes(content)
    (save / "data-file-schema.xml").write_text("<xml/>")
    return attempt


def _prov(request_id: str, *, attempt_id: str = "attempt-1") -> dict:
    return {"run_id": "run",
            "attempt": {"request_id": request_id, "attempt_id": attempt_id},
            "reference_fingerprint": "qe-pbe-d3:deadbeef",
            "nat": 2, "species": ["Si"], "disk_io": "nowf",
            "source": {"kind": "attempt", "directory": "calculations/eval-000000"}}


def _files(density_name: str = "charge-density.dat") -> list[tuple[str, str]]:
    return [(f"tmp/pyraimd2.save/{density_name}", "charge-density"),
            ("tmp/pyraimd2.save/data-file-schema.xml", "metadata")]


def test_publish_normal_and_idempotent_retry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = _source(run_dir)
    first = publish_density_generation(run_dir, source, files=_files(),
                                       provenance=_prov("req-1"))
    assert first["generation"] == 1 and first["reused"] is False
    latest = latest_density_generation(run_dir)
    assert latest["generation"] == 1
    again = publish_density_generation(run_dir, source, files=_files(),
                                       provenance=_prov("req-1"))
    assert again["reused"] is True and again["generation"] == 1
    (source / "tmp" / "pyraimd2.save" / "charge-density.dat").write_bytes(b"v2")
    with pytest.raises(RestartError, match="immutable"):
        publish_density_generation(run_dir, source, files=_files(),
                                   provenance=_prov("req-1"))


def test_publish_rejects_external_and_escapes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "charge-density.dat").write_bytes(b"x")
    with pytest.raises(RestartError, match="outside the run directory"):
        publish_density_generation(run_dir, outside,
                                   files=[("charge-density.dat", "charge-density")],
                                   provenance=_prov("req-x"))
    source = _source(run_dir)
    with pytest.raises(RestartError, match="escapes"):
        publish_density_generation(run_dir, source,
                                   files=[("../escape", "charge-density")],
                                   provenance=_prov("req-y"))
    save = source / "tmp" / "pyraimd2.save"
    (save / "linked.dat").symlink_to(outside / "charge-density.dat")
    with pytest.raises(RestartError, match="outside the source|not a regular file"):
        publish_density_generation(run_dir, source,
                                   files=[("tmp/pyraimd2.save/linked.dat",
                                           "charge-density")],
                                   provenance=_prov("req-z"))


def test_interruption_before_commit_leaves_classified_tmp(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _source(run_dir)
    registry = run_dir / "restart" / "density"
    registry.mkdir(parents=True)
    leftover = registry / ".tmp-g000007"
    leftover.mkdir()
    (leftover / "partial").write_bytes(b"half")
    view = inspect_density_registry(run_dir)
    assert view["latest"] is None and view["state"] == "ok" or view["state"] == "fresh"
    tmp_resources = [r for r in view["resources"]
                     if r["kind"] == "publish_tmp_leftover"]
    assert [r["name"] for r in tmp_resources] == [".tmp-g000007"]
    plan = plan_density_reclaim(run_dir)
    assert plan["resources"][0]["decision"] == "hold"
    published = publish_density_generation(run_dir, _source(run_dir),
                                           files=_files(), provenance=_prov("req-1"))
    assert published["generation"] == 8


def test_crash_at_state_commit_leaves_previous_state(tmp_path: Path, monkeypatch) -> None:
    """Failure injected at the REAL os.replace boundary of the state commit
    (the temp state file is fully written first): the previous state stays
    intact, the committed generation is never-attached (hold), and an
    idempotent retry with no newer progress attaches it."""
    run_dir = tmp_path / "run"
    source = _source(run_dir)
    publish_density_generation(run_dir, source, files=_files(),
                               provenance=_prov("req-0"))  # g1 attached
    real_replace = rr.os.replace

    def fail_state_replace(src, dst):
        if Path(dst).name == "state.json":
            raise OSError("crash")
        return real_replace(src, dst)

    monkeypatch.setattr(rr.os, "replace", fail_state_replace)
    with pytest.raises(OSError, match="crash"):
        publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                   files=_files(), provenance=_prov("req-1"))
    monkeypatch.undo()
    monkeypatch.undo()
    view = inspect_density_registry(run_dir)
    assert view["latest"] == 1            # previous state intact
    assert view["attach_history"] == [1]  # g2 never provably attached
    plan = plan_density_reclaim(run_dir)
    g2 = next(r for r in plan["resources"]
              if r["kind"] == "generation" and r["generation"] == 2)
    assert g2["decision"] == "hold" and "never attached" in g2["reasons"][0]
    retried = publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                         files=_files(), provenance=_prov("req-1"))
    assert retried["reused"] is True
    assert latest_density_generation(run_dir)["generation"] == 2


def test_first_publish_crash_then_retry_recovers(tmp_path: Path, monkeypatch) -> None:
    """First publish: the fresh registry persists an explicit empty state
    BEFORE any generation write; a crash at the first state commit leaves
    g1 committed-unattached, and the same request's retry attaches it."""
    run_dir = tmp_path / "run"
    real_replace = rr.os.replace

    def fail_state_replace(src, dst):
        # fail only the state COMMIT (payload sets a latest), not the fresh
        # registry's explicit empty-state write
        if Path(dst).name == "state.json" and b'"latest": 1' in Path(src).read_bytes():
            raise OSError("crash")
        return real_replace(src, dst)

    monkeypatch.setattr(rr.os, "replace", fail_state_replace)
    with pytest.raises(OSError, match="crash"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov("req-1"))
    monkeypatch.undo()
    view = inspect_density_registry(run_dir)
    # the empty state was persisted first: the registry is NOT 'corrupt'
    assert view["state"] == "ok"
    assert view["latest"] is None and view["attach_history"] == []
    g1 = next(r for r in view["resources"] if r["kind"] == "generation")
    assert g1["generation"] == 1
    retried = publish_density_generation(run_dir, _source(run_dir),
                                         files=_files(), provenance=_prov("req-1"))
    assert retried["reused"] is True
    assert latest_density_generation(run_dir)["generation"] == 1


def test_corrupt_state_refuses_all_publish_paths(tmp_path: Path) -> None:
    """An existing registry with a damaged state file refuses every publish
    (no state-or-default reset): new requests AND reuse retries alike."""
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    (run_dir / "restart" / "density" / "state.json").write_text(
        '{"latest": 1, "attached": []}')  # self-contradictory: 1 not attached
    with pytest.raises(RestartError, match="corrupt"):
        publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                   files=_files(), provenance=_prov("req-2"))
    with pytest.raises(RestartError, match="corrupt"):
        publish_density_generation(run_dir, _source(run_dir),
                                   files=_files(), provenance=_prov("req-1"))
    # and nothing was written: no new generation, state untouched
    view = inspect_density_registry(run_dir)
    assert [r["generation"] for r in view["resources"]
            if r["kind"] == "generation"] == [1]


def test_stale_retry_never_moves_the_pointer_back(tmp_path: Path) -> None:
    """g2 committed but its attach crashed; g3 then publishes fine; a later
    retry of g2 returns the existing resource WITHOUT moving latest back."""
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-0"))
    orig_commit = rr._commit_latest

    def crash_for_g2(registry, generation, state):
        if generation == 2:
            raise OSError("crash")
        orig_commit(registry, generation, state)

    rr._commit_latest = crash_for_g2
    try:
        with pytest.raises(OSError, match="crash"):
            publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                       files=_files(), provenance=_prov("req-1"))
    finally:
        rr._commit_latest = orig_commit
    publish_density_generation(run_dir, _source(run_dir, name="eval-000002"),
                               files=_files(), provenance=_prov("req-2"))
    assert latest_density_generation(run_dir)["generation"] == 3
    retried = publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                         files=_files(), provenance=_prov("req-1"))
    assert retried["reused"] is True
    assert latest_density_generation(run_dir)["generation"] == 3  # no rollback
    plan = plan_density_reclaim(run_dir)
    g2 = next(r for r in plan["resources"] if r.get("generation") == 2)
    assert g2["decision"] == "hold"  # superseded before ever attaching


def test_payload_fsync_failure_prevents_commit(tmp_path: Path, monkeypatch) -> None:
    """The copied payload is fsynced BEFORE the generation commit: an fsync
    failure during the copy leaves neither a generation nor a pointer."""
    run_dir = tmp_path / "run"
    _source(run_dir)

    def no_fsync_files(dst, *a, **k):
        raise OSError("fsync failed mid-copy")

    monkeypatch.setattr(rr, "_copy_file_fsynced", no_fsync_files)
    with pytest.raises(OSError, match="fsync"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov("req-1"))
    view = inspect_density_registry(run_dir)
    assert view["latest"] is None
    assert not [r for r in view["resources"] if r["kind"] == "generation"]


def test_corrupt_state_is_not_fresh_registry(tmp_path: Path) -> None:
    """A run with published generations whose state file is truncated is
    CORRUPT, never 'never published': resume is unresolvable and reclaim is
    blocked."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    (run_dir / "restart" / "density" / "state.json").write_text("{truncated")
    view = inspect_density_registry(run_dir)
    assert view["state"] == "corrupt"
    resolved = resolve_density_for_resume(run_dir)
    assert resolved["branch"] == "unresolvable"
    assert "cannot be determined" in resolved["reason"]
    plan = plan_density_reclaim(run_dir)
    assert plan["blocked"]
    assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])
    # and a genuinely fresh run is the FRESH branch, a different state
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert inspect_density_registry(fresh)["state"] == "fresh"
    assert resolve_density_for_resume(fresh)["branch"] == \
        "external_initialization_required"


def test_state_missing_with_generations_is_corrupt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    (run_dir / "restart" / "density" / "state.json").unlink()
    view = inspect_density_registry(run_dir)
    assert view["state"] == "corrupt"
    assert plan_density_reclaim(run_dir)["blocked"]


def _write_real_checkpoints(run_dir: Path, density_refs: dict[int, int | None]) -> None:
    manager = CheckpointManager(run_dir)
    for generation, density_generation in sorted(density_refs.items()):
        extra = ({} if density_generation is None else
                 {"density_generation": density_generation})
        manager.write(generation, {"step": generation},
                      {"positions": np.zeros((1, 3))}, extra)


def test_references_and_reclaim_plan_with_real_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 1, 2: 2})
    plan = plan_density_reclaim(run_dir)
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[1]["decision"] == "keep" and "retained checkpoint 1" in by_gen[1]["reasons"][0]
    assert by_gen[2]["decision"] == "keep"
    assert by_gen[3]["decision"] == "keep" and by_gen[3]["reasons"] == ["latest pointer"]
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-4"))
    plan = plan_density_reclaim(run_dir)
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[3]["decision"] == "reclaim_candidate"
    assert by_gen[4]["decision"] == "keep"
    assert plan["dry_run"] is True
    assert all((run_dir / "restart" / "density" / f"g{i:06d}").is_dir()
               for i in (1, 2, 3, 4))


def test_unknown_checkpoint_schema_blocks_not_legacy(tmp_path: Path) -> None:
    """A real checkpoint whose manifest is mutated to an unknown schema is
    NOT a legacy checkpoint: reclaim blocks with the reason."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 1, 2: 2})
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-3"))  # g2 superseded
    path = run_dir / "checkpoints" / "1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_schema_version"] = 99
    path.write_text(json.dumps(manifest))
    plan = plan_density_reclaim(run_dir)
    assert any("unknown schema" in b for b in plan["blocked"])
    assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])
    # the resume path refuses the same manifest
    resolved = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert resolved["branch"] == "unresolvable"


def test_resume_prefers_checkpoint_reference_over_latest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 1, 2: 2})
    resolved = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert resolved["branch"] == "ok" and resolved["generation"] == 1
    _write_real_checkpoints(run_dir, {3: None})
    legacy = resolve_density_for_resume(run_dir, checkpoint_generation=3)
    assert legacy["branch"] == "external_initialization_required"
    _write_real_checkpoints(run_dir, {4: 99})
    missing = resolve_density_for_resume(run_dir, checkpoint_generation=4)
    assert missing["branch"] == "unresolvable"


def test_no_density_published_keeps_external_init(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved = resolve_density_for_resume(run_dir)
    assert resolved["branch"] == "external_initialization_required"
    plan = plan_density_reclaim(run_dir)
    assert plan["resources"] == []


def _real_attempt_record(run_dir: Path, scratch_root: Path, *,
                         request_id="req-live", attempt_id="attempt-1") -> dict:
    """A real scratch.allocate attempt with a fake density tree inside."""
    handle = scratch_mod.allocate(
        run_root=run_dir, scratch_root=scratch_root, run_uuid="uuid1",
        backend_role="reference", request_id=request_id, attempt_id=attempt_id,
        archive_dir=run_dir / "calculations" / "eval-000000",
        retention="all")
    save = handle.scratch_dir / "tmp" / "pyraimd2.save"
    save.mkdir(parents=True)
    (save / "charge-density.dat").write_bytes(b"from-scratch")
    (save / "data-file-schema.xml").write_text("<xml/>")
    return handle.load_record()


def test_managed_scratch_entry_publishes_verified_source(tmp_path: Path) -> None:
    """The bounded entry accepts a real managed-scratch attempt record (the
    density lives in scratch; the archive holds only pw.in/pw.out)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "calculations").mkdir()
    scratch_root = tmp_path / "tmp"
    record = _real_attempt_record(run_dir, scratch_root)
    result = publish_density_generation_from_attempt(
        run_dir, record, files=_files(), provenance=_prov("req-live"))
    assert result["generation"] == 1
    copied = (Path(result["directory"]) / "tmp" / "pyraimd2.save"
              / "charge-density.dat")
    assert copied.read_bytes() == b"from-scratch"
    # a tampered record (wrong run root) is refused
    record_bad = dict(record)
    record_bad["run_root"] = str(tmp_path / "other-run")
    with pytest.raises(RestartError, match="not.*this run|belongs to"):
        publish_density_generation_from_attempt(
            run_dir, record_bad, files=_files(), provenance=_prov("req-x"))


def test_unknown_scratch_record_blocks_reclaim(tmp_path: Path) -> None:
    """A real scratch record mutated to an unknown state blocks reclaim with
    a reason; it is never silently skipped."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    record = _real_attempt_record(run_dir, tmp_path / "tmp")
    record_path = next((run_dir / "scratch_records").rglob("*.json"))
    mutated = dict(record, state="mystery-state")
    record_path.write_text(json.dumps(mutated))
    plan = plan_density_reclaim(run_dir)
    assert any("mystery-state" in b or "unknown state" in b
               for b in plan["blocked"])
    assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])


def test_in_flight_attempt_input_referenced(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    record = _real_attempt_record(run_dir, tmp_path / "tmp")
    record_path = next((run_dir / "scratch_records").rglob("*.json"))
    record["density_generation"] = 1
    record_path.write_text(json.dumps(record))
    refs = compute_references(run_dir)
    assert 1 in refs["references"]
    assert any("in-flight" in reason for reason in refs["references"][1])


def test_registry_symlink_escape_refused(tmp_path: Path) -> None:
    """run-a's registry linked at run-b: publish and read both refuse."""
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_a.mkdir()
    run_b.mkdir()
    publish_density_generation(run_b, _source(run_b), files=_files(),
                               provenance=_prov("req-b"))
    (run_a / "restart").mkdir()
    (run_a / "restart" / "density").symlink_to(run_b / "restart" / "density")
    with pytest.raises(RestartError, match="symlink"):
        publish_density_generation(run_a, _source(run_a), files=_files(),
                                   provenance=_prov("req-a"))
    with pytest.raises(RestartError, match="symlink"):
        inspect_density_registry(run_a)


def test_generation_symlink_escape_is_corrupt_not_ok(tmp_path: Path) -> None:
    """A generation directory linked at another run's generation never
    resolves as ok: it is classified corrupt and held."""
    run_dir = tmp_path / "run"
    other = tmp_path / "other"
    other.mkdir()
    publish_density_generation(other, _source(other), files=_files(),
                               provenance={**_prov("req-o"), "run_id": "other"})
    registry = run_dir / "restart" / "density"
    registry.mkdir(parents=True)
    (registry / "g000099").symlink_to(other / "restart" / "density" / "g000001")
    view = inspect_density_registry(run_dir)
    kinds = {r["kind"] for r in view["resources"]}
    assert kinds == {"corrupt_generation"}
    # generations exist but no state file: corrupt registry, never ok
    assert resolve_density_for_resume(run_dir)["branch"] == "unresolvable"


def test_generation_validation_closes_reuse_and_classify(tmp_path: Path) -> None:
    """Missing file list, missing seed file, and directory/manifest mismatch
    are all corrupt — and a retry over a damaged generation refuses to claim
    reuse."""
    run_dir = tmp_path / "run"
    source = _source(run_dir)
    publish_density_generation(run_dir, source, files=_files(),
                               provenance=_prov("req-1"))
    generation = run_dir / "restart" / "density" / "g000001"
    # seed deleted after publish: the retry must NOT report reused
    (generation / "tmp" / "pyraimd2.save" / "charge-density.dat").unlink()
    with pytest.raises(RestartError, match="no longer valid"):
        publish_density_generation(run_dir, source, files=_files(),
                                   provenance=_prov("req-1"))
    assert latest_density_generation(run_dir) is None
    plan = plan_density_reclaim(run_dir)
    assert plan["resources"][0]["kind"] == "corrupt_generation"
    assert plan["resources"][0]["decision"] == "hold"
    # an empty file list is never ok
    manifest_path = generation / "manifest.json"
    (generation / "tmp" / "pyraimd2.save").mkdir(parents=True, exist_ok=True)
    (generation / "tmp" / "pyraimd2.save" / "charge-density.dat").write_bytes(b"density-v1")
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = []
    manifest_path.write_text(json.dumps(manifest))
    assert resolve_density_for_resume(run_dir)["branch"] == "unresolvable"
    plan = plan_density_reclaim(run_dir)
    assert plan["resources"][0]["kind"] == "corrupt_generation"


def test_hdf5_role_registers_alongside_dat(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = _source(run_dir, density_name="charge-density.hdf5",
                     content=b"hdf5-marker")
    result = publish_density_generation(run_dir, source,
                                        files=_files("charge-density.hdf5"),
                                        provenance=_prov("req-h5"))
    roles = {f["role"] for f in result["manifest"]["files"]}
    assert roles == {"charge-density", "metadata"}
    names = {f["path"] for f in result["manifest"]["files"]}
    assert "tmp/pyraimd2.save/charge-density.hdf5" in names


def test_extra_keep_and_space_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    plan = plan_density_reclaim(run_dir, extra_keep=(1,))
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[1]["decision"] == "keep" and by_gen[1]["reasons"] == ["extra_keep"]
    assert by_gen[2]["decision"] == "reclaim_candidate"
    space = plan["space"]
    assert space["extra_kept_seed_bytes"] > 0
    assert space["referenced_seed_bytes"] > 0
    assert space["attached_unreferenced_seed_bytes"] > 0
    assert "bound" in plan["space_bound_note"]


def test_corrupt_generation_manifest_held(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    (run_dir / "restart" / "density" / "g000001" / "manifest.json").write_text("{bad")
    view = inspect_density_registry(run_dir)
    assert view["resources"][0]["kind"] == "corrupt_generation"
    plan = plan_density_reclaim(run_dir)
    assert plan["resources"][0]["decision"] == "hold"


def test_density_reference_field_validity(tmp_path: Path) -> None:
    """density_generation: absent=legacy, null=explicit none, positive int=
    reference, anything else blocks (string "1" is NOT a reference)."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 1, 2: 2})
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-3"))  # g2 superseded
    # a string reference on a retained checkpoint: blocked, never "no field"
    path = run_dir / "checkpoints" / "1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["density_generation"] = "1"
    path.write_text(json.dumps(manifest))
    plan = plan_density_reclaim(run_dir)
    assert any("not a valid positive integer" in b for b in plan["blocked"])
    assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])
    resolved = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert resolved["branch"] == "unresolvable"
    # null is the explicit no-density declaration (distinct from legacy):
    # resume answers the external-init branch without being 'legacy'
    manifest["density_generation"] = None
    path.write_text(json.dumps(manifest))
    resolved = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert resolved["branch"] == "external_initialization_required"
    assert "explicit" in resolved["reason"]
    # checkpoint with files={} is malformed, not legacy
    manifest["density_generation"] = 1
    manifest["files"] = {}
    path.write_text(json.dumps(manifest))
    resolved = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert resolved["branch"] == "unresolvable"
    assert "files block" in resolved["reason"]


def test_scratch_record_invalid_reference_blocks(tmp_path: Path) -> None:
    """A real scratch record with density_generation as the STRING "1"
    blocks reclaim (the reference cannot be determined); with the integer 1
    the reference keeps g1."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    record = _real_attempt_record(run_dir, tmp_path / "tmp")
    record_path = next((run_dir / "scratch_records").rglob("*.json"))
    record["density_generation"] = 1
    record_path.write_text(json.dumps(record))
    refs = compute_references(run_dir)
    assert 1 in refs["references"] and not refs["blocked"]
    record["density_generation"] = "1"
    record_path.write_text(json.dumps(record))
    refs = compute_references(run_dir)
    assert 1 not in refs["references"]
    assert any("not a valid positive integer" in b for b in refs["blocked"])
    plan = plan_density_reclaim(run_dir)
    assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])


def test_managed_source_bound_to_the_verified_attempt(tmp_path: Path) -> None:
    """Record A + provenance B is refused before any write; B with its own
    record then publishes normally."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "calculations").mkdir()
    scratch_root = tmp_path / "tmp"
    record_a = _real_attempt_record(run_dir, scratch_root, request_id="req-A")
    with pytest.raises(RestartError, match="does not match the verified"):
        publish_density_generation_from_attempt(
            run_dir, record_a, files=_files(), provenance=_prov("req-B"))
    assert inspect_density_registry(run_dir)["resources"] == []
    record_b = _real_attempt_record(run_dir, scratch_root, request_id="req-B")
    result = publish_density_generation_from_attempt(
        run_dir, record_b, files=_files(), provenance=_prov("req-B"))
    assert result["generation"] == 1
    assert result["manifest"]["attempt"]["request_id"] == "req-B"


def test_nested_payload_directories_fsynced_before_commit(tmp_path: Path, monkeypatch) -> None:
    """The fsync order covers payload file -> nested payload dirs -> tmp root
    -> registry, before the state commit."""
    run_dir = tmp_path / "run"
    calls: list[str] = []
    real_fsync_dir = rr._fsync_dir

    def recording(path):
        calls.append(str(Path(path)))
        real_fsync_dir(path)

    monkeypatch.setattr(rr, "_fsync_dir", recording)
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    save_dir = str(run_dir / "restart" / "density" / ".tmp-g000001"
                   / "tmp" / "pyraimd2.save")
    tmp_root = str(run_dir / "restart" / "density" / ".tmp-g000001")
    registry_dir = str(run_dir / "restart" / "density")
    assert save_dir in calls and tmp_root in calls and registry_dir in calls
    # leaf payload dir fsynced before the tmp root, both before the state
    # commit (the LAST registry fsync is the state commit's parent fsync)
    def last(name: str) -> int:
        return max(i for i, c in enumerate(calls) if c == name)
    assert calls.index(save_dir) < last(tmp_root) < last(registry_dir)


def test_state_invariant_missing_latest_key_is_corrupt(tmp_path: Path) -> None:
    """latest KEY missing is not a legal null: read/plan/publish/retry all
    refuse, and the state file is never rewritten."""
    run_dir = tmp_path / "run"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    state_path = run_dir / "restart" / "density" / "state.json"
    state_path.write_text('{"attached": [1]}')
    before = state_path.read_bytes()
    view = inspect_density_registry(run_dir)
    assert view["state"] == "corrupt"
    assert plan_density_reclaim(run_dir)["blocked"]
    with pytest.raises(RestartError, match="corrupt"):
        publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                                   files=_files(), provenance=_prov("req-2"))
    with pytest.raises(RestartError, match="corrupt"):
        publish_density_generation(run_dir, _source(run_dir),
                                   files=_files(), provenance=_prov("req-1"))
    assert state_path.read_bytes() == before  # user history never rewritten


def test_state_invariant_latest_must_be_max_attached(tmp_path: Path) -> None:
    """latest=1 with attached=[1,2] (or latest=null with attached=[1,2]) is
    contradictory: corrupt — never a readable state that could list the
    live latest density as a reclaim candidate."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    state_path = run_dir / "restart" / "density" / "state.json"
    for bad in ('{"latest": 1, "attached": [1, 2]}',
                '{"latest": null, "attached": [1, 2]}'):
        state_path.write_text(bad)
        view = inspect_density_registry(run_dir)
        assert view["state"] == "corrupt", bad
        plan = plan_density_reclaim(run_dir)
        assert plan["blocked"]
        assert all(r["decision"] != "reclaim_candidate" for r in plan["resources"])
        assert resolve_density_for_resume(run_dir)["branch"] == "unresolvable"
        with pytest.raises(RestartError, match="corrupt"):
            publish_density_generation(run_dir, _source(run_dir, name="eval-x"),
                                       files=_files(), provenance=_prov("req-x"))


def test_state_invariant_normal_protocol_still_works(tmp_path: Path) -> None:
    """Empty fresh state, monotonic g1->g2 attach and legacy checkpoint
    references are unchanged by the invariant."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert inspect_density_registry(run_dir)["state"] == "fresh"
    publish_density_generation(run_dir, _source(run_dir), files=_files(),
                               provenance=_prov("req-1"))
    publish_density_generation(run_dir, _source(run_dir, name="eval-000001"),
                               files=_files(), provenance=_prov("req-2"))
    view = inspect_density_registry(run_dir)
    assert view["state"] == "ok"
    assert view["latest"] == 2 and view["attach_history"] == [1, 2]
    _write_real_checkpoints(run_dir, {1: None})
    legacy = resolve_density_for_resume(run_dir, checkpoint_generation=1)
    assert legacy["branch"] == "external_initialization_required"
    plan = plan_density_reclaim(run_dir)
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[1]["decision"] == "reclaim_candidate"  # superseded, normal
    assert by_gen[2]["decision"] == "keep"


# ---------------------------------------------------------------------------
# execute_density_reclaim: lock-held, tombstoned, reference-re-reading deletion


def test_execute_reclaims_candidates_with_durable_tombstones(tmp_path: Path) -> None:
    """A superseded, unreferenced generation is actually deleted; the
    tombstone keeps the deletion distinguishable from corruption, the
    attach history is intact, and a repeated execution is a no-op."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    g1_dir = run_dir / "restart" / "density" / "g000001"
    assert g1_dir.is_dir()
    receipt = execute_density_reclaim(run_dir)
    assert receipt["status"] == "ok" and receipt["dry_run"] is False
    assert [r["generation"] for r in receipt["reclaimed"]] == [1]
    assert receipt["reclaimed"][0]["bytes"] > 0
    assert not g1_dir.exists()
    # the tombstone: inspection classifies the deletion as reclaimed —
    # never as corruption or a missing referenced generation
    view = inspect_density_registry(run_dir)
    assert view["reclaimed"] == [1]
    assert view["attach_history"] == [1, 2]
    assert view["latest"] == 2
    kinds = {r["name"]: r["kind"] for r in view["resources"]}
    assert kinds["g000001"] == "reclaimed_generation"
    assert kinds["g000002"] == "generation"
    plan = plan_density_reclaim(run_dir)
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[1]["decision"] == "reclaimed"
    assert by_gen[2]["decision"] == "keep"
    # re-entry: nothing to do, nothing fails, no number is reused
    again = execute_density_reclaim(run_dir)
    assert again["status"] == "ok"
    assert again["reclaimed"] == [] and again["failed"] == []
    published = publish_density_generation(
        run_dir, _source(run_dir, name="eval-000009"), files=_files(),
        provenance=_prov("req-3"))
    assert published["generation"] == 3  # never a reused number
    assert view["attach_history"] == [1, 2]


def test_execute_resumes_an_interrupted_deletion(tmp_path: Path,
                                                 monkeypatch) -> None:
    """A deletion that fails after the tombstone: the receipt is
    incomplete and inspectable, the generation stays on disk marked
    tombstoned, and a later execution resumes and completes it by the
    tombstone alone (no re-validation of a half-deleted tree)."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))

    def boom(registry, name):
        raise OSError("injected deletion failure")

    monkeypatch.setattr(rr, "_delete_generation_tree", boom)
    receipt = execute_density_reclaim(run_dir)
    assert receipt["status"] == "incomplete"
    assert [f["generation"] for f in receipt["failed"]] == [1]
    assert "injected deletion failure" in receipt["failed"][0]["reason"]
    monkeypatch.undo()
    # the tombstone persisted; the half-state is marked, not corrupt
    view = inspect_density_registry(run_dir)
    assert view["reclaimed"] == [1]
    g1 = next(r for r in view["resources"] if r["name"] == "g000001")
    assert g1["kind"] == "generation" and g1.get("reclaim_tombstoned") is True
    plan = plan_density_reclaim(run_dir)
    p1 = next(r for r in plan["resources"] if r["generation"] == 1)
    assert p1["decision"] == "reclaim_candidate"
    assert "did not complete" in p1["reasons"][0]
    # the retry finishes it without re-validating
    receipt = execute_density_reclaim(run_dir)
    assert receipt["status"] == "ok"
    assert [r["generation"] for r in receipt["reclaimed"]] == [1]
    assert not (run_dir / "restart" / "density" / "g000001").exists()


def test_execute_refuses_a_stale_plan(tmp_path: Path) -> None:
    """A caller-supplied dry-run plan is re-verified against the live run
    under the lock; any change refuses the execution before any delete."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    plan = plan_density_reclaim(run_dir)
    assert [r["generation"] for r in plan["resources"]
            if r["decision"] == "reclaim_candidate"] == [1]
    # the run moved on: a new publication changes the reference set
    publish_density_generation(run_dir, _source(run_dir, name="eval-000007"),
                               files=_files(), provenance=_prov("req-3"))
    receipt = execute_density_reclaim(run_dir, plan=plan)
    assert receipt["status"] == "refused"
    assert "stale plan" in receipt["reason"]
    assert (run_dir / "restart" / "density" / "g000001").is_dir()
    assert inspect_density_registry(run_dir)["reclaimed"] == []
    # the same still-current plan executes
    fresh = plan_density_reclaim(run_dir)
    receipt = execute_density_reclaim(run_dir, plan=fresh)
    assert receipt["status"] == "ok"
    assert sorted(r["generation"] for r in receipt["reclaimed"]) == [1, 2]
    # a non-plan is not a deletion credential either
    receipt = execute_density_reclaim(run_dir, plan={"dry_run": False})
    assert receipt["status"] == "refused"


def test_execute_refuses_while_the_run_lock_is_held(tmp_path: Path) -> None:
    """The standalone entry needs the run's single-writer lock; a live run
    refuses rather than deleting under a racing writer."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    with EventLog(run_dir):
        receipt = execute_density_reclaim(run_dir)
        assert receipt["status"] == "refused"
        assert "lock" in receipt["reason"]
    assert (run_dir / "restart" / "density" / "g000001").is_dir()
    # and with the lock free the same call works
    receipt = execute_density_reclaim(run_dir)
    assert receipt["status"] == "ok"


def test_execute_honors_every_protection(tmp_path: Path) -> None:
    """latest, retained checkpoints, extra_keep and the unconsumed producer
    seed all survive; only the genuinely unreferenced generation goes."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3", "req-4"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 2})  # retained checkpoint needs g2
    # g3's producer attempt is still kept: the seed was never consumed
    record = _real_attempt_record(run_dir, tmp_path / "tmp",
                                  request_id="req-3", attempt_id="attempt-1")
    record_path = next((run_dir / "scratch_records").rglob("*.json"))
    record["state"] = "kept"
    record_path.write_text(json.dumps(record))
    receipt = execute_density_reclaim(run_dir, extra_keep=(1,))
    assert receipt["status"] == "ok"
    assert [r["generation"] for r in receipt["reclaimed"]] == []  # nothing!
    # g1: extra_keep; g2: checkpoint; g3: unconsumed producer; g4: latest
    plan = plan_density_reclaim(run_dir, extra_keep=(1,))
    by_gen = {r["generation"]: r for r in plan["resources"]}
    assert by_gen[1]["decision"] == "keep" and "extra_keep" in by_gen[1]["reasons"]
    assert by_gen[2]["decision"] == "keep"
    assert by_gen[3]["decision"] == "keep"
    assert any("unconsumed producer seed" in reason
               for reason in by_gen[3]["reasons"])
    assert by_gen[4]["decision"] == "keep"
    # the consumption proof completes (the producer attempt is released)
    record["state"] = "archived"
    record["archived"] = [{"file": "pw.out", "bytes": 1, "sha256": "0" * 64}]
    record_path.write_text(json.dumps(record))
    receipt = execute_density_reclaim(run_dir, extra_keep=(1,))
    assert receipt["status"] == "ok"
    assert [r["generation"] for r in receipt["reclaimed"]] == [3]
    assert not (run_dir / "restart" / "density" / "g000003").exists()
    for generation in (1, 2, 4):
        assert (run_dir / "restart" / "density"
                / f"g{generation:06d}").is_dir()


def test_execute_never_touches_unowned_or_undetermined(tmp_path: Path) -> None:
    """Publish leftovers, never-attached publications, corrupt generations
    and blocked reference sets are all preserved by the executor."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    registry = run_dir / "restart" / "density"
    leftover = registry / ".tmp-g000009"
    leftover.mkdir()
    (leftover / "partial").write_bytes(b"half")
    # g2 as never-attached: a consistent state listing only 1 and 3
    state_path = registry / "state.json"
    state_path.write_text(json.dumps(
        {"latest": 3, "attached": [1, 3], "reclaimed": []}))
    # g1 corrupt (member tampered): held with its reason
    (registry / "g000001" / "tmp" / "pyraimd2.save"
     / "charge-density.dat").write_bytes(b"tampered")
    receipt = execute_density_reclaim(run_dir)
    assert receipt["status"] == "ok"
    assert receipt["reclaimed"] == []
    assert leftover.is_dir() and (leftover / "partial").read_bytes() == b"half"
    for generation in (1, 2, 3):
        assert (registry / f"g{generation:06d}").is_dir()
    plan = plan_density_reclaim(run_dir)
    decisions = {r["name"]: r["decision"] for r in plan["resources"]}
    assert decisions[".tmp-g000009"] == "hold"
    assert decisions["g000001"] == "hold"   # corrupt, never auto-deleted
    assert decisions["g000002"] == "hold"   # never attached: diagnose first
    assert decisions["g000003"] == "keep"   # latest


def test_execute_resumes_after_a_partial_payload_deletion(tmp_path: Path,
                                                          monkeypatch) -> None:
    """The real interruption shape: the tombstone is durable, ONE payload
    file is already deleted, and the delete then fails.  The partial tree
    keeps its verifiable generation identity (never mistaken for plain
    corruption), a fresh execute completes it without re-validating the
    missing payload, a third run is a no-op, and the protected resources
    (retained checkpoint's reference, latest) are untouched throughout."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2", "req-3"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    _write_real_checkpoints(run_dir, {1: 2})  # g2 referenced by a checkpoint
    registry = run_dir / "restart" / "density"

    def partial_delete(registry_, name):
        (registry_ / name / "tmp/pyraimd2.save/charge-density.dat").unlink()
        raise OSError("interrupted after the first payload unlink")

    monkeypatch.setattr(rr, "_delete_generation_tree", partial_delete)
    first = execute_density_reclaim(run_dir)
    monkeypatch.undo()
    assert first["status"] == "incomplete"
    assert [f["generation"] for f in first["failed"]] == [1]
    assert (registry / "g000001").is_dir()  # partial tree remains
    # the partial tree keeps its verifiable identity
    view = inspect_density_registry(run_dir)
    g1 = next(r for r in view["resources"] if r["name"] == "g000001")
    assert g1["kind"] == "corrupt_generation"
    assert g1["generation"] == 1
    assert g1["reclaim_tombstoned"] is True
    plan = plan_density_reclaim(run_dir)
    p1 = next(r for r in plan["resources"] if r.get("generation") == 1)
    assert p1["decision"] == "reclaim_candidate"
    assert "did not complete" in p1["reasons"][0]
    # the fresh execution finishes the interruption (no KeyError, no
    # re-validation of the deleted payload)
    second = execute_density_reclaim(run_dir)
    assert second["status"] == "ok"
    assert [r["generation"] for r in second["reclaimed"]] == [1]
    assert not (registry / "g000001").exists()
    # a third run is a clean no-op
    third = execute_density_reclaim(run_dir)
    assert third["status"] == "ok"
    assert third["reclaimed"] == [] and third["failed"] == []
    # the protected resources survived every pass
    assert (registry / "g000002").is_dir()  # retained checkpoint reference
    assert (registry / "g000003").is_dir()  # latest
    view = inspect_density_registry(run_dir)
    assert view["reclaimed"] == [1]
    assert view["latest"] == 3 and view["attach_history"] == [1, 2, 3]


def test_interrupted_deletion_with_new_reference_stays_suspended(
        tmp_path: Path, monkeypatch) -> None:
    """A partial tree whose generation becomes referenced again is NOT
    resumed: the fresh reference set wins over the stale tombstone."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    registry = run_dir / "restart" / "density"

    def partial_delete(registry_, name):
        (registry_ / name / "tmp/pyraimd2.save/charge-density.dat").unlink()
        raise OSError("interrupted after the first payload unlink")

    monkeypatch.setattr(rr, "_delete_generation_tree", partial_delete)
    first = execute_density_reclaim(run_dir)
    monkeypatch.undo()
    assert first["status"] == "incomplete"
    # g1 is referenced again before the retry (a retained checkpoint)
    _write_real_checkpoints(run_dir, {1: 1})
    plan = plan_density_reclaim(run_dir)
    p1 = next(r for r in plan["resources"] if r.get("generation") == 1)
    assert p1["decision"] == "keep"
    assert any("referenced" in reason for reason in p1["reasons"])
    second = execute_density_reclaim(run_dir)
    assert second["status"] == "ok"
    assert second["reclaimed"] == []
    assert (registry / "g000001").is_dir()  # suspended, not deleted


def test_interrupted_deletion_with_confused_identity_never_resumes(
        tmp_path: Path, monkeypatch) -> None:
    """A partial tree whose manifest names ANOTHER generation is a
    confused/foreign leftover: held as corrupt, never resumed by the
    tombstone of the directory's number."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    registry = run_dir / "restart" / "density"

    def partial_delete(registry_, name):
        (registry_ / name / "tmp/pyraimd2.save/charge-density.dat").unlink()
        raise OSError("interrupted after the first payload unlink")

    monkeypatch.setattr(rr, "_delete_generation_tree", partial_delete)
    first = execute_density_reclaim(run_dir)
    monkeypatch.undo()
    assert first["status"] == "incomplete"
    # confuse the identity: the manifest now claims another generation
    manifest_path = registry / "g000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = 99
    manifest_path.write_text(json.dumps(manifest))
    view = inspect_density_registry(run_dir)
    g1 = next(r for r in view["resources"] if r["name"] == "g000001")
    assert g1["kind"] == "corrupt_generation"
    assert "generation" not in g1  # identity not verifiable
    assert "reclaim_tombstoned" not in g1
    second = execute_density_reclaim(run_dir)
    assert second["status"] == "ok"
    assert second["reclaimed"] == []
    assert (registry / "g000001").is_dir()  # held, never auto-deleted


def test_interrupted_deletion_with_foreign_same_number_tree_never_resumes(
        tmp_path: Path, monkeypatch) -> None:
    """A different run's same-number generation copied into an interrupted
    tombstone slot: the readable manifest explicitly contradicts ownership
    (its run_root names the other run).  Inspection reports the
    contradiction and grants no resumable identity; execution keeps the
    foreign tree — the durable tombstone of THIS run never deletes it."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    registry = run_dir / "restart" / "density"

    def partial_delete(registry_, name):
        (registry_ / name / "tmp/pyraimd2.save/charge-density.dat").unlink()
        raise OSError("interrupted after the first payload unlink")

    monkeypatch.setattr(rr, "_delete_generation_tree", partial_delete)
    first = execute_density_reclaim(run_dir)
    monkeypatch.undo()
    assert first["status"] == "incomplete"
    # a foreign run's g1 (different content, readable manifest naming ITS
    # run root) replaces the partial tree in the slot
    foreign = tmp_path / "foreign-run"
    publish_density_generation(
        foreign, _source(foreign, content=b"foreign-owned-density"),
        files=_files(), provenance=_prov("foreign-req"))
    (registry / "g000001").rename(tmp_path / "saved-original-partial-g1")
    shutil.copytree(foreign / "restart" / "density" / "g000001",
                    registry / "g000001")
    (registry / "g000001" / "foreign-sentinel.txt").write_text("keep me")
    view = inspect_density_registry(run_dir)
    g1 = next(r for r in view["resources"] if r["name"] == "g000001")
    assert g1["kind"] == "corrupt_generation"
    assert "run_root" in g1["reason"] and "does not match" in g1["reason"]
    assert "generation" not in g1  # no verifiable identity: not resumable
    assert "reclaim_tombstoned" not in g1
    plan = plan_density_reclaim(run_dir)
    assert next(r for r in plan["resources"]
                if r["name"] == "g000001")["decision"] == "hold"
    second = execute_density_reclaim(run_dir)
    assert second["status"] == "ok" and second["reclaimed"] == []
    assert (registry / "g000001" / "foreign-sentinel.txt").is_file()
    # and the run's own interrupted tree is untouched by all of this
    assert (tmp_path / "saved-original-partial-g1").is_dir()


def test_interrupted_deletion_without_manifest_never_resumes(
        tmp_path: Path, monkeypatch) -> None:
    """A partial tree whose manifest is also gone has no verifiable
    identity at all: fail-closed — held as corrupt, never resumed by the
    tombstone alone (ownership cannot be positively established)."""
    run_dir = tmp_path / "run"
    for req in ("req-1", "req-2"):
        publish_density_generation(run_dir, _source(run_dir), files=_files(),
                                   provenance=_prov(req))
    registry = run_dir / "restart" / "density"

    def partial_delete(registry_, name):
        (registry_ / name / "tmp/pyraimd2.save/charge-density.dat").unlink()
        (registry_ / name / "manifest.json").unlink()
        raise OSError("interrupted after payload and manifest unlink")

    monkeypatch.setattr(rr, "_delete_generation_tree", partial_delete)
    first = execute_density_reclaim(run_dir)
    monkeypatch.undo()
    assert first["status"] == "incomplete"
    view = inspect_density_registry(run_dir)
    g1 = next(r for r in view["resources"] if r["name"] == "g000001")
    assert g1["kind"] == "corrupt_generation"
    assert "generation" not in g1 and "reclaim_tombstoned" not in g1
    second = execute_density_reclaim(run_dir)
    assert second["status"] == "ok" and second["reclaimed"] == []
    assert (registry / "g000001").is_dir()
