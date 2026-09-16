"""Density publication bridge (``density_registry_run_dir``): opt-in
run-owned, save-only publication of a successful attempt's candidate seed
pack through both real compute entries — fake pw.x only, no real QE.

The synthetic ``data-file-schema.xml`` written by the fake scripts is a
placeholder for the artifact rule, NOT a sufficiency proof of the seed
pack for real QE restarts.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from ase import Atoms

from pyraimd2.engines import density_publish
from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import QeConfig, QeEngine, QeEngineError
from pyraimd2.runtime import scratch as scratch_mod
from pyraimd2.runtime.restart import (
    RestartError,
    inspect_density_registry,
    latest_density_generation,
)

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


def _success_script(tmp_path: Path, *, density_name="charge-density.dat",
                    with_density=True, with_xml=True, with_paw=False,
                    count_file: Path | None = None) -> tuple[str, ...]:
    body = "#!/bin/bash\n"
    if count_file is not None:
        body += f"echo x >> {count_file}\n"
    if with_density or with_xml or with_paw:
        body += "mkdir -p tmp/pyraimd2.save\n"
    if with_density:
        body += (f"echo fake-density > tmp/pyraimd2.save/{density_name}\n")
    if with_xml:
        # synthetic placeholder schema document — see the module docstring
        body += "echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml\n"
    if with_paw:
        # the QE 7.5 PAW becsum a startpot='file' restart reads
        body += "echo paw-becsum > tmp/pyraimd2.save/paw.txt\n"
    body += f"cat {FIXTURE.resolve()}\n"
    return _fake_pwx(tmp_path, body)


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The real top-level layout: owner run dir + its calculations subtree."""
    owner = tmp_path / "run"
    run_root = owner / "calculations"
    run_root.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    return owner, run_root, scratch


def _engine(kind: str, tmp_path: Path, *, script, owner: Path | None,
            run_root: Path, scratch: Path | None, **config_kwargs):
    config = QeConfig(pseudo_dir="/pseudo", pw_cmd=script,
                      scratch_root=(None if scratch is None else str(scratch)),
                      **config_kwargs)
    if kind == "qe":
        return QeEngine(config, run_root=run_root,
                        density_registry_run_dir=owner)
    return AseQeEngine(config, run_root=run_root,
                       density_registry_run_dir=owner)


def _records(run_root: Path) -> list[dict]:
    root = run_root / scratch_mod.SCRATCH_RECORD_DIR
    return [json.loads(p.read_text())
            for p in sorted(root.rglob("*.json"))] if root.is_dir() else []


def _registry(owner: Path) -> Path:
    return owner / "restart" / "density"


def _payload(owner: Path, generation: int, density_name: str) -> bytes:
    gen_dir = _registry(owner) / f"g{generation:06d}"
    return (gen_dir / "tmp" / "pyraimd2.save" / density_name).read_bytes()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_bridge_off_by_default(kind, tmp_path):
    run_root = tmp_path / "runs"
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=None, run_root=run_root,
                     scratch=tmp_path / "scratch")
    engine.compute(_si(), label="si")
    record = engine.last_attempt_records[-1]
    assert "density_publish" not in record
    assert not (run_root / "restart").exists()
    # the scratch records stay bound to the engine run root, as before
    assert _records(run_root)[0]["run_root"] == str(run_root.resolve())


@pytest.mark.parametrize("kind,density_name",
                         [("qe", "charge-density.dat"),
                          ("ase", "charge-density.hdf5")])
def test_publish_and_independent_readback(kind, density_name, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path,
                                            density_name=density_name),
                     owner=owner, run_root=run_root, scratch=scratch)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    record = engine.last_attempt_records[-1]
    publish = record["density_publish"]
    assert publish["status"] == "published" and publish["reused"] is False
    assert publish["generation"] == 1
    assert Path(publish["directory"]).parent == _registry(owner).resolve()

    # the managed attempt record is bound to the explicit run owner, not
    # the engine's calculations subtree
    scratch_records = _records(owner)
    assert len(scratch_records) == 1
    assert scratch_records[0]["run_root"] == str(owner.resolve())
    assert scratch_records[0]["state"] == "kept"

    latest = latest_density_generation(owner)
    assert latest["generation"] == 1
    manifest = latest["manifest"]
    assert manifest["reference_fingerprint"] == engine.fingerprint
    assert manifest["nat"] == 2 and manifest["species"] == ["Si"]
    assert manifest["attempt"] == {
        "request_id": scratch_records[0]["request_id"],
        "attempt_id": "attempt-1"}
    assert manifest["run_root"] == str(owner.resolve())
    view = inspect_density_registry(owner)
    assert view["state"] == "ok" and view["attach_history"] == [1]

    # the persistent copy holds the payload; the source is still in place
    source_save = (Path(record["directory"]) / "tmp" / "pyraimd2.save")
    assert (source_save / density_name).is_file()  # save-only: source kept
    assert _payload(owner, 1, density_name) == \
        (source_save / density_name).read_bytes()

    # the registry reads independently of the source scratch tree
    away = tmp_path / "scratch-away"
    scratch.rename(away)
    try:
        still = latest_density_generation(owner)
        assert still["generation"] == 1
        assert _payload(owner, 1, density_name) == b"fake-density\n"
    finally:
        away.rename(scratch)


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_rebuilt_engine_and_directory_depth_never_collide(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    for index in range(2):
        engine = _engine(kind, tmp_path / f"s{index}",
                         script=_success_script(tmp_path / f"s{index}"),
                         owner=owner, run_root=run_root, scratch=scratch)
        engine.compute(_si(), label="si")
    # a rebuilt engine continues the on-disk numbering instead of
    # re-issuing a process-counter request id
    rebuilt = _engine(kind, tmp_path / "s2", script=_success_script(tmp_path / "s2"),
                      owner=owner, run_root=run_root, scratch=scratch)
    rebuilt.compute(_si(), label="si")
    # a second adapter at a different directory depth under the same owner
    deeper = _engine(kind, tmp_path / "s3", script=_success_script(tmp_path / "s3"),
                     owner=owner, run_root=run_root / "deeper",
                     scratch=scratch)
    deeper.compute(_si(), label="si")

    view = inspect_density_registry(owner)
    assert view["state"] == "ok"
    assert view["latest"] == 4 and view["attach_history"] == [1, 2, 3, 4]
    identities = [r["manifest"]["attempt"]["request_id"]
                  for r in view["resources"]]
    assert len(set(identities)) == 4
    # directory-derived persistent identities: a fixed short purpose prefix
    # plus the path digest; the allocated compute directory itself is named
    # in the provenance source record, never spliced into the id
    assert all(identity.startswith("dir#") for identity in identities)
    assert all(len(identity) <= 32 for identity in identities)
    sources = [r["manifest"]["source"]["archive_dir"]
               for r in view["resources"]]
    assert any("deeper" in archive_dir for archive_dir in sources)
    for generation in (1, 2, 3, 4):
        assert _payload(owner, generation, "charge-density.dat") == \
            b"fake-density\n"


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_failed_scf_publishes_nothing(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    script = _fake_pwx(tmp_path, "#!/bin/bash\necho total garbage; exit 1\n")
    engine = _engine(kind, tmp_path, script=script, owner=owner,
                     run_root=run_root, scratch=scratch)
    with pytest.raises(QeEngineError):
        engine.compute(_si(), label="si")
    assert engine.last_attempt_records[-1]["status"] == "failed"
    assert "density_publish" not in engine.last_attempt_records[-1]
    assert not (owner / "restart").exists()


@pytest.mark.parametrize("scenario", ["no_file", "residue_under_none"])
def test_no_new_density_keeps_previous_generation(scenario, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine("qe", tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch)
    first.compute(_si(), label="si")
    assert first.last_attempt_records[-1]["density_publish"]["generation"] == 1

    if scenario == "no_file":
        # a successful run that simply left no density behind
        script = _success_script(tmp_path / "b", with_density=False,
                                 with_xml=False)
        config_kwargs: dict = {}
    else:
        # disk_io=none: any density file present is the staged input
        # residue, never this attempt's product
        script = _success_script(tmp_path / "b")
        config_kwargs = {"disk_io": "none"}
    second = _engine("qe", tmp_path / "b", script=script, owner=owner,
                     run_root=run_root, scratch=scratch, **config_kwargs)
    result = second.compute(_si(), label="si")
    assert result.forces is not None
    record = second.last_attempt_records[-1]
    assert record["density_available"] is False
    assert "density_publish" not in record
    latest = latest_density_generation(owner)
    assert latest["generation"] == 1  # the previous generation stands


def test_missing_seed_metadata_is_not_published(tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine("qe", tmp_path,
                     script=_success_script(tmp_path, with_xml=False),
                     owner=owner, run_root=run_root, scratch=scratch)
    engine.compute(_si(), label="si")
    publish = engine.last_attempt_records[-1]["density_publish"]
    assert publish["status"] == "not_published"
    assert "data-file-schema.xml" in publish["reason"]
    assert not _registry(owner).exists()  # source and state untouched


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_publish_failure_records_and_never_reruns(kind, tmp_path, monkeypatch):
    owner, run_root, scratch = _layout(tmp_path)
    count_file = tmp_path / "launches"
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path, count_file=count_file),
                     owner=owner, run_root=run_root, scratch=scratch)

    def boom(*args, **kwargs):
        raise RestartError("injected registry refusal")

    monkeypatch.setattr(
        density_publish.restart_mod, "publish_density_generation_from_attempt",
        boom)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the delivered label stands
    record = engine.last_attempt_records[-1]
    assert record["status"] == "success"
    publish = record["density_publish"]
    assert publish["status"] == "failed"
    assert "injected registry refusal" in publish["reason"]
    # exactly one launch — no SCF rerun on a publication failure
    assert len(engine.last_attempt_records) == 1
    assert len(count_file.read_text().splitlines()) == 1
    # the source scratch and the archived light results survive
    assert Path(record["directory"]).is_dir()
    assert not _registry(owner).exists()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_owner_and_mode_refusals(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="does not own"):
        _engine(kind, tmp_path, script=("bash", "-c", "true"), owner=elsewhere,
                run_root=run_root, scratch=scratch)
    with pytest.raises(ValueError, match="retention"):
        _engine(kind, tmp_path, script=("bash", "-c", "true"), owner=owner,
                run_root=run_root, scratch=scratch, retention="results")
    with pytest.raises(ValueError, match="overlap"):
        _engine(kind, tmp_path, script=("bash", "-c", "true"), owner=owner,
                run_root=run_root, scratch=owner / "restart")


def test_unmanaged_source_publishes_via_run_owned_entry(tmp_path):
    owner, run_root, _ = _layout(tmp_path)
    engine = _engine("qe", tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=None)
    engine.compute(_si(), label="si")
    record = engine.last_attempt_records[-1]
    publish = record["density_publish"]
    assert publish["status"] == "published" and publish["generation"] == 1
    assert _records(owner) == []  # no managed scratch involved
    # the persistent copy reads independently of the archived source
    source = Path(record["directory"])
    assert (source / "tmp" / "pyraimd2.save" / "charge-density.dat").is_file()
    away = owner / "calculations-away"
    run_root.rename(away)
    try:
        assert latest_density_generation(owner)["generation"] == 1
        assert _payload(owner, 1, "charge-density.dat") == b"fake-density\n"
    finally:
        away.rename(run_root)


def test_bridge_never_publishes_under_another_attempt_or_run(tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine("qe", tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch)
    engine.compute(_si(), label="si")
    record = engine.last_attempt_records[-1]
    scratch_record = _records(owner)[0]
    handle = engine._last_scratch_handle
    work_dir = Path(record["directory"])
    save_tree = work_dir / "tmp" / "pyraimd2.save"
    density_file = save_tree / "charge-density.dat"

    common = {"source_root": work_dir, "density_file": density_file,
              "reference_fingerprint": engine.fingerprint, "nat": 2,
              "species": ["Si"], "disk_io": None,
              "source_desc": {"kind": "qe-attempt"}}
    # a foreign attempt identity is refused against the verified record
    mismatched = density_publish.publish_attempt_density(
        owner_dir=owner, scratch_handle=handle, request_id="tampered",
        attempt_id="attempt-1", **common)
    assert mismatched["status"] == "failed"
    assert "does not match" in mismatched["reason"]
    # another run's registry refuses this run's attempt record
    other = tmp_path / "other-run"
    other.mkdir()
    foreign = density_publish.publish_attempt_density(
        owner_dir=other, scratch_handle=handle,
        request_id=scratch_record["request_id"], attempt_id="attempt-1",
        **common)
    assert foreign["status"] == "failed"
    assert "belongs to run root" in foreign["reason"]
    # neither refusal touched the valid first generation
    latest = latest_density_generation(owner)
    assert latest["generation"] == 1
    assert not (other / "restart").exists()


def test_directory_request_id_encoding_unambiguous(tmp_path):
    owner = (tmp_path / "run").resolve()
    flat = owner / "calculations" / "a__b" / "si-000000"
    nested = owner / "calculations" / "a" / "b" / "si-000000"
    id_flat = density_publish.directory_request_id(owner, flat)
    id_nested = density_publish.directory_request_id(owner, nested)
    # same readable prefix, different identity digests
    assert id_flat.split("#")[0] == id_nested.split("#")[0]
    assert id_flat != id_nested
    # the same actual directory keeps the same id (retry idempotency)
    assert density_publish.directory_request_id(owner, flat) == id_flat


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_double_underscore_paths_never_share_identity(kind, tmp_path):
    """The N036 sidecar repro: calculations/a__b and calculations/a/b must
    not collapse into one request identity through the real compute entry."""
    owner = tmp_path / "run"
    owner.mkdir()
    scratch = tmp_path / "scratch"
    generations = []
    for name, run_root in (("flat", owner / "calculations" / "a__b"),
                           ("nested", owner / "calculations" / "a" / "b")):
        engine = _engine(kind, tmp_path / name,
                         script=_success_script(tmp_path / name),
                         owner=owner, run_root=run_root, scratch=scratch)
        engine.compute(_si(), label="si")
        publish = engine.last_attempt_records[-1]["density_publish"]
        generations.append((publish["generation"], publish["reused"]))
    # two real computations, two generations — the second is never mistaken
    # for an idempotent retry of the first, even with identical payload bytes
    assert generations == [(1, False), (2, False)]
    view = inspect_density_registry(owner)
    assert view["latest"] == 2 and view["attach_history"] == [1, 2]
    ids = [r["manifest"]["attempt"]["request_id"] for r in view["resources"]]
    assert len(set(ids)) == 2
    assert ids[0].split("#")[0] == ids[1].split("#")[0]  # same readable prefix


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_transient_record_read_failure_recorded_label_delivered(
        kind, tmp_path, monkeypatch):
    owner, run_root, scratch = _layout(tmp_path)
    count_file = tmp_path / "launches"
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path, count_file=count_file),
                     owner=owner, run_root=run_root, scratch=scratch)
    real_load = scratch_mod.AttemptScratch.load_record
    calls = {"fired": False}

    def flaky(self):
        # the failure starts at the publish boundary: reads during the
        # archive step (record still "allocated") are unaffected
        state = json.loads(self.record_path.read_text()).get("state")
        if state == "archived" and not calls["fired"]:
            calls["fired"] = True
            raise OSError("injected transient read failure")
        return real_load(self)

    monkeypatch.setattr(scratch_mod.AttemptScratch, "load_record", flaky)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the verified label still delivers
    record = engine.last_attempt_records[-1]
    publish = record["density_publish"]
    assert publish["status"] == "failed"
    assert "transient read failure" in publish["reason"]
    # exactly one launch — no SCF rerun; source and light results survive
    assert len(count_file.read_text().splitlines()) == 1
    assert Path(record["directory"]).is_dir()
    assert not _registry(owner).exists()
    # the failure was transient: the follow-up keep marking still persisted
    assert record["scratch_cleanup"]["status"] == "kept"


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_persistent_record_read_failure_never_escapes(
        kind, tmp_path, monkeypatch):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch)

    real_load = scratch_mod.AttemptScratch.load_record

    def dead(self):
        # persistent failure from the publish boundary onwards: the archive
        # step's own reads (state "allocated") still succeed
        state = json.loads(self.record_path.read_text()).get("state")
        if state == "archived":
            raise OSError("injected persistent read failure")
        return real_load(self)

    monkeypatch.setattr(scratch_mod.AttemptScratch, "load_record", dead)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    record = engine.last_attempt_records[-1]
    assert record["density_publish"]["status"] == "failed"
    # mark_kept hits the same unreadable record and reports it in the
    # receipt instead of raising the same error again
    assert record["scratch_cleanup"]["status"] == "keep_persist_failed"
    assert not _registry(owner).exists()


def test_directory_request_id_short_and_bounded(tmp_path):
    owner = (tmp_path / "run").resolve()
    deep = owner / "calculations" / ("x" * 170) / "si-000000"
    request_id = density_publish.directory_request_id(owner, deep)
    # fixed purpose prefix plus the identity digest only: the path itself
    # is never spliced into a file-name component (N038 length regression)
    assert request_id.startswith("dir#")
    assert "calculations" not in request_id and "si-000000" not in request_id
    # bounded so the scratch record file and its atomic temp suffix stay
    # far below the filesystem component limit
    assert len(request_id) <= 32
    assert density_publish.directory_request_id(owner, deep) == request_id


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_long_directory_component_stays_publishable(kind, tmp_path):
    """The N038 sidecar repro: a 170-character directory component ran fine
    on the pre-registry baseline, but the readable-prefix request id pushed
    the scratch record's atomic temp file name past the filesystem limit
    and failed the launch before any SCF."""
    owner, run_root, scratch = _layout(tmp_path)
    count_file = tmp_path / "launches"
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path, count_file=count_file),
                     owner=owner, run_root=run_root / ("x" * 170),
                     scratch=scratch)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None
    record = engine.last_attempt_records[-1]
    publish = record["density_publish"]
    assert publish["status"] == "published" and publish["generation"] == 1
    assert len(count_file.read_text().splitlines()) == 1
    # the durable record names stay bounded; the full path lives in the
    # record payload, not in the file name
    records = _records(owner)
    assert len(records) == 1
    names = [p.name for p in
             (owner / scratch_mod.SCRATCH_RECORD_DIR).rglob("*.json")]
    assert names and max(len(name) for name in names) <= 96
    view = inspect_density_registry(owner)
    assert view["latest"] == 1
    source = view["resources"][0]["manifest"]["source"]
    assert "x" * 170 in source["archive_dir"]


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_metadata_query_failure_inside_publish_boundary(
        kind, tmp_path, monkeypatch):
    """The N038 sidecar repro: an I/O error from the seed-metadata is_file
    query after a successful, archived SCF is a publication failure record
    — the delivered result, the source and the previous registry state are
    never sacrificed to a post-processing stat error."""
    owner, run_root, scratch = _layout(tmp_path)
    count_file = tmp_path / "launches"
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path, count_file=count_file),
                     owner=owner, run_root=run_root, scratch=scratch)
    real_is_file = Path.is_file
    scratch_resolved = scratch.resolve()
    hits = []

    def broken(path):
        if (path.name == density_publish.SEED_METADATA_NAME
                and path.is_relative_to(scratch_resolved)):
            hits.append(str(path))
            raise OSError("injected metadata stat failure")
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", broken)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the verified label still delivers
    assert hits  # the injection really fired at the publication query
    record = engine.last_attempt_records[-1]
    publish = record["density_publish"]
    assert publish["status"] == "failed"
    assert "injected metadata stat failure" in publish["reason"]
    # exactly one launch — no SCF rerun; source and light results survive
    assert len(count_file.read_text().splitlines()) == 1
    assert Path(record["directory"]).is_dir()
    assert not _registry(owner).exists()
    # the follow-up keep marking still persisted
    assert record["scratch_cleanup"]["status"] == "kept"


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_paw_becsum_included_in_the_seed_pack(kind, tmp_path):
    """QE 7.5 PAW builds write paw.txt (the becsum) next to the density and
    read it back on a startpot='file' restart; the published seed pack must
    carry it.  A run without paw.txt publishes exactly as before."""
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path / "paw",
                     script=_success_script(tmp_path / "paw", with_paw=True),
                     owner=owner, run_root=run_root, scratch=scratch)
    engine.compute(_si(), label="si")
    publish = engine.last_attempt_records[-1]["density_publish"]
    assert publish["status"] == "published"
    gen_dir = _registry(owner) / "g000001"
    assert (gen_dir / "tmp" / "pyraimd2.save" / "paw.txt").read_text() \
        == "paw-becsum\n"
    view = inspect_density_registry(owner)
    roles = sorted(entry["role"]
                   for entry in view["resources"][0]["manifest"]["files"])
    assert roles == ["charge-density", "metadata", "paw-becsum"]
    # the load bridge exposes the save tree holding the becsum file
    payload, reason, generation = density_publish.load_published_density(
        owner, generation=None,
        reference_fingerprint=engine.fingerprint, nat=2, species=["Si"])
    assert reason == "" and payload is not None and generation == 1
    assert (payload[1] / "paw.txt").is_file()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_no_paw_file_publishes_without_the_role(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path,
                     script=_success_script(tmp_path, with_paw=False),
                     owner=owner, run_root=run_root, scratch=scratch)
    engine.compute(_si(), label="si")
    publish = engine.last_attempt_records[-1]["density_publish"]
    assert publish["status"] == "published"
    view = inspect_density_registry(owner)
    roles = sorted(entry["role"]
                   for entry in view["resources"][0]["manifest"]["files"])
    assert roles == ["charge-density", "metadata"]
