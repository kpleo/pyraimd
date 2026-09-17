"""The run-owned persistent density chain: [density] persist + [scratch]
on plain serial reference MD — configuration scope, owner injection, the
verified generation load bridge, registry-aware source selection, and the
checkpoint/resume/reclaim chain through the real workflow entries.

Fake pw.x only, no real QE.  Assertions read the persistent records
(registry view, scratch records, event log, checkpoint manifests), never
the engine's memory.  The synthetic ``data-file-schema.xml`` written by
the fake scripts is a placeholder for the artifact rule, NOT a
sufficiency proof of the seed pack for real QE restarts.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pyraimd2 import __version__
from pyraimd2.config import ConfigError, load_config, load_resolved_config
from pyraimd2.engines import density_publish
from pyraimd2.engines.ase_qe import AseQeEngine
from pyraimd2.engines.qe_engine import QeConfig, QeEngine, QeEngineError
from pyraimd2.runtime import scratch as scratch_mod
from pyraimd2.runtime.checkpoint import CheckpointManager
from pyraimd2.runtime.events import (
    ATTEMPT_LEDGER_PHYSICAL_V1,
    EVALUATION_COMMITTED,
    EVENT_SCHEMA_VERSION,
    RESUMED,
    STEP_COMPLETED,
    EventLog,
)
from pyraimd2.runtime.restart import (
    RestartError,
    inspect_density_registry,
    latest_density_generation,
    plan_density_reclaim,
)
from pyraimd2.store import Store
from pyraimd2.store.store import STORE_SCHEMA_VERSION
from pyraimd2.workflows import resume_workflow, run_workflow
from pyraimd2.workflows.setup import WorkflowError, _factory_run_kwargs

FIXTURE = Path(__file__).parents[1] / "data" / "qe_si_scf.out"

STRUCTURE = (
    "2\n"
    'Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" '
    'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
    "Si 0.0 0.0 0.0\nSi 1.36 1.36 1.36\n")


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


def _success_script(tmp_path: Path, *, with_density=True, with_xml=True,
                    count_file: Path | None = None) -> tuple[str, ...]:
    body = "#!/bin/bash\n"
    if count_file is not None:
        body += f"echo x >> {count_file}\n"
    # a genuine warm start: the staged input density exists BEFORE this run
    # writes its own products — report the read exactly like QE 7.5 does
    # (an atomic start has no staged file and prints no read marker)
    body += ("seed=tmp/pyraimd2.save\n"
             "if test -f \"$seed/charge-density.dat\" || "
             "test -f \"$seed/charge-density.hdf5\"; then\n"
             "  echo '     The initial density is read from file :'\n"
             "  echo '     ./tmp/pyraimd2.save/charge-density'\n"
             "fi\n")
    if with_density or with_xml:
        body += "mkdir -p tmp/pyraimd2.save\n"
    if with_density:
        body += "echo fake-density > tmp/pyraimd2.save/charge-density.dat\n"
    if with_xml:
        # synthetic placeholder schema document — see the module docstring
        body += "echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml\n"
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


# ---------------------------------------------------------------------------
# [density] configuration scope


def _write_md_toml(root: Path, *, kind="md", mode="reference",
                   reference='[reference]\nbackend = "qe"\npseudo_dir = "/pseudo"\n',
                   surrogate=None, policy="", extra="") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(STRUCTURE)
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \"run\"\n"
        "seed = 42\n[task]\nkind = \"" + kind + "\"\nmode = \"" + mode + "\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        + (reference or "") + (surrogate or "") + policy +
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\nsteps = 1\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n" + extra)
    return path


def test_density_section_parsed_and_round_tripped(tmp_path):
    config = load_config(_write_md_toml(
        tmp_path / "on", extra="[density]\npersist = true\n"))
    assert config.density is not None and config.density.persist is True
    document = config.resolved_dict()
    assert document["density"] == {"persist": True}
    run_dir = tmp_path / "on" / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps(document, indent=2) + "\n")
    rebuilt = load_resolved_config(run_dir)
    assert rebuilt.density is not None and rebuilt.density.persist is True


def test_density_section_defaults_and_legacy_identity(tmp_path):
    # a configured but disabled section is inert
    off = load_config(_write_md_toml(
        tmp_path / "off", extra="[density]\npersist = false\n"))
    assert off.density is not None and off.density.persist is False
    # no [density] section: the resolved record is byte-identical to the
    # pre-feature format (no density block at all)
    plain = load_config(_write_md_toml(tmp_path / "plain"))
    assert plain.density is None
    assert "density" not in plain.resolved_dict()


def test_density_section_field_validation(tmp_path):
    bad = _write_md_toml(tmp_path / "a", extra="[density]\npersist = 1\n")
    with pytest.raises(ConfigError, match="density.persist must be a boolean"):
        load_config(bad)
    bad2 = _write_md_toml(tmp_path / "b",
                          extra="[density]\npersist = true\nextra = 1\n")
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(bad2)


@pytest.mark.parametrize("kind,mode", [("singlepoint", "reference"),
                                       ("relax", "reference"),
                                       ("md", "surrogate")])
def test_density_persist_rejects_unsupported_tasks(tmp_path, kind, mode):
    reference = '[reference]\nbackend = "qe"\npseudo_dir = "/pseudo"\n'
    surrogate = None
    if mode == "surrogate":
        # a valid surrogate-mode config, so the density scope check is
        # what fires (not the task-compatibility rule)
        reference = None
        surrogate = ('[surrogate]\nbackend = "harmonic-surrogate"\n'
                     'k = 1.0\nr0 = 0.9\nbias = 0.05\n')
    path = _write_md_toml(tmp_path / f"{kind}-{mode}", kind=kind, mode=mode,
                          reference=reference, surrogate=surrogate,
                          extra="[density]\npersist = true\n")
    with pytest.raises(ConfigError, match="plain serial reference MD"):
        load_config(path)


def test_density_persist_rejects_adaptive_and_non_qe(tmp_path):
    # a complete adaptive config reaches the density scope check (the task
    # compatibility rule fires first otherwise)
    adaptive = _write_md_toml(
        tmp_path / "adaptive", mode="adaptive",
        reference='[reference]\nbackend = "harmonic-reference"\n'
                  'k = 1.0\nr0 = 0.9\n',
        surrogate='[surrogate]\nbackend = "harmonic-surrogate"\n'
                  'k = 1.0\nr0 = 0.9\nbias = 0.05\n',
        policy='[policy]\nforce_budget_eV_A = 0.1\n',
        extra="[density]\npersist = true\n")
    with pytest.raises(ConfigError, match="plain serial reference MD"):
        load_config(adaptive)
    non_qe = _write_md_toml(
        tmp_path / "harmonic",
        reference='[reference]\nbackend = "harmonic-reference"\n'
                  'k = 1.0\nr0 = 0.9\n',
        extra="[density]\npersist = true\n")
    with pytest.raises(ConfigError, match="plain serial reference MD"):
        load_config(non_qe)


def test_density_persist_requires_keep_retention(tmp_path):
    path = _write_md_toml(
        tmp_path / "results",
        extra='[scratch]\nroot = "./sc"\nretention = "results"\n'
              "[density]\npersist = true\n")
    with pytest.raises(ConfigError, match="retention = 'all'"):
        load_config(path)


# ---------------------------------------------------------------------------
# owner injection through the shared factory path


def test_factory_run_kwargs_offer_the_owner_only_when_declared(tmp_path):
    def plain_factory(*, run_root):
        return None

    def aware_factory(*, run_root, density_registry_run_dir):
        return None

    run_dir = tmp_path / "run"
    kwargs, _ = _factory_run_kwargs(plain_factory, run_dir,
                                    density_registry=True)
    assert "density_registry_run_dir" not in kwargs
    kwargs, _ = _factory_run_kwargs(aware_factory, run_dir,
                                    density_registry=True)
    assert kwargs["density_registry_run_dir"] == str(run_dir)
    assert kwargs["run_root"] == run_dir / "calculations"
    # the flag gates the offer even when the factory declares the parameter
    kwargs, _ = _factory_run_kwargs(aware_factory, run_dir,
                                    density_registry=False)
    assert "density_registry_run_dir" not in kwargs
    # validation passes a throwaway root (construction validates only)
    kwargs, tmp = _factory_run_kwargs(aware_factory, None,
                                      density_registry=True)
    assert kwargs["density_registry_run_dir"] == tmp.name
    tmp.cleanup()


def _qe_persist_toml(root: Path, *, steps=1, checkpoint_interval=2,
                     pseudos_dir: Path | None = None,
                     extra_reference="", extra="") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "structure.extxyz").write_text(STRUCTURE)
    # validate_setup requires an existing pseudo_dir and the mapped UPF
    # files; the fake pw.x never reads them, placeholders satisfy the
    # check.  Runs compared against each other share ONE pseudos dir:
    # pseudo_dir is part of the reference fingerprint, so a per-root copy
    # would change the recorded backend identity.
    pseudos = pseudos_dir if pseudos_dir is not None else root / "pseudos"
    pseudos.mkdir(exist_ok=True)
    (pseudos / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text(
        "placeholder UPF for a fake-pwx run\n")
    script = _success_script(root / "script")
    path = root / "run.toml"
    path.write_text(
        "schema_version = 1\n[run]\nid = \"t\"\ndirectory = \"run\"\n"
        "seed = 42\n[task]\nkind = \"md\"\nmode = \"reference\"\n"
        "[structure]\nfile = \"structure.extxyz\"\n"
        "[reference]\nbackend = \"qe\"\n"
        "pseudo_dir = \"" + str(pseudos) + "\"\n"
        "pw_cmd = [\"bash\", \"" + str(script[1]) + "\"]\n"
        "startpot_file = true\n" + extra_reference +
        "[dynamics]\nensemble = \"nve\"\ntimestep_fs = 0.5\n"
        "steps = " + str(steps) + "\n"
        "temperature_K = 300.0\nvelocity_seed = 7\n"
        "[checkpoint]\ninterval_steps = " + str(checkpoint_interval) + "\n"
        'keep_generations = 2\n'
        '[scratch]\nroot = "./sc"\nretention = "all"\n' + extra)
    return path


def test_build_backends_injects_the_run_owner(tmp_path):
    config = load_config(_qe_persist_toml(tmp_path / "on",
                                          extra="[density]\npersist = true\n"))
    engine, surrogate = _backend_pair(config, tmp_path / "on" / "run")
    assert surrogate is None
    assert engine.density_registry_enabled is True
    assert engine._density_registry_dir == (tmp_path / "on" / "run").resolve()
    # the same reference section without [density] keeps the bridge off
    plain = load_config(_qe_persist_toml(tmp_path / "off"))
    engine2, _ = _backend_pair(plain, tmp_path / "off" / "run")
    assert engine2.density_registry_enabled is False
    assert engine2._density_registry_dir is None


def _backend_pair(config, run_dir):
    from pyraimd2.workflows.setup import build_backends

    return build_backends(config, run_dir=run_dir)


def test_recipe_materialized_input_is_refused(tmp_path):
    # a serial-recipe stage config names the controller's materialized
    # input; the persistent chain does not compose with recipes this round
    root = tmp_path / "stage"
    path = _qe_persist_toml(root, extra="[density]\npersist = true\n")
    text = path.read_text().replace('file = "structure.extxyz"',
                                    'file = "run/initial.traj"')
    path.write_text(text)
    config = load_config(path)
    with pytest.raises(WorkflowError, match="serial-recipe"):
        run_workflow(config, verbose=False, handle_sigint=False)


# ---------------------------------------------------------------------------
# the verified generation load bridge (through the real publish entry)


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_load_published_density_bridge(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    engine.compute(_si(), label="si")
    latest = latest_density_generation(owner)
    assert latest["generation"] == 1

    fingerprint = engine.fingerprint
    for generation in (None, 1):
        payload, reason, resolved = density_publish.load_published_density(
            owner, generation=generation, reference_fingerprint=fingerprint,
            nat=2, species=["Si"])
        assert payload is not None and reason == "" and resolved == 1
        generation_dir, save_dir, manifest = payload
        assert generation_dir == _registry(owner) / "g000001"
        # the save dir comes from the charge-density role entry's parent —
        # never a hardcoded .save layout assumption
        entry = next(e for e in manifest["files"]
                     if e["role"] == "charge-density")
        assert save_dir == (generation_dir / entry["path"]).parent
        assert (save_dir / "charge-density.dat").is_file()

    # a strict generation that does not exist is a reason, never a swap
    payload, reason, resolved = density_publish.load_published_density(
        owner, generation=2, reference_fingerprint=fingerprint,
        nat=2, species=["Si"])
    assert payload is None and resolved is None
    assert "g000002" in reason
    # provenance mismatches refuse honestly
    for kwargs, match in (
            ({"reference_fingerprint": "other:00"}, "reference settings"),
            ({"nat": 3}, "atom count"),
            ({"species": ["C"]}, "species")):
        payload, reason, resolved = density_publish.load_published_density(
            owner, generation=None,
            **{"reference_fingerprint": fingerprint, "nat": 2,
               "species": ["Si"], **kwargs})
        assert payload is None and resolved is None and match in reason
    # a fresh registry answers "no published generation"
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    payload, reason, resolved = density_publish.load_published_density(
        fresh, generation=None, reference_fingerprint=fingerprint,
        nat=2, species=["Si"])
    assert payload is None and "no published density generation" in reason


# ---------------------------------------------------------------------------
# registry-aware source selection on both compute entries


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_fresh_engine_uses_the_registry_latest(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    assert first.last_attempt_records[-1]["density_publish"]["generation"] == 1
    assert first.current_density_generation() == 1

    # a rebuilt engine (a fresh process holds no in-memory latest density)
    # takes the verified persistent generation as its next start
    second = _engine(kind, tmp_path / "b", script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    second.compute(_si(), label="si")
    decision = second.last_density_decision
    assert decision["via"] == "density_registry"
    assert decision["generation"] == 1 and "pinned" not in decision
    assert decision["origin"] == str(_registry(owner).resolve() / "g000001")
    record = second.last_attempt_records[-1]
    assert record["density_input_generation"] == 1
    assert record["density_publish"]["generation"] == 2
    assert second.current_density_generation() == 2
    # the durable attempt record carried the in-flight input reference from
    # before the launch: the first (atomic) attempt declared None, the
    # second declared the registry generation it consumed
    inputs = [r["density_generation"] for r in _records(owner)]
    assert sorted(inputs, key=lambda v: -1 if v is None else v) == [None, 1]
    view = inspect_density_registry(owner)
    assert view["latest"] == 2 and view["attach_history"] == [1, 2]


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_pinned_generation_is_strictly_resolved(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    engine.compute(_si(), label="si")
    engine.compute(_si(), label="si")
    assert latest_density_generation(owner)["generation"] == 2

    # the pin binds exactly the referenced generation — not the latest
    engine.pin_density_generation(1)
    assert engine.last_density_decision["pinned"] is True
    engine.compute(_si(), label="si")
    decision = engine.last_density_decision
    assert decision["via"] == "density_registry"
    assert decision["generation"] == 1 and decision["pinned"] is True
    record = engine.last_attempt_records[-1]
    assert record["density_input_generation"] == 1
    assert record["density_publish"]["generation"] == 3
    # the pin is consumed by the first successful compute
    assert engine._pinned_density_generation is None
    assert engine.current_density_generation() == 3

    # a lost reference refuses the evaluation — never a swap to latest
    engine.pin_density_generation(99)
    with pytest.raises(QeEngineError, match="resume-pinned.*g000099"):
        engine.compute(_si(), label="si")
    assert engine.last_attempt_records == []  # nothing launched


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_fixed_policy_never_reads_the_registry(kind, tmp_path):
    # an external density of known origin (one unmanaged attempt's
    # product; the two adapters lay their attempt directories out
    # differently, so locate the manifest instead of naming a layout)
    external_root = tmp_path / "ext" / "runs"
    external = _engine(kind, tmp_path / "ext",
                       script=_success_script(tmp_path / "ext"),
                       owner=None, run_root=external_root, scratch=None)
    external.compute(_si(), label="si")
    external_source = next(p.parent for p in external_root.rglob("density_manifest.json"))

    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True,
                     density_source=str(external_source),
                     density_source_policy="fixed")
    for _ in range(2):
        engine.compute(_si(), label="si")
        # the configured external source wins every evaluation; the
        # registry is published into but never consulted for the source
        decision = engine.last_density_decision
        assert decision["via"] == "config.density_source"
        assert decision["origin"] == str(external_source.resolve())
        assert engine.last_attempt_records[-1]["density_input_generation"] \
            is None
        # the fixed external policy never holds a registry reference
        assert engine.current_density_generation() is None
    view = inspect_density_registry(owner)
    assert view["latest"] == 2 and view["attach_history"] == [1, 2]
    # the external source is read-only and never touched
    assert (external_source / "tmp" / "pyraimd2.save"
            / "charge-density.dat").is_file()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_no_new_density_keeps_the_previous_generation(kind, tmp_path):
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    assert latest_density_generation(owner)["generation"] == 1

    # a successful run that leaves no density behind still delivers; the
    # previous verified generation stays the source and is honestly
    # recorded as the actual input.  disk_io="minimal" is QE's own "never
    # writes a charge density" mode: the staged input copy stays residue,
    # never this attempt's product.
    second = _engine(kind, tmp_path / "b",
                     script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True, disk_io="minimal")
    result = second.compute(_si(), label="si")
    assert result.forces is not None
    decision = second.last_density_decision
    assert decision["via"] == "density_registry" and decision["generation"] == 1
    record = second.last_attempt_records[-1]
    assert record["status"] == "success"
    assert record["density_available"] is False
    assert record["density_input_generation"] == 1
    assert "density_publish" not in record
    # the head keeps the consumed generation: the current state still
    # derives from it
    assert second.current_density_generation() == 1
    latest = latest_density_generation(owner)
    assert latest["generation"] == 1  # the previous generation stands


# ---------------------------------------------------------------------------
# checkpoint/resume binding through the real workflow entries (fake pw.x)

DENSITY_EXTRA = "[density]\npersist = true\n"


def _persist_run(root: Path, *, steps: int, checkpoint_interval: int = 2,
                 pseudos_dir: Path | None = None) -> Path:
    config = load_config(_qe_persist_toml(
        root, steps=steps, checkpoint_interval=checkpoint_interval,
        pseudos_dir=pseudos_dir, extra=DENSITY_EXTRA))
    run_workflow(config, verbose=False, handle_sigint=False)
    return config.run.directory


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()
            if line.strip()]


def _commits(events: list[dict]) -> dict[int, dict]:
    return {int(e["context"]["evaluation_id"]): e
            for e in events if e.get("type") == EVALUATION_COMMITTED}


def _io_copies(events: list[dict]) -> list[dict]:
    """The density-staging io events, in log order."""
    return [e for e in events if e.get("type") == "task"
            and e.get("purpose") == "density_copy"]


def _checkpoint_manifest(run_dir: Path, generation: int) -> dict:
    return json.loads((run_dir / "checkpoints" / str(generation)
                       / "manifest.json").read_text())


def _rows(run_dir, run_id="t"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def _assert_same_run(rows_a, rows_b):
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        assert int(a.key_value_pairs["step"]) == int(b.key_value_pairs["step"])
        assert a.key_value_pairs["route"] == b.key_value_pairs["route"]
        assert a.data.get("engine_label_id") == b.data.get("engine_label_id")
        np.testing.assert_allclose(a.toatoms().positions, b.toatoms().positions,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.toatoms().get_momenta(),
                                   b.toatoms().get_momenta(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(a.data["driving"]["forces"], dtype=float),
            np.asarray(b.data["driving"]["forces"], dtype=float),
            rtol=0, atol=1e-12)


def _resumed_event(events: list[dict]) -> dict:
    return next(e for e in events if e.get("type") == RESUMED)


def _generation_dir(run_dir: Path, generation: int) -> str:
    return str(run_dir.resolve() / "restart" / "density"
               / f"g{generation:06d}")


def test_resume_binds_the_committed_boundary_generation(tmp_path):
    """Normal tail: coordinates/forces resume from the last committed
    evaluation, so the density reference comes from THAT evaluation's own
    commit record — never from the older selected checkpoint and never
    from a plain registry-latest guess."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=4,
                                  pseudos_dir=pseudos)

    stopped_dir = _persist_run(tmp_path / "stopped", steps=3,
                               pseudos_dir=pseudos)
    # three steps: evaluations 0..3 published g1..g4; the only checkpoint
    # (step 2) recorded the head at its own boundary, g3 — the restored
    # boundary is evaluation 3, whose commit record binds g4
    assert _checkpoint_manifest(stopped_dir, 1)["density_generation"] == 3
    assert latest_density_generation(stopped_dir)["generation"] == 4

    result = resume_workflow(stopped_dir, 1, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 4
    events = _events(stopped_dir)
    resumed = _resumed_event(events)
    assert resumed["checkpoint_generation"] == 1
    assert resumed["density"]["branch"] == "ok"
    assert resumed["density"]["generation"] == 4
    assert resumed["density"]["boundary_evaluation"] == 3
    # the first post-resume evaluation read exactly the boundary's
    # generation (the checkpoint's older g3 was deliberately not used)
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == _generation_dir(stopped_dir, 4)
    # the chain moved on: the resumed evaluation published the next one
    assert latest_density_generation(stopped_dir)["generation"] == 5
    # and the committed evaluation record binds the new head
    assert _commits(events)[4]["density_generation"] == 5
    _assert_same_run(_rows(continuous_dir), _rows(stopped_dir))


def test_resume_heal_binds_the_committed_evaluation(tmp_path, monkeypatch):
    """Crash window: evaluation committed and published, step commit never
    written.  Resume heals the step from the committed trajectory and binds
    the HEALED evaluation's own recorded generation (not the older
    checkpoint field)."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=5,
                                  pseudos_dir=pseudos)

    root = tmp_path / "crashed"
    config = load_config(_qe_persist_toml(
        root, steps=4, checkpoint_interval=2, pseudos_dir=pseudos,
        extra=DENSITY_EXTRA))
    real_append_once = EventLog.append_once
    armed = {"crash": True}

    def flaky_append_once(self, key, event_type, payload):
        if armed["crash"] and key == "step:t:3":
            raise RuntimeError("injected crash before the step commit")
        return real_append_once(self, key, event_type, payload)

    monkeypatch.setattr(EventLog, "append_once", flaky_append_once)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    # evaluation 4 committed and published g5; step 4 never committed; the
    # only checkpoint (step 2) still records g3
    crashed_events = _events(run_dir)
    assert _commits(crashed_events)[4]["density_generation"] == 5
    assert _checkpoint_manifest(run_dir, 1)["density_generation"] == 3
    assert latest_density_generation(run_dir)["generation"] == 5

    armed["crash"] = False
    g5_manifest = json.loads((run_dir / "restart" / "density"
                            / "g000005" / "manifest.json").read_text())
    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 5
    events = _events(run_dir)
    resumed = _resumed_event(events)
    # the healed boundary's own commit record (g5) wins over the selected
    # checkpoint's older field (g3)
    assert resumed["density"]["branch"] == "ok"
    assert resumed["density"]["generation"] == 5
    assert resumed["density"]["boundary_evaluation"] == 4
    assert resumed["density"]["content_digest"] == g5_manifest["content_digest"]
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == _generation_dir(run_dir, 5)
    # the healed step was bound, not recomputed: six real launches in total
    # (evaluations 0..5), exactly like the continuous five-step run
    attempts = [e for e in events
                if e.get("type") == "attempt" and e.get("status") == "success"]
    assert len(attempts) == 6
    assert latest_density_generation(run_dir)["generation"] == 6
    _assert_same_run(_rows(continuous_dir), _rows(run_dir))


def test_resume_falls_back_over_a_corrupt_latest_checkpoint(tmp_path):
    """The latest checkpoint damaged: read_latest_valid walks back, but the
    restored boundary is still the last committed evaluation — the density
    reference comes from that boundary's commit record, not from the older
    checkpoint actually selected."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=5,
                                  pseudos_dir=pseudos)

    run_dir = _persist_run(tmp_path / "damaged", steps=4,
                           pseudos_dir=pseudos)
    # checkpoints: generation 1 (step 2, density g3), generation 2
    # (step 4, density g5); the committed boundary is evaluation 4 (g5)
    assert _checkpoint_manifest(run_dir, 1)["density_generation"] == 3
    assert _checkpoint_manifest(run_dir, 2)["density_generation"] == 5
    # tear the latest checkpoint's state so its digest no longer validates
    state_path = run_dir / "checkpoints" / "2" / "state.json"
    state_path.write_text(state_path.read_text() + "torn")

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 5
    events = _events(run_dir)
    resumed = _resumed_event(events)
    assert resumed["checkpoint_generation"] == 1  # fell back from 2
    # the restored boundary is evaluation 4 — its own record binds g5,
    # not the fallback checkpoint's g3
    assert resumed["density"]["branch"] == "ok"
    assert resumed["density"]["generation"] == 5
    assert resumed["density"]["boundary_evaluation"] == 4
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == _generation_dir(run_dir, 5)
    _assert_same_run(_rows(continuous_dir), _rows(run_dir))

    # now damage the generation the NEW boundary (evaluation 5 → g6)
    # references: the resume must refuse — never substitute another one
    payload = (run_dir / "restart" / "density" / "g000006" / "tmp"
               / "pyraimd2.save" / "charge-density.dat")
    payload.write_text("tampered")
    before = _events(run_dir)
    with pytest.raises(WorkflowError, match=r"g000006.*substituted"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    # nothing was appended, computed or advanced; the registry is untouched
    assert _events(run_dir) == before
    assert len(_rows(run_dir)) == len(_rows(continuous_dir))
    view = inspect_density_registry(run_dir)
    assert view["latest"] == 6
    assert next(r for r in view["resources"]
                if r["name"] == "g000006")["kind"] == "corrupt_generation"


def test_resume_with_legacy_boundary_record_falls_back(tmp_path):
    """A committed boundary whose record predates density tracking (no
    density_generation field anywhere: an old-format log) resumes with the
    old external-initialization policy — no fabricated reference, and no
    guess at a neighboring generation."""
    pseudos = tmp_path / "pseudos"
    run_dir = _persist_run(tmp_path / "legacy", steps=2, pseudos_dir=pseudos)
    # simulate the pre-chain record format: strip the field from EVERY
    # evaluation commit and from the checkpoint manifest
    events_path = run_dir / "events.jsonl"
    rewritten = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == EVALUATION_COMMITTED:
            removed = event.pop("density_generation")
            assert isinstance(removed, int)  # the pre-strip record had it
        rewritten.append(json.dumps(event, sort_keys=True))
    events_path.write_text("\n".join(rewritten) + "\n")
    manifest_path = run_dir / "checkpoints" / "1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.pop("density_generation") == 3
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
    events = _events(run_dir)
    resumed = _resumed_event(events)
    assert resumed["density"]["branch"] == "external_initialization_required"
    assert resumed["density"]["generation"] is None
    assert resumed["density"]["boundary_evaluation"] == 2
    # no pin: the first new evaluation resolves by policy — the verified
    # registry latest at that moment
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == _generation_dir(run_dir, 3)
    assert latest_density_generation(run_dir)["generation"] == 4


def test_resume_with_explicit_null_boundary_initializes_externally(tmp_path):
    """A boundary whose commit record DECLARES no density reference
    (explicit null — here every commit of a fixed-external-source run)
    takes the external-initialization branch; the first new evaluation
    follows the configured fixed source, never a registry guess."""
    pseudos = tmp_path / "pseudos"
    # an external density of known origin (one unmanaged attempt's product,
    # produced with the SAME pseudopotential identity so the reference
    # fingerprint matches the workflow engine's)
    pseudos.mkdir(exist_ok=True)
    (pseudos / "Si.pbe-n-kjpaw_psl.1.0.0.UPF").write_text(
        "placeholder UPF for a fake-pwx run\n")
    external_root = tmp_path / "ext" / "runs"
    external_root.mkdir(parents=True)
    external = QeEngine(QeConfig(pseudo_dir=str(pseudos),
                                 pw_cmd=_success_script(tmp_path / "ext")),
                        run_root=external_root)
    external.compute(_si(), label="si")
    external_source = next(p.parent
                           for p in external_root.rglob("density_manifest.json"))

    root = tmp_path / "fixed"
    config_path = _qe_persist_toml(root, steps=2, pseudos_dir=pseudos,
                                   extra=DENSITY_EXTRA)
    config_path.write_text(config_path.read_text().replace(
        "startpot_file = true\n",
        "startpot_file = true\n"
        f'density_source = "{external_source}"\n'
        'density_source_policy = "fixed"\n'))
    config = load_config(config_path)
    run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    # every commit of a fixed-policy run declares the explicit null
    commits = _commits(_events(run_dir))
    assert all(commits[e]["density_generation"] is None for e in range(3))

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 3
    events = _events(run_dir)
    resumed = _resumed_event(events)
    assert resumed["density"]["branch"] == "external_initialization_required"
    assert resumed["density"]["generation"] is None
    assert resumed["density"]["boundary_evaluation"] == 2
    # the first resumed evaluation read the configured external source —
    # no registry generation was ever pinned or consumed
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == str(external_source.resolve())
    inputs = [r["density_generation"] for r in _records(run_dir)]
    assert all(value is None for value in inputs)


def test_resume_with_corrupt_boundary_density_refuses_before_any_write(
        tmp_path):
    """The actual boundary's referenced generation damaged: the resume
    refuses BEFORE the heal event, the RESUMED record or any new
    computation — the run directory stays byte-identical."""
    pseudos = tmp_path / "pseudos"
    run_dir = _persist_run(tmp_path / "corrupt", steps=3, pseudos_dir=pseudos)
    # the committed boundary is evaluation 3 → g4
    member = (run_dir / "restart" / "density" / "g000004" / "tmp"
              / "pyraimd2.save" / "charge-density.dat")
    member.write_text("corrupt boundary dependency")
    before = _events(run_dir)
    outputs_before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    with pytest.raises(WorkflowError, match=r"g000004.*substituted"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert _events(run_dir) == before
    # no new calculation, no new file anywhere in the run directory
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == \
        outputs_before
    view = inspect_density_registry(run_dir)
    assert view["latest"] == 4
    assert next(r for r in view["resources"]
                if r["name"] == "g000004")["kind"] == "corrupt_generation"


def test_resume_heal_with_corrupt_boundary_density_writes_nothing(
        tmp_path, monkeypatch):
    """Crash window + the healed boundary's density damaged: the refusal
    precedes the heal append — the step event is NOT written."""
    pseudos = tmp_path / "pseudos"
    root = tmp_path / "crashed"
    config = load_config(_qe_persist_toml(
        root, steps=4, checkpoint_interval=2, pseudos_dir=pseudos,
        extra=DENSITY_EXTRA))
    real_append_once = EventLog.append_once
    armed = {"crash": True}

    def flaky_append_once(self, key, event_type, payload):
        if armed["crash"] and key == "step:t:3":
            raise RuntimeError("injected crash before the step commit")
        return real_append_once(self, key, event_type, payload)

    monkeypatch.setattr(EventLog, "append_once", flaky_append_once)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_workflow(config, verbose=False, handle_sigint=False)
    run_dir = config.run.directory
    armed["crash"] = False
    # the healed boundary (evaluation 4) binds g5 — corrupt it
    member = (run_dir / "restart" / "density" / "g000005" / "tmp"
              / "pyraimd2.save" / "charge-density.dat")
    member.write_text("corrupt healed-boundary dependency")
    before = _events(run_dir)
    with pytest.raises(WorkflowError, match=r"g000005.*substituted"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    after = _events(run_dir)
    assert after == before  # no heal step event, no RESUMED, nothing
    assert not any(e.get("type") == STEP_COMPLETED and e["step_id"] == 3
                   for e in after)


def test_resume_with_invalid_boundary_density_field_refuses(tmp_path):
    """A boundary commit whose density field is present but not a valid
    reference (a torn/foreign write) blocks the resume — never skipped,
    never substituted."""
    pseudos = tmp_path / "pseudos"
    run_dir = _persist_run(tmp_path / "invalid", steps=2, pseudos_dir=pseudos)
    events_path = run_dir / "events.jsonl"
    rewritten = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == EVALUATION_COMMITTED \
                and int(event["context"]["evaluation_id"]) == 2:
            event["density_generation"] = "g3"  # not a positive int
        rewritten.append(json.dumps(event, sort_keys=True))
    events_path.write_text("\n".join(rewritten) + "\n")
    before = _events(run_dir)
    with pytest.raises(WorkflowError, match="not a valid positive integer"):
        resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert _events(run_dir) == before


def _write_zero_step_checkpoint(run_dir: Path, config, *,
                                density: object = "legacy") -> None:
    """A step-0 checkpoint for the defensive zero-step resume branch: the
    state a run that checkpointed its initial boundary would carry.
    ``density`` is the recorded density_generation value; "legacy" omits
    the field entirely (the pre-chain checkpoint record)."""
    store = Store(run_dir / "trajectory.db")
    initial = store._row_at_step(config.run.id, -1)
    atoms = initial.toatoms()
    driving = initial.data["driving"]
    run_start = next(e for e in _events(run_dir) if e.get("type") == "run_start")
    state = {
        "run_id": config.run.id,
        "driver": f"plain-{config.dynamics.ensemble}",
        "section": config.task.mode,
        "nsteps": 0,
        "task_counter": 1,
        "model_id": run_start["model_id"],
        "engine_fingerprint": run_start["reference_id"],
        "timestep_fs": config.dynamics.timestep_fs,
        "integrator": run_start["workflow"]["integrator"],
        "thermostat": None,
        "driving_energy_eV": float(driving["energy"]),
        "constraint": None,
    }
    arrays = {
        "numbers": atoms.numbers,
        "cell": atoms.cell.array,
        "pbc": np.asarray(atoms.pbc),
        "masses": atoms.get_masses(),
        "initial_charges": atoms.get_initial_charges(),
        "initial_magmoms": atoms.get_initial_magnetic_moments(),
        "positions": atoms.positions,
        "momenta": atoms.get_momenta(),
        "driving_forces": np.asarray(driving["forces"], dtype=float),
    }
    manifest_extra = {
        "run_id": config.run.id,
        "nsteps": 0,
        "physical_time_fs": 0.0,
        "last_event_seq": _events(run_dir)[-1]["seq"],
        "store_schema_version": STORE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "attempt_ledger": ATTEMPT_LEDGER_PHYSICAL_V1,
        "software_version": __version__,
    }
    if density != "legacy":
        manifest_extra["density_generation"] = density
    CheckpointManager(run_dir).write(1, state, arrays, manifest_extra)
    store.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_zero_step_binds_the_checkpoint_record(tmp_path, monkeypatch,
                                                      legacy):
    """Zero complete steps with a step-0 checkpoint: the checkpoint's own
    arrays are the restored state, so its recorded field decides the
    density reference; a legacy step-0 checkpoint (no field) falls back to
    external initialization."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=1,
                                  pseudos_dir=pseudos)
    root = tmp_path / "zerostep"
    config = load_config(_qe_persist_toml(root, steps=1, pseudos_dir=pseudos,
                                          extra=DENSITY_EXTRA))
    real_append = EventLog.append

    def flaky_append(self, event_type, payload):
        if event_type == "attempt" \
                and payload.get("request_id") == "t-task-2":
            raise RuntimeError("injected failure at the first MD step")
        return real_append(self, event_type, payload)

    monkeypatch.setattr(EventLog, "append", flaky_append)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_workflow(config, verbose=False, handle_sigint=False)
    monkeypatch.setattr(EventLog, "append", real_append)
    run_dir = config.run.directory
    # zero complete steps, one committed evaluation, one published seed
    assert set(_commits(_events(run_dir))) == {0}
    assert latest_density_generation(run_dir)["generation"] == 1
    _write_zero_step_checkpoint(run_dir, config,
                                density=("legacy" if legacy else 1))

    result = resume_workflow(run_dir, 1, verbose=False, handle_sigint=False)
    assert result.steps_completed == 1
    events = _events(run_dir)
    resumed = _resumed_event(events)
    assert resumed["checkpoint_generation"] == 1
    assert resumed["density"]["boundary_evaluation"] == 0
    if legacy:
        assert resumed["density"]["branch"] == \
            "external_initialization_required"
        assert resumed["density"]["generation"] is None
    else:
        assert resumed["density"]["branch"] == "ok"
        assert resumed["density"]["generation"] == 1
    # either way the first new evaluation read g1 (the only generation)
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    assert len(io_after) == 1
    assert io_after[0]["provenance"]["from"] == _generation_dir(run_dir, 1)
    _assert_same_run(_rows(continuous_dir), _rows(run_dir))


# ---------------------------------------------------------------------------
# protected reclaim of consumed managed scratch


def test_serial_chain_publishes_reuses_and_reclaims(tmp_path):
    """Scenario: normal continuous progress under the delayed release.
    Every evaluation publishes one generation, the next one reads the
    verified registry latest, and a producer's scratch is released and
    reclaimed only once a LATER ordinary calculation independently
    consumed that exact seed — the terminal producer stays kept as a
    protected resource — while latest and the retained checkpoint
    references are never touched."""
    run_dir = _persist_run(tmp_path / "chain", steps=3, checkpoint_interval=2)
    events = _events(run_dir)
    # four evaluations (0..3) published one generation each, in order
    view = inspect_density_registry(run_dir)
    assert view["state"] == "ok"
    assert view["attach_history"] == [1, 2, 3, 4] and view["latest"] == 4
    # every evaluation from step 1 on read the verified persistent chain
    copies = _io_copies(events)
    assert [c["provenance"]["from"] for c in copies] == [
        _generation_dir(run_dir, g) for g in (1, 2, 3)]
    # each commit record binds the persistent head at its own boundary
    commits = _commits(events)
    assert [commits[e]["density_generation"] for e in range(4)] == [1, 2, 3, 4]
    # the checkpoint (step 2) recorded its own boundary's head
    assert _checkpoint_manifest(run_dir, 1)["density_generation"] == 3
    # the release chain: evaluation N's commit recorded the pending receipt
    # on its own attempt; evaluation N+1's independent successful read of
    # gN completed the release of gN's producer
    records = _records(run_dir)
    assert len(records) == 4
    by_output = {r["released_evidence"]["density_output_generation"]: r
                 for r in records}
    for generation in (1, 2, 3):
        record = by_output[generation]
        assert record["state"] == "cleaned"
        assert record["density_generation"] is None  # consumed, unprotected
        evidence = record["released_evidence"]
        # the receipt binds the seed generation, its content digest, the
        # compatible settings and the actual independent read
        assert evidence["proof"] == "consumed"
        assert evidence["evaluation_id"] == generation - 1
        assert evidence["seed_content_digest"]
        assert evidence["reference_fingerprint"]
        consumed_by = evidence["consumed_by"]
        assert consumed_by["evaluation_id"] == generation
        assert consumed_by["staged_from"] == _generation_dir(
            run_dir, generation)
        assert consumed_by["staged_bytes"] > 0
        assert not Path(record["scratch_dir"]).exists()
    # the terminal producer (g4) was never independently consumed: it stays
    # kept with its pending receipt — a protected resource, not a failure
    terminal = by_output[4]
    assert terminal["state"] == "kept"
    assert terminal["released_evidence"]["proof"] == \
        "pending_independent_consumption"
    assert terminal["released_evidence"]["evaluation_id"] == 3
    assert terminal["density_generation"] == 3  # its own consumed input
    assert Path(terminal["scratch_dir"]).is_dir()
    # the lightweight archived results stay, outside the scratch root; the
    # reclaimed attempts' manifests no longer claim the .save trees, the
    # kept terminal attempt's manifest still does (its tree survives)
    assert list((run_dir / "calculations").rglob("pw.out"))
    manifests = [(m, json.loads(m.read_text()))
                 for m in (run_dir / "calculations").rglob(
                     "density_manifest.json")]
    assert manifests
    removed = [m.get("scratch_removed") is True for _, m in manifests]
    assert sorted(removed) == [False, True, True, True]
    # the driver's own reclaim ran after each step commit (and each
    # checkpoint retention update): g1 and g2 were once attached, became
    # unreferenced and were reclaimed with durable tombstones; g3 (the
    # retained checkpoint's reference) and g4 (latest + committed boundary
    # + the kept terminal producer's unproven seed) are kept.  The plan
    # reports the tombstones as reclaimed — never as corruption — and the
    # generation numbers are never reused.
    plan = plan_density_reclaim(run_dir)
    by_generation = {r["generation"]: r["decision"]
                     for r in plan["resources"]
                     if r["kind"] in ("generation", "reclaimed_generation")}
    assert by_generation[4] == "keep" and by_generation[3] == "keep"
    assert by_generation[1] == "reclaimed" and by_generation[2] == "reclaimed"
    assert not (_registry(run_dir) / "g000001").exists()
    assert not (_registry(run_dir) / "g000002").exists()
    assert (_registry(run_dir) / "g000003").is_dir()
    assert (_registry(run_dir) / "g000004").is_dir()
    view = inspect_density_registry(run_dir)
    assert view["reclaimed"] == [1, 2]
    assert view["attach_history"] == [1, 2, 3, 4]  # history intact


def test_resume_then_continue_reclaims(tmp_path):
    """Scenario: resume and keep going.  The first new evaluation uses the
    pinned boundary generation (the last committed evaluation's own
    record), the chain advances from there, and the reclaim of consumed
    scratch continues across the process boundary."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=5,
                                  pseudos_dir=pseudos)
    stopped_dir = _persist_run(tmp_path / "stopped", steps=3,
                               pseudos_dir=pseudos)
    # the terminal producer is unproven at the stop: kept, pending receipt
    by_evaluation = {r["released_evidence"]["evaluation_id"]: r
                     for r in _records(stopped_dir)}
    assert [by_evaluation[e]["state"] for e in range(4)] == \
        ["cleaned"] * 3 + ["kept"]

    result = resume_workflow(stopped_dir, 2, verbose=False,
                             handle_sigint=False)
    assert result.steps_completed == 5
    events = _events(stopped_dir)
    resumed = _resumed_event(events)
    assert resumed["density"]["branch"] == "ok"
    assert resumed["density"]["generation"] == 4
    assert resumed["density"]["boundary_evaluation"] == 3
    after = events.index(resumed)
    io_after = _io_copies(events[after:])
    # the first resumed evaluation used the pinned boundary generation,
    # the second followed the chain's latest (g5, published by step 4)
    assert [c["provenance"]["from"] for c in io_after] == [
        _generation_dir(stopped_dir, 4), _generation_dir(stopped_dir, 5)]
    assert latest_density_generation(stopped_dir)["generation"] == 6
    commits = _commits(events)
    assert commits[4]["density_generation"] == 5
    assert commits[5]["density_generation"] == 6
    # the resumed evaluations' consumed scratch was reclaimed as well:
    # the pre-stop terminal producer (g4) was released when the first
    # resumed evaluation independently consumed its seed, across the
    # process boundary; the new terminal producer stays kept
    records = _records(stopped_dir)
    assert len(records) == 6
    by_evaluation = {r["released_evidence"]["evaluation_id"]: r
                     for r in records}
    assert [by_evaluation[e]["state"] for e in range(6)] == \
        ["cleaned"] * 5 + ["kept"]
    assert [by_evaluation[e]["released_evidence"]["proof"]
            for e in range(6)] == \
        ["consumed"] * 5 + ["pending_independent_consumption"]
    # the pre-stop terminal producer (g4) was released when the first
    # resumed evaluation independently consumed its seed — across the
    # process boundary
    g4_receipt = by_evaluation[3]["released_evidence"]
    assert g4_receipt["density_output_generation"] == 4
    assert g4_receipt["consumed_by"]["evaluation_id"] == 4
    _assert_same_run(_rows(continuous_dir), _rows(stopped_dir))


def test_driver_reclaim_keeps_the_registry_bounded(tmp_path):
    """Longer serial run: the driver reclaims after every step commit and
    checkpoint retention update.  At the end the registry holds exactly
    the retained checkpoints' references plus the latest generation;
    every superseded unreferenced generation is tombstoned; only the
    terminal producer scratch is kept.  A resume afterwards continues to
    bound the registry as checkpoint retention rotates."""
    pseudos = tmp_path / "pseudos"
    continuous_dir = _persist_run(tmp_path / "continuous", steps=10,
                                  pseudos_dir=pseudos)
    run_dir = _persist_run(tmp_path / "bounded", steps=8, pseudos_dir=pseudos)
    # 9 evaluations published g1..g9; checkpoints at steps 2,4,6,8 recorded
    # g3,g5,g7,g9; retention keeps the last two (steps 6 and 8)
    view = inspect_density_registry(run_dir)
    assert view["latest"] == 9
    assert sorted(r["generation"] for r in view["resources"]
                  if r["kind"] == "generation") == [7, 9]
    assert view["reclaimed"] == [1, 2, 3, 4, 5, 6, 8]
    assert view["attach_history"] == list(range(1, 10))
    records = _records(run_dir)
    assert [r["state"] for r in records].count("kept") == 1

    # resume across the reclaimed registry: the boundary (evaluation 8 ->
    # g9) survived every reclaim; the chain continues bounded
    result = resume_workflow(run_dir, 2, verbose=False, handle_sigint=False)
    assert result.steps_completed == 10
    resumed = _resumed_event(_events(run_dir))
    assert resumed["density"]["generation"] == 9
    view = inspect_density_registry(run_dir)
    # checkpoint retention rotated to steps 8 (g9) and 10 (g11); g7 lost
    # its reference at the rotation and was reclaimed in order, and g10
    # (superseded by g11, its producer consumed) was reclaimed as well
    assert sorted(r["generation"] for r in view["resources"]
                  if r["kind"] == "generation") == [9, 11]
    assert view["reclaimed"] == [1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert view["latest"] == 11
    assert view["attach_history"] == list(range(1, 12))
    records = _records(run_dir)
    assert [r["state"] for r in records].count("kept") == 1
    _assert_same_run(_rows(continuous_dir), _rows(run_dir))


def test_release_consumed_state_guard(tmp_path):
    """scratch.release_consumed: only a kept attempt bridges back to
    archived — every other state refuses, never a bypass."""
    root = tmp_path / "tmp"
    archive_dir = tmp_path / "runs" / "case-000000" / "attempt-1"
    archive_dir.mkdir(parents=True)
    handle = scratch_mod.allocate(
        run_root=tmp_path / "runs", scratch_root=root,
        run_uuid="run-deadbeef02", backend_role="reference",
        request_id="req-1", attempt_id="attempt-1",
        archive_dir=archive_dir, retention="all")
    (handle.scratch_dir / "pw.in").write_text("in")
    (handle.scratch_dir / "pw.out").write_text("out")
    with pytest.raises(scratch_mod.ScratchError, match="allocated"):
        scratch_mod.release_consumed(handle, evidence={"evaluation_id": 0})
    scratch_mod.archive(handle, ["pw.in", "pw.out"])
    with pytest.raises(scratch_mod.ScratchError, match="archived"):
        scratch_mod.release_consumed(handle, evidence={"evaluation_id": 0})
    scratch_mod.mark_kept(handle)
    with pytest.raises(scratch_mod.ScratchError, match="evaluation_id"):
        scratch_mod.release_consumed(handle, evidence={})
    record = scratch_mod.release_consumed(
        handle,
        evidence={"evaluation_id": 0, "density_input_generation": None,
                  "density_output_generation": 1, "released_unix": 0.0})
    assert record["state"] == "archived"
    assert record["density_generation"] is None  # in-flight protection ends
    assert record["released_evidence"]["density_output_generation"] == 1
    # the existing machinery takes it from there (verify + delete)
    receipt = scratch_mod.cleanup(handle)
    assert receipt["status"] == "cleaned"
    assert not handle.scratch_dir.exists()


def test_release_reports_not_applicable_without_registry(tmp_path):
    engine = _engine("qe", tmp_path, script=_success_script(tmp_path),
                     owner=None, run_root=tmp_path / "runs",
                     scratch=tmp_path / "scratch", startpot_file=True)
    engine.compute(_si(), label="si")
    receipt = engine.release_consumed_scratch(evaluation_id=0)
    assert receipt["status"] == "not_applicable"
    # the pre-feature lifecycle is untouched
    assert _records(tmp_path / "runs")[0]["state"] == "kept"


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_release_without_new_density_reclaims(kind, tmp_path):
    """Scenario: no new density.  The attempt consumed the previous
    verified generation but produced nothing; nothing persistent depends
    on its save tree, so its own release proceeds directly — and its
    independent successful read completes the producer's pending
    release."""
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    pending = first.release_consumed_scratch(evaluation_id=0)
    assert pending["status"] == "kept"
    assert pending["pending_release"]["proof"] == \
        "pending_independent_consumption"
    second = _engine(kind, tmp_path / "b", script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True, disk_io="minimal")
    second.compute(_si(), label="si")
    receipt = second.release_consumed_scratch(evaluation_id=1)
    assert receipt["status"] == "cleaned"
    assert receipt["release"]["evidence"]["density_output_generation"] is None
    assert receipt["release"]["evidence"]["density_input_generation"] == 1
    # the second attempt's independent read of g1 completed the first
    # attempt's pending release
    consumption = receipt["consumption"]
    assert consumption["status"] == "cleaned"
    assert consumption["generation"] == 1
    evidence = consumption["release"]["evidence"]
    assert evidence["proof"] == "consumed"
    assert evidence["evaluation_id"] == 0  # the producing evaluation
    assert evidence["consumed_by"]["evaluation_id"] == 1
    records = _records(owner)
    assert len(records) == 2
    assert all(r["state"] == "cleaned" for r in records)
    assert all(not Path(r["scratch_dir"]).exists() for r in records)
    # the published generation itself stands
    assert latest_density_generation(owner)["generation"] == 1
    assert second.current_density_generation() == 1


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_publish_failure_keeps_the_scratch(kind, tmp_path, monkeypatch):
    """A publication that did not complete keeps the attempt scratch: it
    holds the only verified copy of the density."""
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)

    def boom(*args, **kwargs):
        raise RestartError("injected registry refusal")

    monkeypatch.setattr(
        density_publish.restart_mod, "publish_density_generation_from_attempt",
        boom)
    result = engine.compute(_si(), label="si")
    assert result.forces is not None  # the delivered label stands
    receipt = engine.release_consumed_scratch(evaluation_id=0)
    assert receipt["status"] == "kept"
    assert "did not complete" in receipt["reason"]
    record = _records(owner)[0]
    assert record["state"] == "kept"
    assert Path(record["scratch_dir"]).is_dir()
    assert not _registry(owner).exists()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_fixed_external_source_survives_reclaim(kind, tmp_path):
    """Scenario: external source.  Every evaluation reads the configured
    external directory, publications still land in the registry — and the
    external directory is never part of any reclaim.  The fixed policy
    never consumes a registry seed, so no publication is ever proven by
    an independent read: every producer scratch stays kept with its
    pending receipt (unproven seeds are protected, never deleted on the
    strength of a hash check alone)."""
    external_root = tmp_path / "ext" / "runs"
    external = _engine(kind, tmp_path / "ext",
                       script=_success_script(tmp_path / "ext"),
                       owner=None, run_root=external_root, scratch=None)
    external.compute(_si(), label="si")
    external_source = next(p.parent
                           for p in external_root.rglob("density_manifest.json"))

    def _tree(root: Path) -> dict:
        return {str(p.relative_to(root)): p.read_bytes()
                for p in sorted(root.rglob("*")) if p.is_file()}

    before = _tree(external_source)
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True, density_source=str(external_source),
                     density_source_policy="fixed")
    for evaluation_id in range(2):
        engine.compute(_si(), label="si")
        receipt = engine.release_consumed_scratch(
            evaluation_id=evaluation_id)
        assert receipt["status"] == "kept"
        assert receipt["pending_release"]["proof"] == \
            "pending_independent_consumption"
        assert receipt["consumption"] is None  # nothing was consumed
        decision = engine.last_density_decision
        assert decision["via"] == "config.density_source"
        assert decision["origin"] == str(external_source.resolve())
        assert engine.current_density_generation() is None
    assert _tree(external_source) == before  # byte-identical, never touched
    assert inspect_density_registry(owner)["latest"] == 2
    records = _records(owner)
    assert all(r["state"] == "kept" for r in records)
    assert all(Path(r["scratch_dir"]).is_dir() for r in records)


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_pending_release_receipt_is_idempotent(kind, tmp_path):
    """The pending receipt is persisted once per attempt: a repeated
    release call (e.g. after a cache-hit evaluation) does not rewrite it
    and completes nothing new."""
    owner, run_root, scratch = _layout(tmp_path)
    engine = _engine(kind, tmp_path, script=_success_script(tmp_path),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    engine.compute(_si(), label="si")
    first = engine.release_consumed_scratch(evaluation_id=0)
    assert first["status"] == "kept"
    record = _records(owner)[0]
    receipt_1 = record["released_evidence"]
    assert receipt_1["proof"] == "pending_independent_consumption"
    # a repeated call (as a cache-hit evaluation would issue) keeps the
    # record byte-identical and completes no consumption
    second = engine.release_consumed_scratch(evaluation_id=1)
    assert second["status"] == "kept"
    assert second["consumption"] is None
    record2 = _records(owner)[0]
    assert record2 == record
    assert record2["released_evidence"] == receipt_1
    assert Path(record2["scratch_dir"]).is_dir()


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_atomic_fallback_is_not_a_consumption(kind, tmp_path):
    """A warm start that fails and recovers by an atomic retry proves
    NOTHING about the seed: the producer of the read generation stays
    kept, and the fallback attempt's own publication is a new unproven
    seed of its own."""
    owner, run_root, scratch = _layout(tmp_path)
    count = tmp_path / "calls"
    # the fake refuses a staged warm start (a charge density without the
    # profile member) and succeeds cold
    script = _fake_pwx(
        tmp_path / "script",
        "#!/bin/bash\nset -eu\n"
        f"echo x >> {count}\n"
        "seed=tmp/pyraimd2.save\n"
        "if test -f \"$seed/charge-density.dat\" && "
        "! test -f \"$seed/profile-required.bin\"; then\n"
        "  echo 'Error in routine review_profile_read (1):'\n"
        "  echo 'synthetic profile required restart member was lost'\n"
        "  exit 1\nfi\n"
        "mkdir -p \"$seed\"\n"
        "echo density > \"$seed/charge-density.dat\"\n"
        "echo '<xml/>' > \"$seed/data-file-schema.xml\"\n"
        "echo unique-restart-state > \"$seed/profile-required.bin\"\n"
        f"cat {FIXTURE.resolve()}\n")
    engine = _engine(kind, tmp_path, script=script, owner=owner,
                     run_root=run_root, scratch=scratch, startpot_file=True,
                     max_retries=2)
    engine.compute(_si(), label="si")  # cold start, publishes g1
    pending = engine.release_consumed_scratch(evaluation_id=0)
    assert pending["status"] == "kept"
    engine.compute(_si(), label="si")  # warm read fails, atomic retry succeeds
    assert engine.last_attempt_records[-1]["start"] == "atomic"
    assert engine.last_attempt_records[-1]["density_input_generation"] is None
    receipt = engine.release_consumed_scratch(evaluation_id=1)
    assert receipt["status"] == "kept"  # the fallback's own seed is unproven
    # the failed warm read is NOT an independent consumption: g1's
    # producer stays kept
    assert receipt["consumption"] is None
    records = _records(owner)
    assert len(records) == 3  # eval0, eval1-attempt-1 (failed), eval1-attempt-2
    states = sorted(r["state"] for r in records)
    assert states == ["failed_kept", "kept", "kept"]
    assert all(Path(r["scratch_dir"]).is_dir() for r in records)
    assert latest_density_generation(owner)["generation"] == 2
    # three physical launches: cold, the failed warm read, the atomic retry
    assert len(count.read_text().splitlines()) == 3


# ---------------------------------------------------------------------------
# actual-read evidence gate (the consumption proof requires the solver to
# have really read the staged seed, not merely to have staged + succeeded)

WARM_READ_EXCERPT = FIXTURE.parent / "qe_warm_density_read_excerpt.out"
COLD_READ_EXCERPT = FIXTURE.parent / "qe_cold_no_read_excerpt.out"

_WARM_INPUT = ("&CONTROL\n  calculation = 'scf'\n  prefix = 'pyraimd2'\n"
               "  outdir = './tmp'\n/\n&ELECTRONS\n  startingpot = 'file'\n/\n")
_ATOMIC_INPUT = ("&CONTROL\n  calculation = 'scf'\n  prefix = 'pyraimd2'\n"
                 "  outdir = './tmp'\n/\n&ELECTRONS\n/\n")


def test_parse_density_read_evidence_real_qe_logs():
    """Positive/negative controls over the returned real-QE A4 logs (QE 7.5,
    HDF5 build, PAW): the warm attempt's output carries exactly one
    density-read marker naming its own staged save tree; the cold
    (atomic-start) output carries none."""
    from pyraimd2.engines.qe_engine import parse_density_read_evidence

    # the returned raw log regions (Si8 warm continuation / cold start);
    # host-specific paths are trimmed out, the read region is verbatim
    warm = WARM_READ_EXCERPT.read_text()
    cold = COLD_READ_EXCERPT.read_text()
    assert "The initial density is read from file" in warm  # fixture sanity
    assert "Initial potential from superposition" in cold
    evidence, reason = parse_density_read_evidence(warm, _WARM_INPUT)
    assert reason is None and evidence is not None
    assert evidence["marker"] == "The initial density is read from file :"
    assert evidence["read_path"] == "./tmp/pyraimd2.save/charge-density"
    assert evidence["startpot"] == "file"
    # the real cold output is honest evidence of NO read: keep the source
    # (an atomic input never requested the read — the input check fires
    # first)
    evidence, reason = parse_density_read_evidence(cold, _ATOMIC_INPUT)
    assert evidence is None and "did not request startingpot" in reason
    # ... and when the input DID request a file start but the build's
    # output stays silent (unknown/older wording: never assume the read)
    evidence, reason = parse_density_read_evidence(cold, _WARM_INPUT)
    assert evidence is None and "no density-read marker" in reason


@pytest.mark.parametrize(
    "output_text,input_text,match",
    [
        # marker but an atomic input: decision and input disagree
        (("x\nThe initial density is read from file :\n"
          "./tmp/pyraimd2.save/charge-density\n"), _ATOMIC_INPUT,
         "did not request startingpot"),
        # two markers: ambiguous
        (("The initial density is read from file :\n"
          "./tmp/pyraimd2.save/charge-density\n"
          "The initial density is read from file :\n"
          "./tmp/pyraimd2.save/charge-density\n"),
         _WARM_INPUT, "ambiguous"),
        # the read path is absolute (not the attempt's staged tree)
        (("The initial density is read from file :\n"
          "/abs/registry/g000001/tmp/pyraimd2.save/"
          "charge-density\n"),
         _WARM_INPUT, "not the attempt's own staged save tree"),
        # the read path escapes the staged tree
        (("The initial density is read from file :\n"
          "./tmp/../elsewhere/charge-density\n"),
         _WARM_INPUT, "not the attempt's own staged save tree"),
        # marker without a path line
        (("The initial density is read from file :\n"),
         _WARM_INPUT, "no following path line"),
    ])
def test_parse_density_read_evidence_rejects_nonproofs(output_text,
                                                       input_text, match):
    from pyraimd2.engines.qe_engine import parse_density_read_evidence

    evidence, reason = parse_density_read_evidence(output_text, input_text)
    assert evidence is None and match in reason


@pytest.mark.parametrize("backend", ["qe", "qe-ase"])
def test_success_without_actual_seed_read_preserves_producer(backend,
                                                             tmp_path):
    """The no-read canary: a successful solver that never reads the staged
    seed (no read marker in its output) must NOT release the producer —
    staging plus success is not consumption.  The run itself completes:
    the labels are delivered; only the deletion gate refuses."""
    root = tmp_path / backend
    config_path = _qe_persist_toml(root, steps=1, extra=DENSITY_EXTRA)
    config_path.write_text(config_path.read_text().replace(
        'backend = "qe"', f'backend = "{backend}"'))
    script = root / "script" / "fake_pwx.sh"
    script.write_text(
        "#!/bin/bash\nset -eu\n"
        "mkdir -p tmp/pyraimd2.save\n"
        "echo fresh-atomic-density > tmp/pyraimd2.save/charge-density.dat\n"
        "echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml\n"
        "echo unique-producer-member > tmp/pyraimd2.save/profile-required.bin\n"
        f"cat {FIXTURE.resolve()}\n")
    config = load_config(config_path)
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 1  # the computation itself is fine
    records = _records(config.run.directory)
    producer = next(r for r in records
                    if r.get("released_evidence", {}).get("evaluation_id")
                    == 0)
    consumer = next(r for r in records
                    if r.get("released_evidence", {}).get("evaluation_id")
                    == 1)
    # the producer stays kept — its unique member survives
    assert producer["state"] == "kept"
    assert producer["released_evidence"]["proof"] == \
        "pending_independent_consumption"
    assert (Path(producer["scratch_dir"]) / "tmp" / "pyraimd2.save"
            / "profile-required.bin").is_file()
    # the consumer's own seed is likewise unproven and kept; both
    # generations stay in the registry
    assert consumer["state"] == "kept"
    view = inspect_density_registry(config.run.directory)
    assert sorted(r["generation"] for r in view["resources"]
                  if r["kind"] == "generation") == [1, 2]


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_release_reports_the_missing_read_evidence(kind, tmp_path):
    """Engine level: the kept receipt names the missing proof, and the
    producer generation stays protected in the reclaim plan."""
    owner, run_root, scratch = _layout(tmp_path)
    no_read = _fake_pwx(
        tmp_path / "noread",
        "#!/bin/bash\nset -eu\n"
        "mkdir -p tmp/pyraimd2.save\n"
        "echo fresh > tmp/pyraimd2.save/charge-density.dat\n"
        "echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml\n"
        f"cat {FIXTURE.resolve()}\n")
    engine = _engine(kind, tmp_path, script=no_read, owner=owner,
                     run_root=run_root, scratch=scratch, startpot_file=True)
    engine.compute(_si(), label="si")
    pending = engine.release_consumed_scratch(evaluation_id=0)
    assert pending["status"] == "kept"
    engine.compute(_si(), label="si")  # stages g1, succeeds, never reads it
    record = engine.last_attempt_records[-1]
    assert record["density_input_generation"] == 1
    assert "density_read_evidence" not in record
    assert "no density-read marker" in record["density_read_evidence_missing"]
    receipt = engine.release_consumed_scratch(evaluation_id=1)
    consumption = receipt["consumption"]
    assert consumption["status"] == "kept"
    assert consumption["generation"] == 1
    assert "no actual density-read evidence" in consumption["reason"]
    records = _records(owner)
    assert [r["state"] for r in records] == ["kept", "kept"]
    # the unproven producer seed is reclaim-protected
    plan = plan_density_reclaim(owner)
    by_generation = {r["generation"]: r
                     for r in plan["resources"] if r["kind"] == "generation"}
    assert by_generation[1]["decision"] == "keep"
    assert any("unconsumed producer seed" in reason
               for reason in by_generation[1]["reasons"])


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_genuine_read_releases_the_producer(kind, tmp_path):
    """Positive control: the fake reports the read exactly like QE 7.5,
    and the consumption receipt binds the raw-output evidence (marker,
    read path, input/output digests) to the released producer."""
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    first.release_consumed_scratch(evaluation_id=0)
    second = _engine(kind, tmp_path / "b", script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    second.compute(_si(), label="si")
    receipt = second.release_consumed_scratch(evaluation_id=1)
    consumption = receipt["consumption"]
    assert consumption["status"] == "cleaned"
    evidence = consumption["release"]["evidence"]
    assert evidence["proof"] == "consumed"
    read = evidence["consumed_by"]["read_evidence"]
    assert read["marker"] == "The initial density is read from file :"
    assert read["read_path"] == "./tmp/pyraimd2.save/charge-density"
    assert read["startpot"] == "file"
    # the evidence binds the consumer's own raw output (archived with the
    # attempt) and launch input by content digest
    consumer_record = next(r for r in _records(owner)
                           if r.get("released_evidence", {})
                           .get("evaluation_id") == 1)
    archived = {entry["file"]: entry["sha256"]
                for entry in consumer_record["archived"]}
    output_name = read["output"]["file"]
    assert output_name in archived
    assert read["output"]["sha256"] == archived[output_name]


@pytest.mark.parametrize("kind", ["qe", "ase"])
@pytest.mark.parametrize("field", ["seed_content_digest",
                                   "reference_fingerprint"])
def test_contradicting_pending_receipt_preserves_producer(kind, field,
                                                          tmp_path):
    """A persisted pending receipt whose recorded seed identity
    contradicts the live registry manifest (damaged or foreign) must not
    be 'repaired' into an approval: the producer is preserved with the
    contradiction named, and the seed stays protected in the plan."""
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    first.release_consumed_scratch(evaluation_id=0)
    producer_handle = first._last_scratch_handle
    pending = dict(producer_handle.load_record()["released_evidence"])
    actual = pending[field]
    pending[field] = ("0" * 64 if field == "seed_content_digest"
                      else "qe-other:0000000000000000")
    assert pending[field] != actual
    producer_handle.update_record(released_evidence=pending)

    second = _engine(kind, tmp_path / "b", script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    second.compute(_si(), label="si")  # genuine read of the unchanged g1
    receipt = second.release_consumed_scratch(evaluation_id=1)
    consumption = receipt["consumption"]
    assert consumption["status"] == "kept"
    assert consumption["generation"] == 1
    assert "contradict" in consumption["reason"] \
        or "incomplete" in consumption["reason"] \
        or "inconsistent" in consumption["reason"]
    after = producer_handle.load_record()
    assert after["state"] == "kept"
    assert after["released_evidence"]["seed_content_digest"] == \
        pending["seed_content_digest"]  # the damaged receipt is untouched
    assert Path(after["scratch_dir"]).is_dir()
    plan = plan_density_reclaim(owner)
    g1 = next(r for r in plan["resources"]
              if r.get("generation") == 1)
    assert g1["decision"] == "keep"


@pytest.mark.parametrize("kind", ["qe", "ase"])
def test_normal_read_still_releases_with_matching_identity(kind, tmp_path):
    """Companion control: with every identity agreeing (pending receipt,
    live manifest, consumer pin), the producer releases exactly as
    before."""
    owner, run_root, scratch = _layout(tmp_path)
    first = _engine(kind, tmp_path / "a", script=_success_script(tmp_path / "a"),
                    owner=owner, run_root=run_root, scratch=scratch,
                    startpot_file=True)
    first.compute(_si(), label="si")
    first.release_consumed_scratch(evaluation_id=0)
    second = _engine(kind, tmp_path / "b", script=_success_script(tmp_path / "b"),
                     owner=owner, run_root=run_root, scratch=scratch,
                     startpot_file=True)
    second.compute(_si(), label="si")
    receipt = second.release_consumed_scratch(evaluation_id=1)
    assert receipt["consumption"]["status"] == "cleaned"
    evidence = receipt["consumption"]["release"]["evidence"]
    manifest = json.loads((_registry(owner) / "g000001"
                           / "manifest.json").read_text())
    assert evidence["seed_content_digest"] == manifest["content_digest"]
    assert evidence["reference_fingerprint"] == \
        manifest["reference_fingerprint"]


@pytest.mark.parametrize("backend", ["qe", "qe-ase"])
def test_wavefunction_read_text_is_not_density_read_evidence(backend,
                                                            tmp_path):
    """A solver output carrying only a WAVEFUNCTION-read line (never QE's
    density-read marker) proves nothing for the density gate: the producer
    stays kept.  The two evidence channels never cross-authorize."""
    root = tmp_path / backend
    config_path = _qe_persist_toml(root, steps=1, extra=DENSITY_EXTRA)
    config_path.write_text(config_path.read_text().replace(
        'backend = "qe"', f'backend = "{backend}"'))
    script = root / "script" / "fake_pwx.sh"
    script.write_text(
        "#!/bin/bash\nset -eu\n"
        "mkdir -p tmp/pyraimd2.save\n"
        "echo fresh-density > tmp/pyraimd2.save/charge-density.dat\n"
        "echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml\n"
        "echo '     Reading wavefunction from file "
        "tmp/pyraimd2.save/wfc1.dat'\n"
        f"cat {FIXTURE.resolve()}\n")
    config = load_config(config_path)
    result = run_workflow(config, verbose=False, handle_sigint=False)
    assert result.steps_completed == 1
    records = _records(config.run.directory)
    # every attempt stays kept: no density-read evidence anywhere, and the
    # wavefunction line never substitutes for it
    assert all(r["state"] == "kept" for r in records)
    assert all(r["released_evidence"]["proof"]
               == "pending_independent_consumption" for r in records)
