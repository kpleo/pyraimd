"""--probe-surrogate / probe_surrogate_setup: the surrogate-only readiness
probe (a fourth validate scope alongside configuration / --check-environment
/ --probe-backends).

All backends here are synthetic sentinels registered through the real
entry-point discovery path; no real MACE/QE is touched.  The reference
sentinel records any contact and fails the test if the probe constructs or
evaluates it.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from pyraimd2.backends import registry
from pyraimd2.cli import main as cli_main
from pyraimd2.workflows.templates import HARMONIC_STRUCTURE

# ---------------------------------------------------------------------------
# sentinel backends (loaded by the registry through the entry points below)

CALLS: dict[str, int] = {
    "reference_factory": 0,
    "reference_compute": 0,
    "ok_factory": 0,
    "ok_predict": 0,
}


class SentinelReference:
    def __init__(self):
        CALLS["reference_factory"] += 1

    @property
    def fingerprint(self):
        return "sentinel-reference:v1"

    def compute(self, atoms):
        CALLS["reference_compute"] += 1
        raise AssertionError(
            "the surrogate probe must never evaluate the reference")


class ProbeOkSurrogate:
    def __init__(self):
        CALLS["ok_factory"] += 1

    @property
    def capabilities(self):
        from pyraimd2.surrogate.base import SurrogateCapabilities
        return SurrogateCapabilities(
            energy_kind="energy", force_consistent=True,
            forces_conservative=True, stress_available=False,
            uncertainty_available=False)

    @property
    def fingerprint(self):
        return "probe-ok-surrogate:v1"

    def predict(self, atoms):
        from pyraimd2.surrogate.base import SurrogatePrediction
        CALLS["ok_predict"] += 1
        n = len(atoms)
        return SurrogatePrediction(
            energy=1.5, forces=np.full((n, 3), 0.25), stress=None,
            uncertainty=np.full(n, np.nan),
            energy_kind="energy", force_consistent=True)


class ImportFailSurrogate:
    @property
    def fingerprint(self):
        return "import-fail:v1"

    def predict(self, atoms):  # pragma: no cover - never reached
        raise AssertionError("unreachable")


class PredictFailSurrogate:
    @property
    def capabilities(self):
        from pyraimd2.surrogate.base import SurrogateCapabilities
        return SurrogateCapabilities(
            energy_kind="energy", force_consistent=True,
            forces_conservative=True, stress_available=False,
            uncertainty_available=False)

    @property
    def fingerprint(self):
        return "predict-fail:v1"

    def predict(self, atoms):
        raise RuntimeError("synthetic predict failure")


class BadShapeSurrogate(PredictFailSurrogate):
    @property
    def fingerprint(self):
        return "bad-shape:v1"

    def predict(self, atoms):
        from pyraimd2.surrogate.base import SurrogatePrediction
        return SurrogatePrediction(
            energy=1.0, forces=np.full((len(atoms), 2), 0.1), stress=None,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind="energy", force_consistent=True)


class NanSurrogate(PredictFailSurrogate):
    @property
    def fingerprint(self):
        return "nan:v1"

    def predict(self, atoms):
        from pyraimd2.surrogate.base import SurrogatePrediction
        return SurrogatePrediction(
            energy=float("nan"),
            forces=np.full((len(atoms), 3), 0.1), stress=None,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind="energy", force_consistent=True)


class WeakSurrogate(ProbeOkSurrogate):
    """No conservative-E/F declarations: refused for task.mode 'mts'."""

    @property
    def capabilities(self):
        from pyraimd2.surrogate.base import SurrogateCapabilities
        return SurrogateCapabilities(
            energy_kind="energy", force_consistent=False,
            forces_conservative=False, stress_available=False,
            uncertainty_available=False)

    @property
    def fingerprint(self):
        return "weak:v1"


class KindConflictSurrogate(ProbeOkSurrogate):
    """Declares energy_kind='energy' but returns 'free_energy'."""

    @property
    def fingerprint(self):
        return "kind-conflict:v1"

    def predict(self, atoms):
        from pyraimd2.surrogate.base import SurrogatePrediction
        return SurrogatePrediction(
            energy=1.5, forces=np.full((len(atoms), 3), 0.25), stress=None,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind="free_energy", force_consistent=True)


class FcConflictSurrogate(ProbeOkSurrogate):
    """Declares force_consistent=True but returns False."""

    @property
    def fingerprint(self):
        return "fc-conflict:v1"

    def predict(self, atoms):
        from pyraimd2.surrogate.base import SurrogatePrediction
        return SurrogatePrediction(
            energy=1.5, forces=np.full((len(atoms), 3), 0.25), stress=None,
            uncertainty=np.full(len(atoms), np.nan),
            energy_kind="energy", force_consistent=False)


class MetadataFailSurrogate(ProbeOkSurrogate):
    """Fingerprint metadata read raises (constructed successfully)."""

    @property
    def fingerprint(self):
        raise OSError("synthetic fingerprint metadata unavailable")


class UnknownKindSurrogate(ProbeOkSurrogate):
    """Declares energy_kind unknown; the returned 'energy' is compatible."""

    @property
    def capabilities(self):
        from pyraimd2.surrogate.base import SurrogateCapabilities
        return SurrogateCapabilities(
            energy_kind="unknown", force_consistent=True,
            forces_conservative=True, stress_available=False,
            uncertainty_available=False)

    @property
    def fingerprint(self):
        return "unknown-kind:v1"


class PlainResultSurrogate(ProbeOkSurrogate):
    """Predict returns an object without energy_kind/force_consistent
    attributes — undeclared return metadata is compatible."""

    @property
    def fingerprint(self):
        return "plain-result:v1"

    def predict(self, atoms):
        from types import SimpleNamespace
        return SimpleNamespace(energy=0.5,
                               forces=np.zeros((len(atoms), 3)))


class UndeclaredFcSurrogate(ProbeOkSurrogate):
    """Declares force_consistent=None (unknown); a returned True is
    compatible (the MTS None rule)."""

    @property
    def capabilities(self):
        from pyraimd2.surrogate.base import SurrogateCapabilities
        return SurrogateCapabilities(
            energy_kind="energy", force_consistent=None,
            forces_conservative=True, stress_available=False,
            uncertainty_available=False)

    @property
    def fingerprint(self):
        return "undeclared-fc:v1"


def sentinel_reference_factory():
    return SentinelReference()


def ok_surrogate_factory():
    return ProbeOkSurrogate()


def importfail_surrogate_factory():
    raise ImportError("No module named 'definitely_missing_dep'")


def predictfail_surrogate_factory():
    return PredictFailSurrogate()


def badshape_surrogate_factory():
    return BadShapeSurrogate()


def nan_surrogate_factory():
    return NanSurrogate()


def weak_surrogate_factory():
    return WeakSurrogate()


def kindconflict_surrogate_factory():
    return KindConflictSurrogate()


def fcconflict_surrogate_factory():
    return FcConflictSurrogate()


def metafail_surrogate_factory():
    return MetadataFailSurrogate()


def unknownkind_surrogate_factory():
    return UnknownKindSurrogate()


def plainresult_surrogate_factory():
    return PlainResultSurrogate()


def undeclaredfc_surrogate_factory():
    return UndeclaredFcSurrogate()


_SENTINELS = {
    "sentinel-reference": "sentinel_reference_factory",
    "probe-ok-surrogate": "ok_surrogate_factory",
    "probe-importfail-surrogate": "importfail_surrogate_factory",
    "probe-predictfail-surrogate": "predictfail_surrogate_factory",
    "probe-badshape-surrogate": "badshape_surrogate_factory",
    "probe-nan-surrogate": "nan_surrogate_factory",
    "probe-weak-surrogate": "weak_surrogate_factory",
    "probe-kindconflict-surrogate": "kindconflict_surrogate_factory",
    "probe-fcconflict-surrogate": "fcconflict_surrogate_factory",
    "probe-metafail-surrogate": "metafail_surrogate_factory",
    "probe-unknownkind-surrogate": "unknownkind_surrogate_factory",
    "probe-plainresult-surrogate": "plainresult_surrogate_factory",
    "probe-undeclaredfc-surrogate": "undeclaredfc_surrogate_factory",
}


@pytest.fixture()
def sentinels(monkeypatch):
    module = __name__
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    eps = [
        importlib.metadata.EntryPoint(
            name=name, value=f"{module}:{attr}", group="pyraimd2.backends")
        for name, attr in _SENTINELS.items()
    ]
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: eps if group == "pyraimd2.backends" else [],
    )
    registry._reset_entry_point_cache()
    for key in CALLS:
        CALLS[key] = 0
    yield
    registry._reset_entry_point_cache()


# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, surrogate_block: str,
                  *, reference: bool = True, mode: str | None = None,
                  extra_dynamics: str = "") -> Path:
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    if mode is None:
        # 'surrogate' mode forbids [reference]; MTS takes both sections
        mode = "mts" if reference else "surrogate"
    if mode == "mts":
        extra_dynamics = ('integrator = "respa"\nouter_ratio = 4\n'
                          + extra_dynamics)
    steps = 8 if mode == "mts" else 2
    reference_block = ('[reference]\nbackend = "sentinel-reference"\n'
                       if reference else "")
    path = tmp_path / "run.toml"
    path.write_text(f"""schema_version = 1
[run]
id = "probe-test"
directory = "run"
[task]
kind = "md"
mode = "{mode}"
[structure]
file = "structure.extxyz"
[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = {steps}
{extra_dynamics}
{reference_block}
{surrogate_block}
""")
    return path


def _run_json(capsys, *argv: str):
    code = cli_main(list(argv))
    out = capsys.readouterr()
    return code, json.loads(out.out), out.err


# ---------------------------------------------------------------------------
# success path


def test_probe_surrogate_success_json(sentinels, tmp_path, capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-ok-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["validation_scope"] == "surrogate_probe"
    assert report["readiness"] == "ready"
    assert report["configuration_valid"] is True
    assert report["checked_backends"] == ["surrogate"]
    assert report["unchecked_backends"] == ["reference"]
    assert report["reference_evaluations"] == 0
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 1
    assert report["surrogate"]["backend"] == "probe-ok-surrogate"
    assert report["surrogate"]["fingerprint"] == "probe-ok-surrogate:v1"
    assert report["surrogate"]["energy_kind"] == "energy"
    assert report["surrogate"]["force_consistent"] is True
    assert report["probe"]["energy_eV"] == 1.5
    assert report["probe"]["forces_shape"] == [3, 3]
    assert report["probe"]["forces_norm_eV_A"] == pytest.approx(
        float(np.linalg.norm(np.full((3, 3), 0.25))))
    assert report["probe"]["predict_s"] >= 0.0
    assert report["probe"]["total_s"] >= report["probe"]["predict_s"]
    # the reference was never constructed or evaluated
    assert CALLS["reference_factory"] == 0
    assert CALLS["reference_compute"] == 0
    assert CALLS["ok_predict"] == 1
    # no run artifacts
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob("**/events.jsonl"))
    assert not list(tmp_path.glob("**/trajectory.db"))


def test_probe_surrogate_success_text(sentinels, tmp_path, capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-ok-surrogate"\n')
    code = cli_main(["validate", str(config), "--probe-surrogate"])
    out = capsys.readouterr().out
    assert code == 0
    assert "surrogate     : probe-ok-surrogate" in out
    assert "probe energy  : 1.500000 eV" in out
    assert "reference" in out and "not constructed" in out
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_python_api(sentinels, tmp_path) -> None:
    from pyraimd2.config import load_config
    from pyraimd2.workflows import probe_surrogate_setup
    config = load_config(_write_config(
        tmp_path, '[surrogate]\nbackend = "probe-ok-surrogate"\n'))
    report = probe_surrogate_setup(config)
    assert report["readiness"] == "ready"
    assert report["prediction_successes"] == 1
    assert CALLS["reference_factory"] == 0


# ---------------------------------------------------------------------------
# the five failure paths: counts, exit codes, JSON


def test_probe_surrogate_missing_dependency(sentinels, tmp_path,
                                            capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-importfail-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["readiness"] == "blocked"
    assert report["error"]["stage"] == "initialize"
    assert "definitely_missing_dep" in report["error"]["message"]
    assert report["prediction_attempts"] == 0
    assert report["prediction_successes"] == 0
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_predict_raises(sentinels, tmp_path, capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-predictfail-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "predict"
    assert "synthetic predict failure" in report["error"]["message"]
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 0


def test_probe_surrogate_shape_error(sentinels, tmp_path, capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-badshape-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "contract"
    assert "shape" in report["error"]["message"]
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 0


def test_probe_surrogate_non_finite(sentinels, tmp_path, capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-nan-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "contract"
    assert "non-finite" in report["error"]["message"]
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 0


def test_probe_surrogate_no_surrogate_section(sentinels, tmp_path,
                                              capsys) -> None:
    config = _write_config(tmp_path, "", reference=True,
                           mode="reference")
    code = cli_main(["validate", str(config), "--probe-surrogate"])
    err = capsys.readouterr().err
    assert code == 2
    assert "no [surrogate] section" in err
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_missing_structure(sentinels, tmp_path,
                                           capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-ok-surrogate"\n')
    (tmp_path / "structure.extxyz").unlink()
    code = cli_main(["validate", str(config), "--probe-surrogate"])
    err = capsys.readouterr().err
    assert code == 2
    assert "structure.file not found" in err
    assert CALLS["ok_factory"] == 0


# ---------------------------------------------------------------------------
# scaled wrapper composition: one inner call, correctly scaled E/F


def test_probe_surrogate_scaled_counts_once(sentinels, tmp_path,
                                            capsys) -> None:
    block = ('[surrogate]\nbackend = "scaled"\nscale = 1.25\n'
             'base = { name = "probe-ok-surrogate", kwargs = {} }\n')
    config = _write_config(tmp_path, block)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["surrogate"]["backend"] == "scaled"
    assert report["surrogate"]["fingerprint"].startswith("scaled:")
    # the base was constructed/evaluated exactly once; E and F scaled
    assert CALLS["ok_factory"] == 1
    assert CALLS["ok_predict"] == 1
    assert report["probe"]["energy_eV"] == pytest.approx(1.25 * 1.5)
    assert report["probe"]["forces_norm_eV_A"] == pytest.approx(
        1.25 * float(np.linalg.norm(np.full((3, 3), 0.25))))
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 1
    assert CALLS["reference_factory"] == 0


# ---------------------------------------------------------------------------
# MTS declaration gate


def test_probe_surrogate_mts_requires_conservative_declarations(
        sentinels, tmp_path, capsys) -> None:
    block = '[surrogate]\nbackend = "probe-weak-surrogate"\n'
    config = _write_config(tmp_path, block, mode="mts")
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "initialize"
    assert "force_consistent" in report["error"]["message"]
    assert report["prediction_attempts"] == 0
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_mts_ok_with_declarations(sentinels, tmp_path,
                                                  capsys) -> None:
    block = '[surrogate]\nbackend = "probe-ok-surrogate"\n'
    config = _write_config(tmp_path, block, mode="mts")
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["readiness"] == "ready"
    assert report["prediction_successes"] == 1
    assert CALLS["reference_factory"] == 0


# ---------------------------------------------------------------------------
# MACE local-weights rule (never imports mace; refused before construction)


def test_probe_surrogate_mace_alias_refused(sentinels, tmp_path,
                                            capsys) -> None:
    block = '[surrogate]\nbackend = "mace"\nmodel = "small"\n'
    config = _write_config(tmp_path, block)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "initialize"
    assert "model" in report["error"]["message"]
    assert report["prediction_attempts"] == 0
    import sys
    assert "mace" not in sys.modules


def test_probe_surrogate_mace_nested_alias_refused(sentinels, tmp_path,
                                                   capsys) -> None:
    block = ('[surrogate]\nbackend = "scaled"\nscale = 1.25\n'
             'base = { name = "mace", kwargs = { model = "medium" } }\n')
    config = _write_config(tmp_path, block)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "initialize"
    assert "scaled.base" in report["error"]["message"]
    assert report["prediction_attempts"] == 0


def test_probe_surrogate_mace_missing_file_refused(sentinels, tmp_path,
                                                   capsys) -> None:
    block = ('[surrogate]\nbackend = "mace"\n'
             'model = "weights/absent.model"\n')
    config = _write_config(tmp_path, block)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["error"]["stage"] == "initialize"
    assert "not an existing local file" in report["error"]["message"]


# ---------------------------------------------------------------------------
# returned-vs-declared contract (the MTS UNKNOWN/None-compatible rule)


def test_probe_surrogate_kind_conflict_blocked(sentinels, tmp_path,
                                               capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-kindconflict-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["readiness"] == "blocked"
    assert report["error"]["stage"] == "contract"
    assert "free_energy" in report["error"]["message"]
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 0
    assert report["reference_evaluations"] == 0
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_fc_conflict_blocked(sentinels, tmp_path,
                                             capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-fcconflict-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 1
    assert report["readiness"] == "blocked"
    assert report["error"]["stage"] == "contract"
    assert "force_consistent" in report["error"]["message"]
    assert report["prediction_attempts"] == 1
    assert report["prediction_successes"] == 0


def test_probe_surrogate_metadata_read_failure_structured(
        sentinels, tmp_path, capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-metafail-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    # a single blocked JSON object, no escaping traceback
    assert code == 1
    assert report["readiness"] == "blocked"
    assert report["error"]["stage"] == "initialize"
    assert "fingerprint" in report["error"]["message"]
    assert report["prediction_attempts"] == 0
    assert report["prediction_successes"] == 0
    assert report["reference_evaluations"] == 0
    assert CALLS["reference_factory"] == 0


def test_probe_surrogate_unknown_declared_kind_compatible(
        sentinels, tmp_path, capsys) -> None:
    # non-MTS config: an unknown declared kind is compatible with a known
    # returned kind (MTS mode would require the declaration up front)
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-unknownkind-surrogate"\n',
        reference=False)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["readiness"] == "ready"
    assert report["surrogate"]["energy_kind"] == "unknown"
    assert report["probe"]["returned_contract"]["energy_kind"] == "energy"
    assert report["prediction_successes"] == 1


def test_probe_surrogate_result_without_contract_fields_compatible(
        sentinels, tmp_path, capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-plainresult-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["readiness"] == "ready"
    assert report["probe"]["returned_contract"]["energy_kind"] is None
    assert report["probe"]["returned_contract"]["force_consistent"] is None
    assert report["prediction_successes"] == 1


def test_probe_surrogate_undeclared_fc_compatible(sentinels, tmp_path,
                                                  capsys) -> None:
    config = _write_config(
        tmp_path, '[surrogate]\nbackend = "probe-undeclaredfc-surrogate"\n',
        reference=False)
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["readiness"] == "ready"
    assert report["surrogate"]["force_consistent"] is None
    assert report["probe"]["returned_contract"]["force_consistent"] is True
    assert report["unchecked_backends"] == []


def test_probe_surrogate_success_reports_declared_and_returned(
        sentinels, tmp_path, capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-ok-surrogate"\n')
    code, report, _ = _run_json(capsys, "validate", str(config),
                                "--probe-surrogate", "--json")
    assert code == 0
    assert report["probe"]["declared_contract"] == {
        "energy_kind": "energy", "force_consistent": True}
    assert report["probe"]["returned_contract"] == {
        "energy_kind": "energy", "force_consistent": True}


# ---------------------------------------------------------------------------
# mutual exclusion, rejected before any factory/predict


def test_probe_surrogate_mutually_exclusive(sentinels, tmp_path,
                                            capsys) -> None:
    config = _write_config(tmp_path,
                           '[surrogate]\nbackend = "probe-ok-surrogate"\n')
    for extra in (("--probe-backends",), ("--check-environment",)):
        code = cli_main(["validate", str(config), "--probe-surrogate",
                         *extra])
        err = capsys.readouterr().err
        assert code == 2
        assert "mutually exclusive" in err
        code, report, _ = _run_json(
            capsys, "validate", str(config), "--probe-surrogate", *extra,
            "--json")
        assert code == 2
        assert "mutually exclusive" in report["error"]["message"]
    # three-way conflict is refused the same way
    code = cli_main(["validate", str(config), "--probe-surrogate",
                     "--probe-backends", "--check-environment"])
    assert code == 2
    capsys.readouterr()
    # nothing was constructed or predicted in any conflict
    assert CALLS["ok_factory"] == 0
    assert CALLS["ok_predict"] == 0
    assert CALLS["reference_factory"] == 0
