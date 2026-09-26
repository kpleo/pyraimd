"""Surrogate-only readiness probe: ``pyramid validate CONFIG --probe-surrogate``.

The existing validation layers answer different questions:

- ``pyramid validate CONFIG`` is static schema/structure/capability
  validation; nothing is evaluated.
- ``--check-environment`` is strictly read-only local metadata.
- ``--probe-backends`` evaluates the structure once with EVERY configured
  backend, reference first — for an expensive reference (DFT) that means
  paying the reference cost before the machine-learning surrogate's
  dependencies or weights are exercised at all.

This module adds the missing fourth entry point: construct ONLY the
configured surrogate (through the same registry factories, including
correction wrappers such as ``scaled`` around their declared base), run
one real prediction on the configured structure, and report whether the
selected surrogate is ready.  The reference backend is never constructed
and no reference compute or external program runs; ``reference_evaluations``
is always 0.  Nothing is written: no run directory, event log, trajectory,
checkpoint or resumable state, and the structure is never modified (no
thermalization, no q/p changes).

Readiness here is the SELECTED surrogate's status, not the whole
configuration: the reference backend is unchecked, and the cross-backend
energy-contract certification a full run performs is NOT made here.

This is a real model call, not a static check: model weights load and one
forward evaluation runs, so run it on a node authorized for compute.  For
the MACE backend the probe requires an explicit LOCAL weights file (also
inside a wrapper's nested base spec): a bare base-model name such as
``small``/``medium`` is refused before construction, because it could turn
into a network download inside this entry point.  (The ordinary run
interfaces keep their existing semantics; this stricter rule is local to
the probe.)

Report schema (``schema_version = 1``, independent of the configuration
schema): ``validation_scope`` is ``"surrogate_probe"``; ``readiness`` is
``"ready"`` or ``"blocked"``; ``prediction_attempts`` /
``prediction_successes`` count the single metered prediction (an attempt
is recorded once the backend is constructed and the predict call is
entered; a construction failure records 0 attempts).  No automatic retry
and no fallback to the reference side exist.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pyraimd2.backends.registry import backend_capabilities
from pyraimd2.config import PyramidConfig
from pyraimd2.runtime.identity import fingerprint_of
from pyraimd2.workflows.setup import (
    _OPTIONAL_EXTRAS,
    WorkflowError,
    backend_path_problems,
    create_configured_backend,
    load_structure,
)

SCHEMA_VERSION = 1

#: stages in execution order; the failure record names exactly one
STAGES = ("schema", "structure", "initialize", "predict", "contract")

#: stages whose failures are static (usage/configuration) problems rather
#: than run-time faults — the CLI maps these to its usage exit code
STATIC_STAGES = ("schema", "structure")


class SurrogateProbeError(WorkflowError):
    """A surrogate-probe failure with the stage and the partial report.

    ``stage`` is one of :data:`STAGES`; ``report`` is the probe report as
    far as it could be built (attempt/success counters included), so a
    JSON caller emits exactly one object even on failure.
    """

    def __init__(self, stage: str, message: str, report: dict) -> None:
        super().__init__(message)
        self.stage = stage
        self.report = report


def _base_report(config: PyramidConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_scope": "surrogate_probe",
        "configuration_valid": False,
        "readiness": "blocked",
        "checked_backends": [],
        "unchecked_backends": (
            ["reference"] if config.reference is not None else []),
        "reference_evaluations": 0,
        "prediction_attempts": 0,
        "prediction_successes": 0,
    }


def _fail(report: dict, stage: str, message: str) -> SurrogateProbeError:
    report["error"] = {"code": "SurrogateProbeError", "stage": stage,
                       "message": message}
    return SurrogateProbeError(stage, message, report)


def _mace_local_model_problems(name: str, options: dict[str, Any],
                               *, chain: tuple[str, ...]) -> list[str]:
    """Every ``mace`` in the (possibly wrapped) surrogate chain must name an
    existing local weights file — a bare alias could download inside this
    entry point.  Nested wrapper specs are followed through ``base``."""
    problems: list[str] = []
    dotted = "surrogate" + "".join(f".{c}.base" for c in chain)
    if name == "mace":
        model = options.get("model")
        if model is None:
            problems.append(
                f"{dotted}.model: no local weights file configured — the "
                "surrogate probe refuses the base-model default ('small'), "
                "which could download; set model to an existing local file")
        elif not Path(str(model)).is_file():
            problems.append(
                f"{dotted}.model: not an existing local file: {model!r} — "
                "the surrogate probe never downloads weights; prepare the "
                "file and point model at it")
        return problems
    base = options.get("base")
    if isinstance(base, dict) and isinstance(base.get("name"), str):
        problems.extend(_mace_local_model_problems(
            base["name"], base.get("kwargs", {}) or {},
            chain=chain + (name,)))
    return problems


def probe_surrogate_setup(config: PyramidConfig) -> dict:
    """Construct only the configured surrogate and predict once on the
    configured structure.

    Returns the readiness report on success; raises
    :class:`SurrogateProbeError` (a WorkflowError) with ``.stage`` and the
    partial ``.report`` on failure.  The reference backend is never
    constructed; no run artifacts are written and the structure is used
    as read.
    """
    total_start = time.perf_counter()
    report = _base_report(config)

    # -- schema: the same task-kind gate as full validation ---------------
    if config.task.kind not in ("singlepoint", "relax", "md"):
        raise _fail(report, "schema",
                    f"task.kind {config.task.kind!r} is not supported; "
                    "choose singlepoint, relax or md")
    if config.surrogate is None:
        raise _fail(report, "schema",
                    "no [surrogate] section configured — the surrogate "
                    "probe tests the surrogate only; there is nothing to "
                    "probe (a reference-only configuration is covered by "
                    "`pyramid validate CONFIG --probe-backends`)")

    # -- structure: always required (a probe evaluates it once) -----------
    try:
        atoms = load_structure(config)
    except WorkflowError as error:
        raise _fail(report, "structure", str(error)) from error
    report["structure"] = {
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "pbc": [bool(p) for p in atoms.pbc],
        "has_momenta": "momenta" in atoms.arrays,
    }
    report["configuration_valid"] = True
    report["checked_backends"] = ["surrogate"]

    # -- initialize: selected-section path checks, the local-weights rule,
    #    then construction through the registry (wrappers included) -------
    surrogate_config = config.surrogate
    problems = backend_path_problems("surrogate", surrogate_config)
    problems.extend(_mace_local_model_problems(
        surrogate_config.name, surrogate_config.options, chain=()))
    if problems:
        raise _fail(report, "initialize",
                    "invalid surrogate options:\n  - " + "\n  - ".join(problems))
    try:
        surrogate = create_configured_backend("surrogate", surrogate_config)
    except WorkflowError as error:
        raise _fail(report, "initialize", str(error)) from error
    except ImportError as error:
        extra = _OPTIONAL_EXTRAS.get(surrogate_config.name)
        hint = (f"install it with `pip install 'pyraimd2[{extra}]'`"
                if extra else "install the backend's package")
        raise _fail(report, "initialize",
                    f"surrogate backend {surrogate_config.name!r} needs an "
                    f"optional dependency that is not installed ({error}); "
                    f"{hint}, or choose another backend") from error
    except Exception as error:
        raise _fail(report, "initialize",
                    f"surrogate backend {surrogate_config.name!r} could not "
                    f"be constructed: {error!r}") from error

    caps = backend_capabilities(surrogate)
    fingerprint = fingerprint_of(surrogate)
    report["surrogate"] = {
        "backend": surrogate_config.name,
        "fingerprint": None if fingerprint is None else str(fingerprint),
        "energy_kind": caps.energy_kind.value,
        "force_consistent": caps.force_consistent,
        "forces_conservative": caps.forces_conservative,
        "stress_available": caps.stress_available,
        "uncertainty_available": getattr(caps, "uncertainty_available",
                                         False),
    }

    if config.task.mode == "mts":
        # the fixed-model MTS path needs the conservative E/F contract
        # declared (and a content fingerprint) on the surrogate side
        declarations = []
        if caps.force_consistent is not True:
            declarations.append("force_consistent=True")
        if caps.forces_conservative is not True:
            declarations.append("forces_conservative=True")
        if caps.energy_kind.value == "unknown":
            declarations.append("a known energy_kind")
        if fingerprint is None:
            declarations.append("a fingerprint (content identity)")
        if declarations:
            raise _fail(report, "initialize",
                        "task.mode 'mts' requires the surrogate to declare "
                        + " and ".join(declarations)
                        + " explicitly; the probed backend does not, so the "
                        "MTS run would refuse it before any evaluation")

    # -- predict: the one metered real call --------------------------------
    report["prediction_attempts"] = 1
    try:
        predict_start = time.perf_counter()
        result = surrogate.predict(atoms)
        predict_s = time.perf_counter() - predict_start
    except ImportError as error:
        extra = _OPTIONAL_EXTRAS.get(surrogate_config.name)
        hint = (f"install it with `pip install 'pyraimd2[{extra}]'`"
                if extra else "install the backend's package")
        raise _fail(report, "predict",
                    f"surrogate backend {surrogate_config.name!r} needs an "
                    f"optional dependency that is not installed ({error}); "
                    f"{hint}, or choose another backend") from error
    except Exception as error:
        raise _fail(report, "predict",
                    f"surrogate self-check failed: {error}; the backend "
                    "could not evaluate the structure — check its "
                    "installation, weights file and options") from error

    # -- contract: finite scalar energy, exactly-(N,3) finite forces -------
    energy = getattr(result, "energy", None)
    try:
        energy = None if energy is None else float(energy)
    except (TypeError, ValueError):
        energy = None
    if energy is None or not np.isfinite(energy):
        raise _fail(report, "contract",
                    "the surrogate returned a missing or non-finite energy "
                    "on the structure; check the backend configuration "
                    "before running")
    raw_forces = getattr(result, "forces", None)
    try:
        forces = (np.empty((0, 0)) if raw_forces is None
                  else np.asarray(raw_forces, dtype=float))
    except (TypeError, ValueError):
        forces = np.empty((0, 0))
    if forces.shape != (len(atoms), 3):
        raise _fail(report, "contract",
                    f"the surrogate returned forces with shape "
                    f"{forces.shape}, expected ({len(atoms)}, 3)")
    if not np.isfinite(forces).all():
        raise _fail(report, "contract",
                    "the surrogate returned non-finite forces on the "
                    "structure; check the backend configuration before "
                    "running")

    report["prediction_successes"] = 1
    report["readiness"] = "ready"
    report["probe"] = {
        "energy_eV": energy,
        "forces_shape": [len(atoms), 3],
        "forces_norm_eV_A": float(np.linalg.norm(forces)),
        "predict_s": predict_s,
        "total_s": time.perf_counter() - total_start,
    }
    report["scope_note"] = (
        "readiness is the SELECTED surrogate's status only: the reference "
        "backend was not constructed and no reference evaluation ran; the "
        "cross-backend energy-contract certification and the full MD "
        "readiness are NOT established by this probe.  total_s includes "
        "backend initialization and model loading — it is not a pure "
        "forward-pass time.  This probe performs a real model call; run it "
        "on a node authorized for compute.")
    return report
