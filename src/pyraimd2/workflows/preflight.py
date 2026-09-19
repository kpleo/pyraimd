"""Read-only local environment preflight for ``pyramid validate``.

Static, side-effect-free dependency checks: nothing is executed, no
backend is started, no ``--version`` subprocess runs, no model weights
are loaded and no download endpoint is contacted.  Optional packages
are probed with :mod:`importlib.util.find_spec` (no import of
torch/MACE/PySCF); launch commands are resolved with ``shlex`` /
``shutil.which`` / filesystem access checks only.

The report schema is versioned independently of the configuration
schema (``schema_version = 1``):

- ``validation_scope``: "configuration" | "environment" | "probe"
- ``configuration_valid``: bool
- ``readiness``: "not_checked" | "ready" | "blocked" | "unverified"
- ``checks``: [{id, role, status, message, remedy?}] with
  status in {"pass", "fail", "unverified"}

Ready means the local prerequisites this version knows how to check
are present; it is NOT an SCF or numerical-accuracy verification.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from pyraimd2.config import PyramidConfig

SCHEMA_VERSION = 1

_SHELL_META = set("|&;<>$`()~*?[]{}#!\\\n\"'")
_LAUNCHER_TOKENS = {"srun", "mpirun", "mpiexec", "mpirun.mpich",
                    "mpiexec.mpich", "orterun", "launcher"}


def _is_wrapper_command(tokens: list[str]) -> bool:
    if len(tokens) > 1:
        return True
    return any(any(c in _SHELL_META for c in tok) for tok in tokens)


def _resolve_executable(token: str) -> tuple[bool, str]:
    """(found_and_executable, detail) for one simple command token."""
    if os.sep in token or (os.altsep and os.altsep in token):
        path = Path(token).expanduser()
        if not path.exists():
            return False, f"no such file: {path}"
        if not path.is_file():
            return False, f"not a file: {path}"
        if not os.access(path, os.X_OK):
            return False, f"not executable: {path}"
        return True, str(path)
    found = shutil.which(token)
    if found is None:
        return False, f"not found on PATH: {token!r}"
    return True, found


def _check_qe_launcher(section: str, options: dict[str, Any]) -> dict:
    raw = options.get("pw_cmd") or ("pw.x",)
    if isinstance(raw, str):
        if os.sep in raw or Path(raw).exists():
            tokens = [raw]  # one literal path token (spaces allowed)
        else:
            tokens = shlex.split(raw)  # command-line form
    else:
        tokens = [str(t) for t in raw]
    if not tokens:
        return {"id": f"{section}.pw_cmd", "role": section,
                "status": "unverified",
                "message": "pw_cmd is empty; cannot determine the solver",
                "remedy": "set [reference] pw_cmd to the pw.x path or a "
                          "launcher command"}
    if _is_wrapper_command(tokens):
        first = tokens[0]
        layer = ("a launcher" if Path(first).name in _LAUNCHER_TOKENS
                 else "a composed command")
        return {"id": f"{section}.pw_cmd", "role": section,
                "status": "unverified",
                "message": (
                    f"pw_cmd invokes {layer} ({first!r}); the actual "
                    "solver layer cannot be confirmed statically — only "
                    "the first token was identified"),
                "remedy": "verify the solver in the real execution "
                          "environment (e.g. `srun pw.x` there), or point "
                          "pw_cmd directly at a pw.x executable"}
    ok, detail = _resolve_executable(tokens[0])
    if ok:
        return {"id": f"{section}.pw_cmd", "role": section, "status": "pass",
                "message": f"pw_cmd executable resolved: {detail} "
                           "(never executed)"}
    return {"id": f"{section}.pw_cmd", "role": section, "status": "fail",
            "message": f"pw_cmd executable {detail}",
            "remedy": "install Quantum ESPRESSO, or set [reference] "
                      "pw_cmd to the pw.x path"}


def _check_pseudo_files(section: str, options: dict[str, Any]) -> dict:
    pseudo_dir = options.get("pseudo_dir")
    pseudos = options.get("pseudos") or {}
    missing = []
    if pseudo_dir and pseudos:
        base = Path(pseudo_dir)
        for species, name in pseudos.items():
            if not (base / name).is_file():
                missing.append(f"{species}:{name}")
    if missing:
        return {"id": f"{section}.pseudopotentials", "role": section,
                "status": "fail",
                "message": "pseudopotential file(s) not found: "
                           + ", ".join(missing),
                "remedy": "fix pseudo_dir/pseudos in the configuration"}
    return {"id": f"{section}.pseudopotentials", "role": section,
            "status": "pass",
            "message": f"{len(pseudos)} pseudopotential file(s) present "
                       "under the resolved pseudo_dir"}


_find_spec = importlib.util.find_spec


def _check_optional_package(section: str, package: str, pip_hint: str,
                            extra: str = "") -> dict:
    if _find_spec(package) is None:
        return {"id": f"{section}.{package}", "role": section,
                "status": "fail",
                "message": f"optional package {package!r} is not "
                           f"installed{extra}",
                "remedy": f"install it in this environment ({pip_hint}) "
                          "or choose a backend without it"}
    return {"id": f"{section}.{package}", "role": section, "status": "pass",
            "message": f"optional package {package!r} is importable "
                       "(not imported, not evaluated)"}


def _check_model_file(section: str, options: dict[str, Any]) -> dict:
    model = options.get("model")
    if model is None:
        return {"id": f"{section}.model", "role": section,
                "status": "unverified",
                "message": "no local model configured; a base-model name "
                           "cannot be confirmed without loading",
                "remedy": "prepare a local model file and set model to "
                          "its path"}
    text = str(model)
    looks_like_path = os.sep in text or Path(text).suffix in (
        ".model", ".pt", ".pth", ".ckpt", ".json") or Path(text).exists()
    if looks_like_path:
        path = Path(text).expanduser()
        if path.is_file():
            return {"id": f"{section}.model", "role": section,
                    "status": "pass",
                    "message": f"local model file present: {path.name} "
                               "(weights not loaded)"}
        return {"id": f"{section}.model", "role": section, "status": "fail",
                "message": f"local model file not found: {path}",
                "remedy": "set model to an existing local model file"}
    return {"id": f"{section}.model", "role": section, "status": "unverified",
            "message": f"model {text!r} is a base-model name; its local "
                       "cache cannot be confirmed without loading weights",
            "remedy": "download/prepare the model explicitly and point "
                      "model at the local file"}


_CORE_BACKENDS = {"harmonic-reference", "harmonic-surrogate",
                  "scaled", "quadratic-corrected"}
_QE_BACKENDS = {"qe", "qe-ase"}


def check_environment(config: PyramidConfig,
                      setup_report: dict | None = None) -> dict:
    """Static environment preflight; never executes or downloads anything.

    ``setup_report`` is the already-computed configuration validation
    report (structure/capabilities); its serializable summaries are
    embedded unchanged.
    """
    checks: list[dict] = []
    for section in ("reference", "surrogate"):
        backend = getattr(config, section)
        if backend is None:
            continue
        name, options = backend.name, backend.options
        if name in _CORE_BACKENDS:
            checks.append({"id": f"{section}.{name}", "role": section,
                           "status": "pass",
                           "message": f"backend {name!r} runs in the core "
                                      "environment (no optional "
                                      "dependencies)"})
        elif name in _QE_BACKENDS:
            checks.append(_check_pseudo_files(section, options))
            checks.append(_check_qe_launcher(section, options))
        elif name == "mace":
            checks.append(_check_optional_package(
                section, "mace", "pip install mace-torch"))
            checks.append(_check_model_file(section, options))
        elif name == "pyscf":
            checks.append(_check_optional_package(
                section, "pyscf", "pip install pyscf"))
        else:
            checks.append({"id": f"{section}.{name}", "role": section,
                           "status": "unverified",
                           "message": f"backend {name!r} is not a builtin; "
                                      "no static environment information "
                                      "is available for it",
                           "remedy": "verify its runtime prerequisites in "
                                     "the execution environment"})

    if any(c["status"] == "fail" for c in checks):
        readiness = "blocked"
    elif any(c["status"] == "unverified" for c in checks):
        readiness = "unverified"
    else:
        readiness = "ready"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validation_scope": "environment",
        "configuration_valid": True,
        "readiness": readiness,
        "checks": checks,
    }
    if setup_report:
        if "structure" in setup_report:
            report["structure"] = setup_report["structure"]
        if "capabilities" in setup_report:
            report["capabilities"] = setup_report["capabilities"]
    return report
