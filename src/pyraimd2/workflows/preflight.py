"""Read-only local environment preflight for ``pyramid validate``.

Static, side-effect-free dependency checks: nothing is executed, no
backend is started, no ``--version`` subprocess runs, no model weights
are loaded and no download endpoint is contacted.  Optional packages
are probed with :mod:`importlib.util.find_spec` (no import of
torch/MACE/PySCF); launch commands go through the one shared
``pw_cmd`` argv contract (:func:`pyraimd2.config.normalize_pw_cmd`) and
are resolved with filesystem/PATH checks only.

The report schema is versioned independently of the configuration
schema (``schema_version = 1``):

- ``validation_scope``: "configuration" | "environment" | "probe"
- ``configuration_valid``: bool
- ``readiness``: "not_checked" | "ready" | "blocked" | "unverified"
- ``checks``: [{id, role, status, message, remedy?}] with
  status in {"pass", "fail", "unverified"}

Ready means the local prerequisites this version knows how to check
are present; it is NOT an SCF or numerical-accuracy verification.
Wrapper backends (``scaled`` / ``quadratic-corrected``) are checked
through their declared base backend, never by the outer name alone;
backends outside the builtin set stay ``unverified`` — an unknown
plugin never upgrades to ready.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from pyraimd2.config import ConfigError, PyramidConfig, normalize_pw_cmd

SCHEMA_VERSION = 1

_LAUNCHER_TOKENS = {"srun", "mpirun", "mpiexec", "mpirun.mpich",
                    "mpiexec.mpich", "orterun", "launcher"}
_SHELL_TOKENS = {"sh", "bash", "zsh", "dash", "csh", "tcsh", "ksh",
                 "fish"}
_OPERATOR_TOKENS = {"|", "||", "&", "&&", ";", "<", ">", ">>",
                    "2>", "2>&1"}


def _resolve_executable(token: str) -> tuple[bool, str]:
    """(found_and_executable, detail) for one simple command token."""
    if os.sep in token or (os.altsep and os.altsep in token):
        path = Path(token)  # literal argv: no tilde expansion
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


def _classify_tokens(tokens: tuple[str, ...]) -> str:
    """direct_argv | launcher | shell_expression, on normalized argv.

    Literal argv words are never shell syntax: parentheses or spaces in
    an existing path are just its name.  A launcher first token
    (mpirun/srun/...) hides the solver layer behind it.  Operator tokens
    or backtick/$(...) fragments only make sense under a shell that the
    executors never start.
    """
    if any(t in _OPERATOR_TOKENS for t in tokens) or any(
            "`" in t or "$(" in t for t in tokens):
        return "shell_expression"
    # an explicit shell wrapper (sh -c "...") hides the actual solver
    # behind a shell that will run it — never confirmed statically
    if Path(tokens[0]).name in _SHELL_TOKENS and any(
            t in ("-c", "-lc") for t in tokens[1:3]):
        return "shell_expression"
    if Path(tokens[0]).name in _LAUNCHER_TOKENS:
        return "launcher"
    return "direct_argv"


def _check_command_tokens(section: str, tokens: tuple[str, ...], *,
                          source: str, prefix: str) -> dict:
    """One classified static check over normalized argv tokens.

    ``source`` is the effective option ("pw_cmd" or, for qe-ase when it
    overrides, "command"); ``prefix`` notes an override in the message.
    """
    check_id = f"{section}.{source}"
    first = tokens[0]
    kind = _classify_tokens(tokens)
    if kind == "shell_expression":
        ok, detail = _resolve_executable(first)
        if not ok:
            return {"id": check_id, "role": section, "status": "fail",
                    "message": prefix + f"{source} is a shell expression "
                               f"whose outer executable cannot be "
                               f"resolved: {detail}",
                    "remedy": "write the plain argv, or point pw_cmd "
                              "directly at a pw.x executable"}
        return {"id": check_id, "role": section, "status": "unverified",
                "message": prefix + f"{source} is a shell expression or "
                "shell wrapper; the solver behind it cannot be confirmed "
                "statically (no shell is invoked by the executors) — use "
                "literal argv words",
                "remedy": "write the plain argv, e.g. pw_cmd = "
                          "[\"mpirun\", \"-np\", \"4\", \"pw.x\"], or point "
                          "it directly at a pw.x executable"}
    if kind == "launcher":
        ok, detail = _resolve_executable(first)
        if not ok:
            return {"id": check_id, "role": section, "status": "fail",
                    "message": prefix + f"{source} invokes the launcher "
                               f"{first!r}, which itself cannot be "
                               f"resolved: {detail}",
                    "remedy": "install the launcher on PATH, or point "
                              "pw_cmd directly at a pw.x executable"}
        return {"id": check_id, "role": section, "status": "unverified",
                "message": prefix + f"{source} invokes a launcher "
                           f"({first!r}); the actual solver layer cannot "
                           "be confirmed statically — only the launcher "
                           f"was resolved ({detail}, never executed)",
                "remedy": "verify the solver in the real execution "
                          "environment (e.g. `srun pw.x` there), or point "
                          "pw_cmd directly at a pw.x executable"}
    ok, detail = _resolve_executable(first)
    args_note = ("" if len(tokens) == 1 else
                 f"; {len(tokens) - 1} argument(s) passed verbatim")
    if ok:
        return {"id": check_id, "role": section, "status": "pass",
                "message": prefix + f"{source} executable resolved: "
                           f"{detail} (never executed{args_note})"}
    return {"id": check_id, "role": section, "status": "fail",
            "message": prefix + f"{source} executable {detail}",
            "remedy": "install Quantum ESPRESSO, or set [reference] "
                      "pw_cmd to the pw.x path"}


def _check_qe_launcher(section: str, name: str, options: dict[str, Any]) -> list[dict]:
    """Launcher checks for the QE backends, on the EFFECTIVE source only.

    qe-ase's adapter lets ``command`` override ``pw_cmd`` (ASE's FileIO
    layer takes one string); the check below mirrors that priority in
    both directions.  qe (the handwritten backend) always runs the
    normalized ``pw_cmd`` argv.
    """
    if name == "qe-ase" and options.get("command") is not None:
        command = options["command"]
        if not isinstance(command, str):
            return [{"id": f"{section}.command", "role": section,
                     "status": "fail",
                     "message": "qe-ase command must be a string launch "
                                f"template, got {type(command).__name__}",
                     "remedy": "write command as one string, e.g. "
                               "command = \"mpirun -np 4 pw.x -in "
                               'espresso.pwi", or remove it to use pw_cmd'}]
        try:
            tokens = tuple(shlex.split(command, posix=True))
        except ValueError as error:
            return [{"id": f"{section}.command", "role": section,
                     "status": "fail",
                     "message": f"qe-ase command string cannot be parsed "
                                f"({error}); quote paths with spaces",
                     "remedy": "quote the path, e.g. command = "
                               "\"'/opt/QE 7.5/pw.x' -in espresso.pwi\""}]
        if not tokens:
            return [{"id": f"{section}.command", "role": section,
                     "status": "fail",
                     "message": "qe-ase command is an empty string; it is "
                                "the effective launch command and cannot "
                                "be empty",
                     "remedy": "set command to the launch string, or "
                              "remove it to use pw_cmd"}]
        return [_check_command_tokens(
            section, tokens, source="command",
            prefix="command overrides pw_cmd for qe-ase; checking the "
                   "effective value: ")]
    raw = options.get("pw_cmd") or ("pw.x",)
    try:
        tokens = normalize_pw_cmd(raw)
    except ConfigError as error:
        return [{"id": f"{section}.pw_cmd", "role": section,
                 "status": "fail", "message": str(error)}]
    return [_check_command_tokens(section, tokens, source="pw_cmd",
                                  prefix="")]


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


def _check_optional_package(prefix: str, role: str, package: str,
                            pip_hint: str, extra: str = "") -> dict:
    if _find_spec(package) is None:
        return {"id": f"{prefix}.{package}", "role": role,
                "status": "fail",
                "message": f"optional package {package!r} is not "
                           f"installed{extra}",
                "remedy": f"install it in this environment ({pip_hint}) "
                          "or choose a backend without it"}
    return {"id": f"{prefix}.{package}", "role": role, "status": "pass",
            "message": f"optional package {package!r} is importable "
                       "(not imported, not evaluated)"}


def _check_model_file(prefix: str, role: str, options: dict[str, Any]) -> dict:
    model = options.get("model")
    if model is None:
        return {"id": f"{prefix}.model", "role": role,
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
            return {"id": f"{prefix}.model", "role": role,
                    "status": "pass",
                    "message": f"local model file present: {path.name} "
                               "(weights not loaded)"}
        return {"id": f"{prefix}.model", "role": role, "status": "fail",
                "message": f"local model file not found: {path}",
                "remedy": "set model to an existing local model file"}
    return {"id": f"{prefix}.model", "role": role, "status": "unverified",
            "message": f"model {text!r} is a base-model name; its local "
                       "cache cannot be confirmed without loading weights",
            "remedy": "download/prepare the model explicitly and point "
                      "model at the local file"}


_CORE_BACKENDS = {"harmonic-reference", "harmonic-surrogate"}
_WRAPPER_BACKENDS = {"scaled", "quadratic-corrected"}
_QE_BACKENDS = {"qe", "qe-ase"}


def _backend_checks(section: str, name: str, options: dict[str, Any],
                    chain: tuple[str, ...]) -> list[dict]:
    """Static prerequisite checks for one backend.

    Wrapper backends contribute their own dependency-free note and then
    recurse into the declared base spec — the outer name alone never
    proves the wrapped base's prerequisites.  ``chain`` carries the
    wrapper names so nested ids stay attributable
    (e.g. ``surrogate.scaled.base.mace``).
    """
    if name in _WRAPPER_BACKENDS:
        checks = [{"id": f"{section}.{name}", "role": section,
                   "status": "pass",
                   "message": f"wrapper backend {name!r} adds no "
                              "dependency of its own; its declared base "
                              "backend is checked separately"}]
        base = options.get("base")
        problem = None
        if not isinstance(base, dict):
            problem = ("wrapper backend needs a base spec inline table: "
                       'base = { name = "<backend>", kwargs = { ... } }')
        elif not isinstance(base.get("name"), str):
            problem = "the wrapper base spec needs the backend 'name' string"
        elif not isinstance(base.get("kwargs", {}), dict):
            problem = "the wrapper base spec 'kwargs' must be a table"
        if problem is not None:
            checks.append({"id": f"{section}.{name}.base", "role": section,
                           "status": "fail", "message": problem,
                           "remedy": "fix the wrapper's base inline table "
                                     "in the configuration"})
            return checks
        base_name = base["name"]
        base_options = base.get("kwargs", {})
        return checks + _backend_checks(section, base_name, base_options,
                                        chain + (name,))
    prefix = (section + "." + ".".join(c + ".base" for c in chain)
              if chain else section)
    if name in _CORE_BACKENDS:
        return [{"id": f"{prefix}.{name}", "role": section,
                 "status": "pass",
                 "message": f"backend {name!r} runs in the core "
                            "environment (no optional dependencies)"}]
    if name in _QE_BACKENDS:
        checks = [_check_pseudo_files(prefix, options)]
        checks.extend(_check_qe_launcher(prefix, name, options))
        return checks
    if name == "mace":
        return [_check_optional_package(prefix, section, "mace",
                                        "pip install mace-torch"),
                _check_model_file(prefix, section, options)]
    if name == "pyscf":
        return [_check_optional_package(prefix, section, "pyscf",
                                        "pip install pyscf")]
    return [{"id": f"{prefix}.{name}", "role": section,
             "status": "unverified",
             "message": f"backend {name!r} is not a builtin; no static "
                        "environment information is available for it",
             "remedy": "verify its runtime prerequisites in the "
                       "execution environment"}]


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
        checks.extend(_backend_checks(section, backend.name,
                                      backend.options, ()))

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
