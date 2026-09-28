#!/usr/bin/env python3
"""Prepared QE launcher — prepare an environment ONCE per Slurm job.

Standard library only; no Pyramid imports.  Two subcommands:

``prepare`` — source the user's trusted setup script in a SEPARATE bash
subprocess (the parent process environment is never modified), resolve
the MPI launcher and pw.x with ``command -v`` in the prepared
environment, and record the environment DELTA versus the parent (an
explicit runtime allowlist by default: PATH, LD_LIBRARY_PATH,
LIBRARY_PATH, XML_CATALOG_FILES, OMP_/OPENBLAS_/MKL_ thread variables and
CONDA_*/MINIFORGE variables; ``--allow-var NAME`` adds more; variables
the setup REMOVES are recorded as removals and unset at run time; shell
noise like ``_``/``SHLVL``/``PWD`` is not a runtime change).
Credential-class variables (KEY/TOKEN/SECRET/PASS/CRED/AUTH/CERT in the
name) that the setup added or changed abort the prepare and are reported
BY NAME ONLY — never stored, never printed.  The state directory is mode
0700 with 0600 files and atomic writes, and an existing state is never
overwritten.  The state binds the current SLURM_JOB_ID (required) and
carries a content SHA256 of its payload.

``run`` — verify BEFORE spawning anything that the current SLURM_JOB_ID
matches and the state payload hash matches ``--expected-sha256``
(refusals never print environment VALUES), then ``os.execvpe`` the
resolved MPI + pw.x with the recorded environment applied and the extra
argv (e.g. ``-in pw.in``) appended — no extra supervisor process, no
shell, and child exit codes and SIGINT propagate naturally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_ALLOW_VARS = ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
                      "XML_CATALOG_FILES")
ALLOW_PREFIXES = ("OMP_", "OPENBLAS_", "MKL_", "CONDA_", "MINIFORGE")
#: shell bookkeeping, not runtime environment
SHELL_NOISE = {"_", "SHLVL", "OLDPWD", "PWD"}
CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASS", "CRED", "AUTH",
                      "CERT")

USAGE = 2
FAILURE = 1


def _allowed(name: str, extra: set[str]) -> bool:
    if name in SHELL_NOISE:
        return False
    return (name in DEFAULT_ALLOW_VARS or name in extra
            or any(name.startswith(prefix) for prefix in ALLOW_PREFIXES))


def _is_credential(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in CREDENTIAL_MARKERS)


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def cmd_prepare(args: argparse.Namespace) -> int:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("error: prepare runs only inside an allocated Slurm job "
              "(SLURM_JOB_ID is not set)", file=sys.stderr)
        return USAGE
    setup = Path(args.setup)
    if not setup.is_file():
        print(f"error: setup script not found: {setup}", file=sys.stderr)
        return USAGE
    if args.ranks < 1:
        print(f"error: --ranks must be >= 1, got {args.ranks}",
              file=sys.stderr)
        return USAGE
    state_dir = Path(args.state)
    state_file = state_dir / "state.json"
    if state_file.exists():
        print(f"error: state already exists: {state_file}; a prepared "
              "state is job-private and never overwritten — prepare a new "
              "directory (a resubmission is a new allocation)",
              file=sys.stderr)
        return USAGE

    # The setup is sourced in a separate bash subprocess; the machine-
    # readable environment goes to a private temp file, so setup's own
    # stdout can never mix into it (setup output passes through to the
    # user untouched).
    with tempfile.TemporaryDirectory(prefix="prepared-qe-") as scratch:
        dump = Path(scratch) / "env"
        mpi_out = Path(scratch) / "mpi"
        pw_out = Path(scratch) / "pw"
        child_script = (
            'source "$1" || { echo "error: setup script failed: $1" >&2;'
            ' exit 95; }\n'
            'env -0 > "$2"\n'
            'command -v -- "$5" > "$3" || { echo "error: MPI launcher'
            ' \'"$5"\' not found after sourcing $1" >&2; exit 96; }\n'
            'command -v -- "$6" > "$4" || { echo "error: pw.x executable'
            ' \'"$6"\' not found after sourcing $1" >&2; exit 97; }\n')
        completed = subprocess.run(
            ["bash", "-c", child_script, "bash", str(setup), str(dump),
             str(mpi_out), str(pw_out), args.mpi, args.pw],
            check=False)
        if completed.returncode != 0:
            return FAILURE
        child_env: dict[str, str] = {}
        for entry in dump.read_bytes().split(b"\0"):
            if entry:
                key, _, value = entry.partition(b"=")
                child_env[key.decode("utf-8", "surrogateescape")] = \
                    value.decode("utf-8", "surrogateescape")
        mpi_path = mpi_out.read_text().strip()
        pw_path = pw_out.read_text().strip()

    parent_env = dict(os.environ)
    # credential-class variables: a setup that adds or changes one is
    # refused by NAME ONLY — values are never stored or printed
    leaked = sorted(name for name, value in child_env.items()
                    if _is_credential(name)
                    and parent_env.get(name) != value)
    if leaked:
        print("error: the setup added or changed credential-class "
              f"variable(s) {leaked}; refusing to record them — fix the "
              "setup script (credentials must never enter the prepared "
              "state)", file=sys.stderr)
        return USAGE
    extra_allow = set(args.allow_var or [])
    env_set = {name: value for name, value in child_env.items()
               if _allowed(name, extra_allow)
               and parent_env.get(name) != value}
    env_unset = sorted(name for name in parent_env
                       if _allowed(name, extra_allow)
                       and name not in child_env)
    for path, label in ((mpi_path, "MPI launcher"), (pw_path, "pw.x")):
        if not Path(path).is_absolute():
            print(f"error: the resolved {label} path is not absolute: "
                  f"{path!r}", file=sys.stderr)
            return FAILURE
    payload = {
        "schema_version": SCHEMA_VERSION,
        "slurm_job_id": job_id,
        "argv": [mpi_path, "-np", str(args.ranks), pw_path],
        "env_set": dict(sorted(env_set.items())),
        "env_unset": env_unset,
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    record = {"payload": payload, "payload_sha256": digest}

    created_dir = not state_dir.exists()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if created_dir:
        os.chmod(state_dir, 0o700)
    fd, tmp_name = tempfile.mkstemp(dir=state_dir, prefix="state.json.",
                                    suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, allow_nan=False)
                         + "\n")
        os.replace(tmp, state_file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"prepared state: {state_file}")
    print(f"  argv  : {payload['argv']}")
    print(f"  env   : {len(env_set)} variable(s) set/changed, "
          f"{len(env_unset)} removed at run time")
    print(f"  sha256: {digest}")
    print("pw_cmd for your configuration (absolute paths):")
    print(f'  pw_cmd = ["{Path(__file__).resolve()}", "run", "--state", '
          f'"{state_dir.resolve()}", "--expected-sha256", "{digest}", "--"]')
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    state_file = Path(args.state) / "state.json"
    try:
        record = json.loads(state_file.read_text(encoding="utf-8"))
        payload = record["payload"]
    except (OSError, ValueError, KeyError) as error:
        print(f"error: cannot read the prepared state at {state_file}: "
              f"{error}", file=sys.stderr)
        return USAGE
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if actual != record.get("payload_sha256"):
        print("error: the prepared state's payload does not match its own "
              "recorded hash — the state directory looks modified; prepare "
              "a fresh one", file=sys.stderr)
        return USAGE
    if actual != args.expected_sha256:
        print("error: the prepared state's content hash does not match "
              "--expected-sha256; this run refuses to use a state it was "
              "not configured for", file=sys.stderr)
        return USAGE
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id != payload["slurm_job_id"]:
        print(f"error: the prepared state belongs to SLURM_JOB_ID "
              f"{payload['slurm_job_id']} but this process has "
              f"{job_id!r}; a prepared state is valid only inside its own "
              "allocation", file=sys.stderr)
        return USAGE
    env = dict(os.environ)
    for name in payload["env_unset"]:
        env.pop(name, None)
    env.update(payload["env_set"])
    argv = list(payload["argv"]) + list(args.argv)
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as error:
        print(f"error: cannot launch {argv[0]!r}: {error}", file=sys.stderr)
        return FAILURE
    return FAILURE  # unreachable: execvpe replaces this process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Prepared QE launcher: prepare the environment once "
                    "per Slurm allocation, then reuse it per call (see "
                    "README.md).")
    commands = parser.add_subparsers(dest="command", metavar="command")
    prepare = commands.add_parser(
        "prepare", help="source the setup once and record the prepared "
                        "state (inside an allocated Slurm job only)")
    prepare.add_argument("--setup", required=True,
                         help="your trusted setup script (sourced in a "
                              "separate bash subprocess)")
    prepare.add_argument("--state", required=True,
                         help="job-private state directory (mode 0700, "
                              "never overwritten)")
    prepare.add_argument("--mpi", required=True,
                         help="MPI launcher name (resolved via command -v "
                              "in the prepared environment)")
    prepare.add_argument("--pw", required=True,
                         help="pw.x executable name (resolved likewise)")
    prepare.add_argument("--ranks", type=int, required=True,
                         help="MPI ranks for the prepared argv (-np N)")
    prepare.add_argument("--allow-var", action="append", default=None,
                         metavar="NAME",
                         help="add NAME to the runtime environment "
                              "allowlist (repeatable)")
    prepare.set_defaults(func=cmd_prepare)
    run = commands.add_parser(
        "run", help="exec the prepared command after verifying job id and "
                    "state hash (binds the prepared state into pw_cmd)")
    run.add_argument("--state", required=True, help="the prepared state "
                                                    "directory")
    run.add_argument("--expected-sha256", required=True,
                     help="the payload hash printed by prepare")
    run.add_argument("argv", nargs=argparse.REMAINDER,
                     help="everything after `--` is appended verbatim "
                          "(e.g. -- -in pw.in)")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "command", None):
        build_parser().print_help()
        return USAGE
    if getattr(args, "argv", None) and args.argv[0] == "--":
        args.argv = args.argv[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
