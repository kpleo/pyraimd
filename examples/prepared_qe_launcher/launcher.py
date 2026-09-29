#!/usr/bin/env python3
"""Prepared QE launcher — prepare an environment ONCE per Slurm job.

Standard library only; no Pyramid imports.  Three subcommands:

``prepare`` — source the user's trusted setup script in a SEPARATE bash
subprocess (the parent process environment is never modified), resolve
the MPI launcher and pw.x with ``command -v`` in the prepared
environment, and record the COMPLETE set of managed runtime variables
as resolved by the setup.  The managed set is an explicit allowlist by
default (PATH, LD_LIBRARY_PATH, LIBRARY_PATH, XML_CATALOG_FILES,
OMP_/OPENBLAS_/MKL_ thread variables and CONDA_*/MINIFORGE variables;
``--allow-var NAME`` adds more) — the REST of the environment is never
recorded.  Any REAL change the setup makes OUTSIDE the managed set
(added, modified or removed; shell noise like ``_``/``SHLVL``/``PWD``/
``BASH_FUNC_*`` excluded) aborts the prepare BEFORE anything is written,
reported by variable NAME only with the ``--allow-var`` remedy — unknown
changes are never silently dropped.  Credential-class variables
(KEY/TOKEN/SECRET/PASS/CRED/AUTH/CERT in the name) abort even via
``--allow-var`` and are never stored or printed.  The state directory is
created EXCLUSIVELY (any pre-existing one is refused; two racing
prepares leave exactly one winner), mode 0700 with 0600 atomic files,
and binds the current SLURM_JOB_ID plus a content SHA256 of the payload
(the managed set and the allow policy are part of the hashed payload).
``prepare`` also writes ``pw_cmd.json`` next to ``state.json``: the
machine-readable argv (the prepare-time Python interpreter, this
launcher's absolute path, the state path and the expected hash).

``run`` — verify BEFORE spawning anything that the current SLURM_JOB_ID
matches and the state payload hash matches ``--expected-sha256``
(refusals never print environment VALUES), rebuild the managed set
exactly (prepared values win; managed variables NOT in the prepared set
are unset even when present now; everything else keeps inheriting from
the current process — the state does NOT freeze the whole environment),
then ``os.execvpe`` the resolved MPI + pw.x with the extra argv
(e.g. ``-in pw.in``) appended — no extra supervisor process, no shell,
and child exit codes and SIGINT propagate naturally.

``render-config`` — substitute the ONE explicit placeholder
``@PREPARED_PW_CMD@`` in a user-provided run.toml.template with the
state's pw_cmd argv, each element quoted as a proper TOML string, and
write the job-specific config atomically (an existing output is refused;
the template itself is never modified).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

SCHEMA_VERSION = 2

DEFAULT_ALLOW_VARS = ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
                      "XML_CATALOG_FILES")
ALLOW_PREFIXES = ("OMP_", "OPENBLAS_", "MKL_", "CONDA_", "MINIFORGE")
#: shell bookkeeping and function exports, not runtime environment
SHELL_NOISE = {"_", "SHLVL", "OLDPWD", "PWD"}
SHELL_NOISE_PREFIXES = ("BASH_FUNC_",)
CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASS", "CRED", "AUTH",
                      "CERT")
PW_CMD_PLACEHOLDER = "@PREPARED_PW_CMD@"

USAGE = 2
FAILURE = 1


def _is_noise(name: str) -> bool:
    return (name in SHELL_NOISE
            or any(name.startswith(prefix) for prefix in SHELL_NOISE_PREFIXES))


def _is_credential(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in CREDENTIAL_MARKERS)


def _policy(extra_allow: list[str]) -> dict:
    return {"base_allow_vars": list(DEFAULT_ALLOW_VARS),
            "allow_prefixes": list(ALLOW_PREFIXES),
            "extra_allow": sorted(extra_allow)}


def _allowed_by_policy(name: str, policy: dict) -> bool:
    return (name in policy["base_allow_vars"]
            or name in policy["extra_allow"]
            or any(name.startswith(prefix)
                   for prefix in policy["allow_prefixes"]))


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _payload_digest(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _write_0600_atomic(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent,
                                    prefix=path.name + ".",
                                    suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
    extra_allow = list(args.allow_var or [])
    bad_allow = [name for name in extra_allow if _is_credential(name)]
    if bad_allow:
        print(f"error: --allow-var names credential-class variable(s) "
              f"{bad_allow}; credentials never enter the prepared state",
              file=sys.stderr)
        return USAGE
    state_dir = Path(args.state)

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
    # Every real change the setup made, in both directions (shell noise
    # excluded).  A change OUTSIDE the managed set is a hard refusal —
    # never silently dropped — reported by NAME only.
    changed_or_added = {name for name, value in child_env.items()
                        if not _is_noise(name)
                        and parent_env.get(name) != value}
    removed = {name for name in parent_env
               if not _is_noise(name) and name not in child_env}
    delta = changed_or_added | removed
    leaked = sorted(name for name in delta if _is_credential(name))
    if leaked:
        print("error: the setup added or changed credential-class "
              f"variable(s) {leaked}; refusing to record them — fix the "
              "setup script (credentials must never enter the prepared "
              "state)", file=sys.stderr)
        return USAGE
    policy = _policy(extra_allow)
    unallowed = sorted(name for name in delta
                       if not _allowed_by_policy(name, policy))
    if unallowed:
        print("error: the setup changed variable(s) OUTSIDE the managed "
              f"set: {unallowed}; unknown runtime changes are never "
              "silently dropped — either remove them from the setup, or "
              "declare each one explicitly with "
              "`prepare --allow-var NAME` (repeatable)",
              file=sys.stderr)
        return USAGE
    managed_env = {name: value for name, value in child_env.items()
                   if _allowed_by_policy(name, policy)}
    managed_credentials = sorted(name for name in managed_env
                                 if _is_credential(name))
    if managed_credentials:
        print(f"error: credential-class variable(s) "
              f"{managed_credentials} matched the managed set; they are "
              "never recorded", file=sys.stderr)
        return USAGE
    for path, label in ((mpi_path, "MPI launcher"), (pw_path, "pw.x")):
        if not Path(path).is_absolute():
            print(f"error: the resolved {label} path is not absolute: "
                  f"{path!r}", file=sys.stderr)
            return FAILURE
    payload = {
        "schema_version": SCHEMA_VERSION,
        "slurm_job_id": job_id,
        "argv": [mpi_path, "-np", str(args.ranks), pw_path],
        "managed_env": dict(sorted(managed_env.items())),
        "env_policy": policy,
    }
    digest = _payload_digest(payload)

    # Exclusive state creation: any pre-existing state dir is refused and
    # two racing prepares leave exactly one winner — no check-then-write
    # window.  Only this process's own creation is ever removed/chmod'ed.
    if not state_dir.parent.exists():
        state_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(state_dir, 0o700)
    except FileExistsError:
        print(f"error: state directory already exists: {state_dir}; a "
              "prepared state is job-private and created exclusively — "
              "prepare a fresh directory (a resubmission is a new "
              "allocation)", file=sys.stderr)
        return USAGE
    except OSError as error:
        print(f"error: cannot create the state directory {state_dir}: "
              f"{error}", file=sys.stderr)
        return FAILURE
    try:
        record = {"payload": payload, "payload_sha256": digest}
        _write_0600_atomic(
            state_dir / "state.json",
            json.dumps(record, indent=2, allow_nan=False) + "\n")
        pw_cmd = {"argv": [sys.executable,
                           str(Path(__file__).resolve()), "run",
                           "--state", str(state_dir.resolve()),
                           "--expected-sha256", digest, "--"]}
        _write_0600_atomic(
            state_dir / "pw_cmd.json",
            json.dumps(pw_cmd, indent=2, allow_nan=False) + "\n")
    except BaseException:
        # only what THIS process created
        for name in ("pw_cmd.json", "state.json"):
            (state_dir / name).unlink(missing_ok=True)
        try:
            state_dir.rmdir()
        except OSError:
            pass
        raise
    print(f"prepared state: {state_dir / 'state.json'}")
    print(f"  argv  : {payload['argv']}")
    print(f"  env   : {len(managed_env)} managed runtime variable(s) "
          "recorded (complete managed set; managed items absent here are "
          "unset at run time)")
    print(f"  sha256: {digest}")
    print(f"pw_cmd argv (also in {state_dir / 'pw_cmd.json'}):")
    print(f"  {pw_cmd['argv']}")
    return 0


def _load_state(state_dir: str, expected_sha256: str) -> dict | None:
    """The verified payload, or None after a printed refusal (never with
    environment values)."""
    state_file = Path(state_dir) / "state.json"
    try:
        record = json.loads(state_file.read_text(encoding="utf-8"))
        payload = record["payload"]
    except (OSError, ValueError, KeyError) as error:
        print(f"error: cannot read the prepared state at {state_file}: "
              f"{error}", file=sys.stderr)
        return None
    actual = _payload_digest(payload)
    if actual != record.get("payload_sha256"):
        print("error: the prepared state's payload does not match its own "
              "recorded hash — the state directory looks modified; prepare "
              "a fresh one", file=sys.stderr)
        return None
    if actual != expected_sha256:
        print("error: the prepared state's content hash does not match "
              "--expected-sha256; this run refuses to use a state it was "
              "not configured for", file=sys.stderr)
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        print(f"error: the prepared state has schema_version "
              f"{payload.get('schema_version')!r}; this launcher reads "
              f"{SCHEMA_VERSION} — prepare a fresh state",
              file=sys.stderr)
        return None
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id != payload["slurm_job_id"]:
        print(f"error: the prepared state belongs to SLURM_JOB_ID "
              f"{payload['slurm_job_id']} but this process has "
              f"{job_id!r}; a prepared state is valid only inside its own "
              "allocation", file=sys.stderr)
        return None
    return payload


def cmd_run(args: argparse.Namespace) -> int:
    payload = _load_state(args.state, args.expected_sha256)
    if payload is None:
        return USAGE
    policy = payload["env_policy"]
    managed = payload["managed_env"]
    env = dict(os.environ)
    # rebuild the managed set exactly: prepared values win; managed
    # variables NOT in the prepared set are unset even when present now;
    # everything outside the managed set keeps inheriting
    for name in list(env):
        if _allowed_by_policy(name, policy) and name not in managed:
            env.pop(name)
    env.update(managed)
    argv = list(payload["argv"]) + list(args.argv)
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as error:
        print(f"error: cannot launch {argv[0]!r}: {error}", file=sys.stderr)
        return FAILURE
    return FAILURE  # unreachable: execvpe replaces this process


def cmd_render_config(args: argparse.Namespace) -> int:
    template = Path(args.template)
    output = Path(args.output)
    try:
        text = template.read_text(encoding="utf-8")
    except OSError as error:
        print(f"error: cannot read the template {template}: {error}",
              file=sys.stderr)
        return USAGE
    count = text.count(PW_CMD_PLACEHOLDER)
    if count != 1:
        print(f"error: the template must contain the placeholder "
              f"{PW_CMD_PLACEHOLDER!r} exactly once (found {count})",
              file=sys.stderr)
        return USAGE
    try:
        pw_cmd = json.loads((Path(args.state) / "pw_cmd.json")
                            .read_text(encoding="utf-8"))["argv"]
    except (OSError, ValueError, KeyError) as error:
        print(f"error: cannot read pw_cmd.json in {args.state}: {error}",
              file=sys.stderr)
        return USAGE
    # JSON string quoting is valid TOML basic-string quoting for path
    # text (no control characters); each argv element stays one TOML
    # string — never sed, never eval
    for element in pw_cmd:
        if any(ord(char) < 0x20 for char in element):
            print("error: the prepared argv contains a control character; "
                  "refusing to render it into TOML", file=sys.stderr)
            return FAILURE
    toml_array = "[" + ", ".join(json.dumps(element) for element in pw_cmd) \
        + "]"
    rendered = text.replace(PW_CMD_PLACEHOLDER, toml_array)
    # syntax-check the rendered text BEFORE anything is published: an
    # invalid render never becomes a runnable target (the full Pyramid
    # schema still goes through the template's own validate step)
    try:
        tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as error:
        print(f"error: the rendered configuration is not valid TOML: "
              f"{error}; nothing was written", file=sys.stderr)
        return USAGE
    if not output.parent.is_dir():
        print(f"error: the output directory does not exist: "
              f"{output.parent}; place the generated config in an "
              "existing directory (next to the template)",
              file=sys.stderr)
        return USAGE
    # atomic no-clobber publish: full content to a same-directory temp,
    # then os.link — an existing file OR symlink at the target is
    # refused, and two concurrent renders leave exactly one creator
    fd, tmp_name = tempfile.mkstemp(dir=output.parent,
                                    prefix=output.name + ".",
                                    suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        try:
            os.link(tmp, output)
        except FileExistsError:
            print(f"error: output config exists: {output}; a "
                  "job-specific config is never overwritten or "
                  "dereferenced — remove it or choose a new name",
                  file=sys.stderr)
            return USAGE
        except OSError as error:
            print(f"error: cannot publish {output}: {error}",
                  file=sys.stderr)
            return FAILURE
    finally:
        tmp.unlink(missing_ok=True)
    print(f"wrote {output} (pw_cmd bound to state "
          f"{Path(args.state).resolve()})")
    return 0


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
                         help="job-private state directory (created "
                              "exclusively, mode 0700; any pre-existing "
                              "one is refused)")
    prepare.add_argument("--mpi", required=True,
                         help="MPI launcher name (resolved via command -v "
                              "in the prepared environment)")
    prepare.add_argument("--pw", required=True,
                         help="pw.x executable name (resolved likewise)")
    prepare.add_argument("--ranks", type=int, required=True,
                         help="MPI ranks for the prepared argv (-np N)")
    prepare.add_argument("--allow-var", action="append", default=None,
                         metavar="NAME",
                         help="add NAME to the managed runtime-variable "
                              "set (repeatable; credential-class names "
                              "are refused)")
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
    render = commands.add_parser(
        "render-config",
        help="render a run.toml.template's @PREPARED_PW_CMD@ placeholder "
             "with the state's pw_cmd argv into a job-specific config "
             "(never modifies the template)")
    render.add_argument("--template", required=True,
                        help="the user's run.toml.template (contains "
                             "@PREPARED_PW_CMD@ exactly once)")
    render.add_argument("--state", required=True,
                        help="the prepared state directory (reads "
                             "pw_cmd.json)")
    render.add_argument("--output", required=True,
                        help="the job-specific config to write (must not "
                             "exist; place it NEXT TO the template so "
                             "relative paths keep resolving)")
    render.set_defaults(func=cmd_render_config)
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
