"""Pyramid command-line interface.

Thin argument parsing over :mod:`pyraimd2.config` and
:mod:`pyraimd2.workflows` — every command's logic lives in the library so
the Python API and the CLI run the same code paths.  ``--help`` and
``--version`` work offline and import nothing beyond argparse/config;
workflow modules load lazily per command.

Exit codes: 0 success, 1 run-time failure, 2 usage/configuration error,
130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyraimd2 import __version__
from pyraimd2.config import ConfigError, load_config

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Pyramid — Python wrapped Ab initio Molecular Dynamics: "
                    "configuration-driven, resumable MD with energetic "
                    "force-error prediction.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    init = commands.add_parser(
        "init", help="write a runnable configuration + structure from a template")
    init.add_argument("--template", default="harmonic",
                      help="template name (default: harmonic)")
    init.add_argument("--output", required=True,
                      help="directory to write run.toml and structure.extxyz into")
    init.add_argument("--force", action="store_true",
                      help="overwrite existing template files")
    init.set_defaults(func=_cmd_init)

    validate = commands.add_parser(
        "validate", help="check a configuration without running it")
    validate.add_argument("config", help="path to the TOML configuration")
    validate.add_argument(
        "--probe-backends", action="store_true",
        help="also run one small backend self-check on the structure "
             "(executes the reference/surrogate once; off by default)")
    validate.set_defaults(func=_cmd_validate)

    run = commands.add_parser("run", help="execute a configuration")
    run.add_argument("config", help="path to the TOML configuration")
    run.set_defaults(func=_cmd_run)

    resume = commands.add_parser(
        "resume", help="continue an adaptive run by N additional steps")
    resume.add_argument("run_dir", help="the run directory (contains "
                                        "resolved_config.json)")
    resume.add_argument("--steps", type=int, required=True,
                        help="number of ADDITIONAL steps to run (the current "
                             "and target step numbers are printed)")
    resume.add_argument("--force-unlock", action="store_true",
                        help="reclaim the event-log writer lock left by a "
                             "crashed process (only when no live writer exists)")
    resume.set_defaults(func=_cmd_resume)

    inspect_cmd = commands.add_parser(
        "inspect", help="show run status, costs and checks")
    inspect_cmd.add_argument("run_dir", help="the run directory")
    inspect_cmd.add_argument("--json", action="store_true",
                             help="print the same information as JSON")
    inspect_cmd.set_defaults(func=_cmd_inspect)

    export = commands.add_parser(
        "export", help="export the trajectory (extxyz) with an explicit force source")
    export.add_argument("run_dir", help="the run directory")
    export.add_argument("--format", default="extxyz", choices=["extxyz"],
                        help="output format (default: extxyz)")
    export.add_argument("--force-source", default="driving",
                        choices=["driving", "reference", "base"],
                        help="which forces to export: the actual driving "
                             "forces (default), the reference labels, or the "
                             "uncorrected surrogate predictions")
    export.add_argument("--output", help="output file (default: "
                                         "<run_dir>/export-<source>.extxyz)")
    export.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    export.set_defaults(func=_cmd_export)

    backends = commands.add_parser(
        "backends", help="list registered backend factories")
    backends.set_defaults(func=_cmd_backends)
    return parser


# ---------------------------------------------------------------------------
# commands


def _cmd_init(args: argparse.Namespace) -> int:
    from pyraimd2.workflows import write_template

    config_path = write_template(args.template, args.output, force=args.force)
    print(f"wrote {config_path} (+ structure.extxyz)")
    print("next steps:")
    print(f"  pyramid validate {config_path}")
    print(f"  pyramid run {config_path}")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    from pyraimd2.workflows import validate_setup

    config = load_config(args.config)
    report = validate_setup(config, probe=args.probe_backends)
    structure = report["structure"]
    momenta = ("present in the file" if structure["has_momenta"]
               else f"thermalized at {config.dynamics.temperature_K} K "
                    f"(velocity_seed {config.dynamics.velocity_seed})")
    print(f"configuration : {config.source_path} (schema_version "
          f"{config.schema_version})")
    print(f"task          : {config.task.kind} / {config.task.mode}")
    print(f"structure     : {structure['formula']}, {structure['n_atoms']} "
          f"atoms, pbc={structure['pbc']}, momenta {momenta}")
    for section in ("reference", "surrogate"):
        backend = getattr(config, section)
        if backend is None:
            continue
        caps = report["capabilities"][section]
        details = ", ".join(f"{key}={value}" for key, value in caps.items())
        print(f"{section:12s}: {backend.name} ({details})")
    if "energy_contract" in report:
        print(f"energy contract: {report['energy_contract']}")
    print(f"run directory : {config.run.directory}")
    print(f"dynamics      : {config.dynamics.steps} steps x "
          f"{config.dynamics.timestep_fs} fs, checkpoint every "
          f"{config.checkpoint.interval_steps} steps")
    checks = config.verification
    if checks.probability > 0:
        print(f"verification  : independent checks p={checks.probability} "
              f"(seed {checks.seed})")
    else:
        print("verification  : independent checks disabled (probability 0)")
    for section, probe in report["probes"].items():
        print(f"probe {section:9s}: energy {probe['energy_eV']:.6f} eV "
              f"({probe['elapsed_s']:.3f} s)")
    print("validate: OK")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    from pyraimd2.engines.base import EngineError
    from pyraimd2.workflows import WorkflowError, run_workflow

    config = load_config(args.config)
    try:
        run_workflow(config, verbose=True, handle_sigint=True)
    except WorkflowError:
        raise
    except (EngineError, RuntimeError, ValueError, OSError) as error:
        print(f"error: run failed: {error!r}", file=sys.stderr)
        print("the run directory keeps the failure record and the cost "
              "ledger; fix the cause and continue adaptive runs with "
              "`pyramid resume <run_dir> --steps N`", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    from pyraimd2.workflows import resume_workflow

    resume_workflow(args.run_dir, args.steps, force_unlock=args.force_unlock,
                    verbose=True, handle_sigint=True)
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    from pyraimd2.runtime.inspect import format_inspection, inspect_run

    info = inspect_run(args.run_dir)
    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(format_inspection(info))
    return EXIT_OK


def _cmd_export(args: argparse.Namespace) -> int:
    from pyraimd2.workflows import export_run

    report = export_run(args.run_dir, force_source=args.force_source,
                        output=args.output, force=args.force)
    print(f"wrote {report['frames']} frames ({report['force_source']} forces) "
          f"to {report['output']}")
    if report["missing_forces_frames"]:
        print(f"  {report['missing_forces_frames']} frame(s) have no "
              f"{report['force_source']} label: marked forces_available=F "
              "with NaN forces — missing data, never zero-filled")
    return EXIT_OK


def _cmd_backends(args: argparse.Namespace) -> int:
    from pyraimd2.backends import available_backends

    print("registered backends:")
    for name, info in available_backends().items():
        print(f"  {name:20s} {info['kind'] or '-':10s} {info['origin']}")
    return EXIT_OK


# ---------------------------------------------------------------------------


def _usage_errors() -> tuple[type[BaseException], ...]:
    """Known, user-actionable errors (imported lazily: --help stays light)."""
    from pyraimd2.backends import BackendRegistryError
    from pyraimd2.engines.base import CapabilityMismatchError
    from pyraimd2.runtime import EventLogError, ResumeError
    from pyraimd2.workflows import ExportError, WorkflowError

    return (ConfigError, WorkflowError, ExportError, BackendRegistryError,
            CapabilityMismatchError, EventLogError, ResumeError)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except _usage_errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
