"""Run the serial relax -> adaptive NVT -> NVE recipe on 8-atom diamond Si.

Usage:
  python run_recipe.py --output results/si-recipe --prepare-only   # setup only
  python run_recipe.py --output results/si-recipe                  # new or continue

The controller and the completed-state reader live in the package
(`pyraimd2.workflows.stages`); this script only assembles stages — no
recovery logic of its own.  Re-invoking the same command is idempotent: a
finished stage is never recomputed, a crashed MD stage resumes through the
ordinary protocol to its configured total, and changed settings refuse
before any new computation.  Stage status, parent provenance and
purpose-split costs land in <output>/workflow.json.

Stage semantics: fixed-MACE FIRE relaxation; an adaptive Langevin NVT
stage (Quantum ESPRESSO reference against the frozen MACE surrogate) whose
momenta are initialized exactly once at 300 K; and a fixed-MACE NVE stage
that keeps the complete NVT momenta and carries no bath settings.
"Completed NVT" means the configured 12 steps ran — it is not a claim of
thermal equilibrium or of unbiased canonical sampling, and the NVE stage
propagates the frozen surrogate potential only, not all-reference AIMD.

Inputs are user-provided (never prefilled): a MACE-MPA-0 medium
checkpoint, a PAW pseudopotential directory, and a `pw.x` launch command —
see README.md for the exact configuration keys.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pyraimd2.workflows.stages import RecipeStage, run_serial_recipe

HERE = Path(__file__).resolve().parent
STAGE_TOMLS = ("relax.toml", "nvt.toml", "nve.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="recipe output directory (user-owned)")
    parser.add_argument("--force-unlock", action="store_true",
                        help="reclaim the writer lock of a crashed MD stage "
                             "deliberately (only after confirming no live "
                             "writer exists)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="only write structure.extxyz and the stage "
                             "TOMLs into --output and say what to edit — "
                             "no computation, no inference, no downloads")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "structure.extxyz").exists():
        subprocess.run([sys.executable, str(HERE / "make_structure.py")],
                       cwd=out, check=True)
    for name in STAGE_TOMLS:
        target = out / name
        template = (HERE / "recipe" / name).read_bytes()
        if not target.exists():
            target.write_bytes(template)
        elif target.read_bytes() != template:
            # Existing user configs are kept, never silently overwritten;
            # the recipe controller reconciles or refuses on differences.
            print(f"  keeping existing {name} (differs from the shipped "
                  "template; edits stay visible)")
    if args.prepare_only:
        print(f"prepared {out} (no computation, no inference, no downloads):")
        print("  structure.extxyz, relax.toml, nvt.toml, nve.toml")
        print("next, place your inputs beside those TOMLs — relative paths "
              "resolve against the output directory, not the recipe/ "
              "template directory:")
        print("  [surrogate] model            e.g. ./mace-mpa-0-medium.model")
        print("  [reference] pseudo_dir       e.g. ./qe_pseudos/ holding the "
              "files named under [reference.pseudos]")
        print("  [reference] pw_cmd           your pw.x launch command")
        print(f"then validate (no computation):  pyramid validate "
              f"{out / 'nvt.toml'}")
        print(f"then run / continue:           python {Path(__file__).name} "
              f"--output {out}")
        return
    manifest = run_serial_recipe(
        out, [RecipeStage("relax", out / "relax.toml"),
              RecipeStage("nvt", out / "nvt.toml", momenta="initialize"),
              RecipeStage("nve", out / "nve.toml", momenta="preserve")],
        force_unlock=args.force_unlock)
    for stage in manifest["stages"]:
        result = stage["result"] or {}
        costs = result.get("reference_by_purpose") or {}
        steps = result.get("complete_steps", result.get("optimizer_steps"))
        unresolved = result.get("reference_unresolved") or 0
        print(f"{stage['name']:>5}: {stage['status']} — run "
              f"{stage['run_id']}, steps {steps}, "
              f"physical time {result.get('physical_time_fs', 0.0)} fs, "
              f"reference executions {result.get('reference_executions')} "
              f"({costs}), inference {result.get('inference_executions')}, "
              f"source {((stage.get('source') or {}).get('source_run_id'))}"
              + (f", UNRESOLVED attempts {unresolved} (cost record "
                 "incomplete)" if unresolved else ""))
    if manifest.get("stopped_early"):
        stop = manifest.get("stop") or {}
        print(f"stopped: stage {stop.get('stage')!r} at "
              f"{stop.get('complete_steps')} complete steps — re-run the "
              "same command to continue")
    print(f"manifest: {out / 'workflow.json'}")


if __name__ == "__main__":
    main()
