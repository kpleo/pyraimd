"""Run the serial relax -> NVT -> NVE recipe on the periodic LJ example.

Usage:
  python run_recipe.py --output results/lj-recipe          # new or continue

Re-invoking the same command continues an unfinished recipe and never
recomputes a finished stage: a crashed MD stage resumes through the normal
protocol to its configured total, and a completed stage is skipped.  The
per-stage state, provenance and costs land in <output>/workflow.json.

Stage semantics: a fixed-surrogate FIRE relaxation; an adaptive Langevin
NVT stage (the LJ reference against the 5%-softer surrogate, fixed model)
whose momenta are initialized exactly once at the target temperature; and a
fixed-surrogate NVE stage that keeps the complete NVT momenta and carries
no bath settings.  "Completed NVT" means the configured steps ran — it is
not a claim of physical thermal equilibrium.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "structure.extxyz").exists():
        subprocess.run([sys.executable, str(HERE / "make_structure.py")],
                       cwd=out, check=True)
    for name in STAGE_TOMLS:
        shutil.copy2(HERE / "recipe" / name, out / name)
    manifest = run_serial_recipe(
        out, [RecipeStage("relax", out / "relax.toml"),
              RecipeStage("nvt", out / "nvt.toml", momenta="initialize"),
              RecipeStage("nve", out / "nve.toml", momenta="preserve")])
    for stage in manifest["stages"]:
        result = stage["result"] or {}
        costs = result.get("reference_by_purpose") or {}
        steps = result.get("complete_steps", result.get("optimizer_steps"))
        print(f"{stage['name']:>5}: {stage['status']} — run "
              f"{stage['run_id']}, steps {steps}, "
              f"physical time {result.get('physical_time_fs', 0.0)} fs, "
              f"reference executions {result.get('reference_executions')} "
              f"({costs}), inference {result.get('inference_executions')}, "
              f"source {((stage.get('source') or {}).get('source_run_id'))}")
    print(f"manifest: {out / 'workflow.json'}")


if __name__ == "__main__":
    main()
