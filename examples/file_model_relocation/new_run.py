"""Step 1 of the file-model relocation example: create a run and exit.

Usage:
  python examples/file_model_relocation/new_run.py --output demo/origin

Writes the model file, structure and run.toml into the caller-chosen
output directory (which must not exist yet or be completely empty), runs
the configured NVE steps in this process and exits.  The run is continued
by resume_run.py — in a new process, possibly after the whole directory
has been moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow

STRUCTURE = "2\nH2 toy molecule\nH 0.85 0.9 0.9\nH 0.95 0.9 0.9\n"

CONFIG_TEMPLATE = """\
schema_version = 1
[run]
id = "file-model-demo"
directory = "."
seed = 42
[task]
kind = "md"
mode = "reference"
[structure]
file = "structure.extxyz"
[reference]
backend = "file_model_reference"
model = {model}
[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = {steps}
temperature_K = 300.0
velocity_seed = 7
[checkpoint]
interval_steps = 1
[output]
trajectory_interval_steps = 1
summary_interval_steps = 2
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="new or completely empty directory that "
                             "becomes the run")
    parser.add_argument("--steps", type=int, default=3,
                        help="NVE steps for this first leg (default 3)")
    parser.add_argument("--stiffness", type=float, default=1.5,
                        help="harmonic stiffness k written to the model "
                             "file, in eV/angstrom^2 (default 1.5)")
    args = parser.parse_args()

    # Output rule: the directory must not exist yet or be completely
    # empty — anything else is refused with nothing written.
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"{output} exists and is not an empty directory; "
                         "choose a new or completely empty directory "
                         "(nothing was written)")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "inputs" / "model.dat"
    model.parent.mkdir()
    model.write_text(f"{args.stiffness}\n", encoding="utf-8")
    (output / "structure.extxyz").write_text(STRUCTURE, encoding="utf-8")
    config_path = output / "run.toml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            # TOML-compatible quoting: keep non-ASCII (including non-BMP)
            # characters as themselves instead of JSON's surrogate-pair
            # escapes, which TOML rejects
            model=json.dumps(str(model), ensure_ascii=False),
            steps=args.steps),
        encoding="utf-8")

    result = run_workflow(load_config(config_path), verbose=False)
    baseline = json.loads(
        (output / "file_resources.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "run_dir": str(output),
        "steps_completed": result.steps_completed,
        "model_file": str(model),
        "declared_resources": {
            key: record["original_path"]
            for key, record in baseline["resources"].items()},
        "next": ("relocate the directory, then resume in a new process: "
                 "resume_run.py --run <new run dir> --extra-steps 2 "
                 "--model <new model path>"),
    }, indent=2))


if __name__ == "__main__":
    main()
