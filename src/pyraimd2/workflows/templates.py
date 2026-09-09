"""Init templates: complete, runnable configuration + structure pairs.

``harmonic`` is the offline demo: analytic builtin backends, a small H2O
molecule, and policy/verification settings that exercise anchoring,
acceptance, checks, checkpoints and resume in a few seconds without any
external program.  The numbers are demonstration values, not accuracy
recommendations for any material.
"""

from __future__ import annotations

from pathlib import Path

from pyraimd2.workflows.setup import WorkflowError

TEMPLATES = ("harmonic", "harmonic-nvt")

HARMONIC_CONFIG = """\
# Pyramid configuration — harmonic offline demo (schema_version 1).
# Units are part of the field names: angstrom, eV, eV/angstrom, fs, K.
# Relative paths resolve against THIS file's directory.
schema_version = 1

[run]
id = "harmonic-demo"
directory = "runs/harmonic-demo"
seed = 42

[task]
kind = "md"                       # singlepoint | relax | md
mode = "adaptive"                 # reference | surrogate | adaptive

[structure]
file = "structure.extxyz"

[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = 20
temperature_K = 300.0             # used only when the structure has no velocities
velocity_seed = 7

[reference]
backend = "harmonic-reference"    # builtin analytic reference
k = 1.0
r0 = 0.9

[surrogate]
backend = "harmonic-surrogate"    # same well plus a deterministic force bias
k = 1.0
r0 = 0.9
bias = 0.05

[policy]
name = "energetic"
force_budget_eV_A = 0.1
probe_steps_A = [0.02, 0.04]
numerical_floor_eV_A = 0.001
time_cap_fs = 3.0
transverse_cap = 0.1

[verification]
probability = 0.1                 # independent checks over accepted evaluations
failure_probability = 0.05
tilt = 0.6931471805599453         # ln 2
seed = 19

[checkpoint]
interval_steps = 5
keep_generations = 2              # fixed by the 0.4 runtime

[output]
trajectory_interval_steps = 1
summary_interval_steps = 10
"""

HARMONIC_STRUCTURE = """\
3
H2O for the harmonic demo, placed near the toy well minimum (r0 = 0.9); no velocities, so momenta are thermalized from dynamics.temperature_K with dynamics.velocity_seed
O       0.870000    0.910000    0.885000
H       0.945000    0.862000    0.918000
H       0.893000    0.948000    0.955000
"""

def _strip_sections(text: str, names: tuple[str, ...]) -> str:
    for name in names:
        start = text.index(name)
        following = text.index("\n[", start + 1)
        text = text[:start] + text[following + 1:]
    return text


HARMONIC_NVT_CONFIG = _strip_sections(
    HARMONIC_CONFIG.replace(
        '# Pyramid configuration — harmonic offline demo (schema_version 1).',
        '# Pyramid configuration — harmonic offline NVT demo (schema_version 1).'
    ).replace(
        'id = "harmonic-demo"', 'id = "harmonic-nvt-demo"'
    ).replace(
        'directory = "runs/harmonic-demo"', 'directory = "runs/harmonic-nvt-demo"'
    ).replace(
        'mode = "adaptive"', 'mode = "surrogate"'
    ).replace(
        """[dynamics]
ensemble = "nve"
timestep_fs = 0.5
steps = 20
temperature_K = 300.0             # used only when the structure has no velocities
velocity_seed = 7""",
        """[dynamics]
ensemble = "nvt"
integrator = "langevin"
timestep_fs = 0.5
steps = 20
temperature_K = 300.0             # bath target temperature
friction_per_fs = 0.01            # bath coupling (required for NVT)
thermostat_seed = 123             # new-run thermostat stream (resume restores it)
velocity_seed = 7"""),
    ("[reference]", "[policy]", "[verification]"))

_CONTENT = {"harmonic": (HARMONIC_CONFIG, HARMONIC_STRUCTURE),
            "harmonic-nvt": (HARMONIC_NVT_CONFIG, HARMONIC_STRUCTURE)}


def write_template(template: str, output_dir: str | Path, *,
                   force: bool = False) -> Path:
    """Write ``run.toml`` + ``structure.extxyz`` for ``template``; returns
    the configuration path.  Existing files are kept unless ``force``."""
    if template not in TEMPLATES:
        raise WorkflowError(
            f"unknown template {template!r}; available: {list(TEMPLATES)}")
    output_dir = Path(output_dir)
    config_text, structure_text = _CONTENT[template]
    config_path = output_dir / "run.toml"
    structure_path = output_dir / "structure.extxyz"
    existing = [p for p in (config_path, structure_path) if p.exists()]
    if existing and not force:
        raise WorkflowError(
            f"{output_dir} already contains {', '.join(p.name for p in existing)}; "
            "pass --force to overwrite or choose a new --output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")
    structure_path.write_text(structure_text, encoding="utf-8")
    return config_path
