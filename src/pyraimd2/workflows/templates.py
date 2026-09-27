"""Init templates: complete, runnable configuration + structure pairs.

``harmonic`` is the offline demo: analytic builtin backends, a small H2O
molecule, and policy/verification settings that exercise anchoring,
acceptance, checks, checkpoints and resume in a few seconds without any
external program.  ``harmonic-nvt`` is the plain NVT variant;
``harmonic-adaptive-nvt`` runs the same toy adaptively under Langevin
dynamics (fixed base model, re-anchoring supported).  ``harmonic-mts`` is
the experimental fixed-model MTS (respa) NVE demo: two analytic test
particles with fixed initial momenta, 128 inner steps of 1 fs with
outer_ratio 4 — no external program, no model download.
``harmonic-compare`` is the offline accuracy-verification chain: one
reference NVE run plus two fixed-model MTS candidates (unscaled and
scaled) from one shared initial structure, then `pyramid compare` on the
completed runs.  The numbers are
demonstration values, not accuracy recommendations for any material.
"""

from __future__ import annotations

from pathlib import Path

from pyraimd2.workflows.setup import WorkflowError

TEMPLATES = ("harmonic", "harmonic-nvt", "harmonic-adaptive-nvt",
             "harmonic-mts", "harmonic-compare")

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

HARMONIC_ADAPTIVE_NVT_CONFIG = HARMONIC_CONFIG.replace(
    '# Pyramid configuration — harmonic offline demo (schema_version 1).',
    '# Pyramid configuration — harmonic offline adaptive-NVT demo\n'
    '# (schema_version 1): a fixed surrogate model drives Langevin dynamics\n'
    '# under the energetic force-error policy with anchoring and independent\n'
    '# reference checks.  The base model is frozen for the whole run\n'
    '# (online training is not available for adaptive NVT); re-anchoring is\n'
    '# supported and recorded per segment.'
).replace(
    'id = "harmonic-demo"', 'id = "harmonic-adaptive-nvt-demo"'
).replace(
    'directory = "runs/harmonic-demo"',
    'directory = "runs/harmonic-adaptive-nvt-demo"'
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
velocity_seed = 7""")

HARMONIC_MTS_CONFIG = """\
# Pyramid configuration — harmonic MTS offline demo (schema_version 1).
# Experimental fixed-model MTS (respa) NVE on the builtin analytic harmonic
# backends — no external program, no model download.  The two Si-labelled
# atoms are test particles in a toy well, not a real material.
# task.mode "mts" integrates the slow residual F_ref - F_fast with symmetric
# outer kicks around outer_ratio inner velocity-Verlet steps; timestep_fs is
# the INNER step and steps count inner steps: this file simulates 128 fs as
# 32 complete outer steps.
schema_version = 1

[run]
id = "harmonic-mts-demo"
directory = "runs/harmonic-mts-demo"

[task]
kind = "md"                       # singlepoint | relax | md
mode = "mts"                      # reference | surrogate | adaptive | mts

[structure]
file = "structure.extxyz"         # carries fixed initial momenta (never re-randomized)

[dynamics]
ensemble = "nve"
integrator = "respa"
timestep_fs = 1.0                 # inner step (fs)
steps = 128                       # inner steps (128 fs); must be a multiple of outer_ratio
outer_ratio = 4                   # inner steps per outer step

[checkpoint]
interval_steps = 8                # inner steps; only complete outer boundaries are checkpointed

[reference]
backend = "harmonic-reference"    # builtin analytic reference
k = 1.0
r0 = 0.9

[surrogate]
backend = "harmonic-surrogate"    # same well at a softer stiffness, no bias
k = 0.9
r0 = 0.9
bias = 0.0
"""

# The exact structure examples/mts_nve/make_structure.py writes: two Si test
# particles (mass 28.085 amu, non-periodic) at center (0.9, 0.9, 0.9) A +/-
# (0.03, -0.02, 0.01) A, atom 2 velocity (0.001, -0.0015, 0.002) A/fs and
# atom 1 the opposite, momenta in ASE units — a fixed initial state, never
# re-randomized.
HARMONIC_MTS_STRUCTURE = """\
2
Properties=species:S:1:pos:R:3:masses:R:1:momenta:R:3 pbc="F F F"
Si       0.87000000       0.92000000       0.89000000      28.08500000      -0.28591950       0.42887925      -0.57183900
Si       0.93000000       0.88000000       0.91000000      28.08500000       0.28591950      -0.42887925       0.57183900
"""

# ---------------------------------------------------------------------------
# harmonic-compare: the offline accuracy-verification chain — one plain
# reference NVE run plus two fixed-model MTS candidates (unscaled and
# `scaled`-wrapped) from ONE shared initial structure, then `pyramid
# compare`.  Same physics as examples/compare_runs/: harmonic reference
# k = 1.0, fast model k = 0.8 (bias 0), scale 1.2 applied to E AND F
# together, inner step 1 fs, 16 inner steps, outer_ratio 4.

HARMONIC_COMPARE_REFERENCE_CONFIG = """\
# Pyramid configuration — harmonic-compare demo (schema_version 1),
# REFERENCE arm: plain reference-driven NVE on the builtin analytic
# harmonic backend — no external program, no model download.  The two
# Si-labelled atoms are test particles in a toy well, not a real material.
# 16 steps of 1 fs: committed complete states at every fs, t = 0 .. 16 fs.
# Relative paths resolve against THIS file's directory.
schema_version = 1

[run]
id = "harmonic-compare-reference"
directory = "runs/reference"

[task]
kind = "md"                       # singlepoint | relax | md
mode = "reference"                # plain NVE driven by the reference engine

[structure]
file = "structure.extxyz"         # carries fixed initial momenta (never re-randomized)

[dynamics]
ensemble = "nve"
integrator = "verlet"
timestep_fs = 1.0                 # fs
steps = 16                        # 16 fs

[checkpoint]
interval_steps = 8

[reference]
backend = "harmonic-reference"    # builtin analytic reference
k = 1.0
r0 = 0.9
"""

HARMONIC_COMPARE_MTS_CONFIG = """\
# Pyramid configuration — harmonic-compare demo (schema_version 1),
# CANDIDATE arm 1: fixed-model MTS (respa) NVE with the plain harmonic
# fast model (k = 0.8 against the reference k = 1.0 — a nonzero residual).
# timestep_fs is the INNER step and steps count inner steps: 16 fs as 4
# complete outer steps of 4 inner steps each, so complete boundaries land
# at t = 0, 4, 8, 12, 16 fs — a subset of the reference arm's 1 fs grid.
# Relative paths resolve against THIS file's directory.
schema_version = 1

[run]
id = "harmonic-compare-mts"
directory = "runs/mts"

[task]
kind = "md"
mode = "mts"

[structure]
file = "structure.extxyz"         # the SAME initial state as the reference arm

[dynamics]
ensemble = "nve"
integrator = "respa"
timestep_fs = 1.0                 # inner step (fs)
steps = 16                        # inner steps (16 fs); must be a multiple of outer_ratio
outer_ratio = 4                   # inner steps per outer step

[checkpoint]
interval_steps = 8                # inner steps; only complete outer boundaries are checkpointed

[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9

[surrogate]
backend = "harmonic-surrogate"    # same well at a softer stiffness, no bias
k = 0.8
r0 = 0.9
bias = 0.0
"""

HARMONIC_COMPARE_MTS_SCALED_CONFIG = """\
# Pyramid configuration — harmonic-compare demo (schema_version 1),
# CANDIDATE arm 2: the same MTS run with the public `scaled` wrapper around
# the same fast model.  `scaled` multiplies BOTH the energy and the forces
# by one frozen scalar (scaling only one side would break force
# consistency).  scale = 1.2 makes the used fast field 0.96 of the
# reference — a small, nonzero residual.  1.2 is a GIVEN teaching
# coefficient here, not fitted and not a recommendation; a real scale must
# be determined in advance on independent calibration data and frozen.
# Relative paths resolve against THIS file's directory.
schema_version = 1

[run]
id = "harmonic-compare-mts-scaled"
directory = "runs/mts-scaled"

[task]
kind = "md"
mode = "mts"

[structure]
file = "structure.extxyz"         # the SAME initial state as the reference arm

[dynamics]
ensemble = "nve"
integrator = "respa"
timestep_fs = 1.0                 # inner step (fs)
steps = 16                        # inner steps (16 fs); must be a multiple of outer_ratio
outer_ratio = 4                   # inner steps per outer step

[checkpoint]
interval_steps = 8                # inner steps; only complete outer boundaries are checkpointed

[reference]
backend = "harmonic-reference"
k = 1.0
r0 = 0.9

[surrogate]
backend = "scaled"
scale = 1.2                       # frozen for the whole run; scales E AND F
base = { name = "harmonic-surrogate", kwargs = { k = 0.8, r0 = 0.9, bias = 0.0 } }
"""

HARMONIC_COMPARE_README = """\
# Harmonic compare demo — verify a cheap trajectory against a reference

Three configurations around ONE shared initial structure (two Si-labelled
test particles in an analytic harmonic well — not a real material): a
plain reference-driven NVE run (`run.toml`, 16 steps of 1 fs, states at
every fs) and two fixed-model MTS candidates (`run_mts.toml` unscaled,
`run_mts_scaled.toml` with the `scaled` wrapper; inner step 1 fs,
outer_ratio 4, complete outer boundaries at t = 0, 4, 8, 12, 16 fs).
Everything runs offline on the builtin analytic backends.

## 1. Validate the three configurations

```sh
pyramid validate run.toml
pyramid validate run_mts.toml
pyramid validate run_mts_scaled.toml
```

`pyramid validate` checks schema, structure, paths and the backend
contract without evaluating anything.  Two optional scopes answer
different questions: `--check-environment` is a strictly read-only check
of local prerequisites (executables, optional packages, model files —
never executes anything), while `--probe-backends` performs one REAL
evaluation per configured backend (here cheap and analytic; with a real
DFT engine it would launch it — use it only where compute is authorized).

## 2. Run the three trajectories

```sh
pyramid run run.toml           # -> runs/reference/
pyramid run run_mts.toml       # -> runs/mts/
pyramid run run_mts_scaled.toml
```

## 3. Compare each candidate against the reference

```sh
pyramid compare runs/reference runs/mts
pyramid compare runs/reference runs/mts-scaled --json
```

Each candidate carries 5 complete states (t = 0, 4, 8, 12, 16 fs) and each
of them must match a unique reference time point exactly (1e-9 fs, no
interpolation); the reference's 17 complete states may be denser.  The
comparison is read-only: nothing is evaluated or written into the run
directories.

## 4. Reading the report

- **position / velocity RMS**: pointwise in the stored continuous
  coordinates (no alignment), normalized per atom; per-time arrays plus
  the whole-window max (`--json`).
- **H drift**: per-trajectory Hamiltonian drift `(H(t) - H(0)) / N`,
  zeroed at each run's own start — descriptive only, never compared
  across runs' absolute energy zeros.
- **coverage**: committed evaluations vs complete states; an incomplete
  tail (a crashed final step) is reported, never counted as a point.
- **thresholds**: `--max-position-rms A` and `--max-velocity-rms A_PER_FS`
  add per-criterion pass/fail (exit 0/1).  Without explicit thresholds
  the tool reports metrics only and never declares a run "reliable".

The 1.2 scale is a given teaching coefficient — not fitted, not a
recommendation; a real scale must be determined beforehand on independent
calibration data and frozen.  This is an analytic operations demo of the
verify chain (init -> validate -> run -> compare), not evidence about any
material, and it makes no performance claims.
"""

_CONTENT = {"harmonic": {"run.toml": HARMONIC_CONFIG,
                         "structure.extxyz": HARMONIC_STRUCTURE},
            "harmonic-nvt": {"run.toml": HARMONIC_NVT_CONFIG,
                             "structure.extxyz": HARMONIC_STRUCTURE},
            "harmonic-adaptive-nvt": {
                "run.toml": HARMONIC_ADAPTIVE_NVT_CONFIG,
                "structure.extxyz": HARMONIC_STRUCTURE},
            "harmonic-mts": {"run.toml": HARMONIC_MTS_CONFIG,
                             "structure.extxyz": HARMONIC_MTS_STRUCTURE},
            "harmonic-compare": {
                "run.toml": HARMONIC_COMPARE_REFERENCE_CONFIG,
                "run_mts.toml": HARMONIC_COMPARE_MTS_CONFIG,
                "run_mts_scaled.toml": HARMONIC_COMPARE_MTS_SCALED_CONFIG,
                "structure.extxyz": HARMONIC_MTS_STRUCTURE,
                "README.md": HARMONIC_COMPARE_README}}

#: every template's primary configuration (the ``write_template`` return)
PRIMARY_CONFIG = "run.toml"


def write_template(template: str, output_dir: str | Path, *,
                   force: bool = False) -> Path:
    """Write the template's files into ``output_dir``; returns the primary
    configuration path (``run.toml``).  Existing files are kept unless
    ``force`` — without it, if ANY of the template's targets exists,
    nothing is written at all; with it, exactly the template's own files
    are overwritten (unrelated files and existing run data are never
    touched)."""
    if template not in TEMPLATES:
        raise WorkflowError(
            f"unknown template {template!r}; available: {list(TEMPLATES)}")
    output_dir = Path(output_dir)
    files = _CONTENT[template]
    existing = [output_dir / name for name in files
                if (output_dir / name).exists()]
    if existing and not force:
        raise WorkflowError(
            f"{output_dir} already contains {', '.join(p.name for p in existing)}; "
            "pass --force to overwrite or choose a new --output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    return output_dir / PRIMARY_CONFIG
