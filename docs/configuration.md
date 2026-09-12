# Pyramid configuration and CLI

Pyramid runs material simulations from a small TOML file. The same
configuration drives the command line and the Python API — the CLI holds no
logic of its own.

- Units are part of the field names and fixed everywhere: positions Å,
  energies eV, forces eV/Å, time fs, temperature K.
- Relative paths in the configuration resolve against the **configuration
  file's directory**, so results land in the same place no matter where the
  command runs. Directory names containing spaces work.
- Unknown sections and fields are rejected with the dotted field name and a
  remedy — never silently ignored. This is what makes schema migrations
  safe (see *Schema migration* below).

## Five minutes

```sh
pyramid init --template harmonic --output my_run
pyramid validate my_run/run.toml
pyramid run my_run/run.toml
pyramid inspect my_run/runs/harmonic-demo
pyramid resume my_run/runs/harmonic-demo --steps 20
pyramid export my_run/runs/harmonic-demo --force-source driving --output traj.extxyz
```

The `harmonic` template uses builtin analytic backends and runs offline in
seconds while exercising anchoring, surrogate acceptance, independent
checks, checkpoints, stop and resume. `Ctrl-C` during `run` stops at the
next complete-step boundary, writes a checkpoint, and the run continues
with `resume`.

The same flow from Python:

```python
from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow, resume_workflow

config = load_config("my_run/run.toml")
run_workflow(config)                       # identical code path to `pyramid run`
resume_workflow(config.run.directory, 20)  # identical to `pyramid resume`
```

## Commands

- `pyramid --version` / `pyramid --help`: work offline; no backend is
  imported and nothing is downloaded.
- `pyramid backends`: list registered backend factories (builtin and
  entry-point plugins) with their declared kind and origin.
- `pyramid init --template harmonic --output DIR [--force]`: write
  `run.toml` + `structure.extxyz`. Existing template files are kept unless
  `--force`. Variants: `harmonic-nvt` (plain NVT) and
  `harmonic-adaptive-nvt` (adaptive Langevin NVT with a fixed base model).
- `pyramid validate CONFIG [--probe-backends]`: check everything that can
  be checked without running: TOML and schema, structure readability and
  sanity, path-like backend options (model files, pseudo directories,
  pseudopotentials), backend construction through the registry (parameter
  validation at the factory), declared capability compatibility, run-id
  collisions. Default validation runs **no** SCF, no inference and
  downloads nothing. `--probe-backends` additionally evaluates the
  structure once with each configured backend (this can execute external
  programs — QE needs `pw.x`, a periodic cell and pseudopotentials).
- `pyramid run CONFIG`: validate, then execute. Refuses an already-used run
  directory (one directory per run; continue with `resume`, never by
  appending).
- `pyramid resume RUN_DIR --steps N [--force-unlock]`: continue an adaptive
  or plain MD run for N *additional* steps; the current and target step numbers are
  printed. Settings come from the run's `resolved_config.json` — same
  physics, model chain and check stream. `--force-unlock` reclaims the
  writer lock left behind by a crashed process (use only when no live
  writer exists).
- `pyramid inspect RUN_DIR [--json]`: status, costs and checks. The JSON
  rendering and the human rendering come from the same structured source.
- `pyramid export RUN_DIR [--force-source driving|reference|base]
  [--output PATH] [--force]`: export the committed trajectory as extxyz.

Exit codes: 0 success, 1 run-time failure (the run directory keeps the
failure record and the cost ledger), 2 usage/configuration error, 130
interrupted.

## Configuration reference

`schema_version = 1` is required semantics (a missing value defaults to 1);
any other version is rejected — see *Schema migration*.

### [run]

- `id` (string, required): run identity inside its directory; no path
  separators.
- `directory` (path, required): the run directory, resolved relative to the
  configuration file.
- `seed` (integer, default 0): base seed. `dynamics.velocity_seed`,
  `dynamics.thermostat_seed` and `verification.seed` default to it; set
  them explicitly to decouple the streams. Plain NVT derives its
  velocity-initialization and bath streams from these seeds by a fixed
  role convention (`role-derive-v1`), so the two never share a generator
  even when both fields default to the same value; the effective stream
  identities are recorded in the run's RUN_START event (`streams` block).
  The independent-check stream exists only in adaptive mode.

### [task]

- `kind`: `singlepoint` (one backend evaluation), `relax` (fixed-model
  optimization with ASE FIRE/BFGS), `md` (plain NVE or NVT dynamics).
- `mode`: `adaptive` (energetic MD: anchored surrogate forces with
  independent reference checks), `reference` (use the reference engine),
  `surrogate` (use the frozen surrogate). The latter two modes apply to
  singlepoint, relaxation and plain MD tasks, NVE or NVT.

Mode rules:

- `adaptive` requires `[reference]`, `[surrogate]` and `[policy]`, and only
  applies to `kind = "md"` — the energetic calculator is never placed
  under an optimizer, because it does not define a fixed potential surface.
- `reference` / `surrogate` require their own backend section and reject
  the other backend section plus `[policy]` and `[verification]` (unused
  sections are errors, not silently ignored).

### [relax]

- `optimizer`: `fire` (default) or `bfgs` — ASE optimizers.
- `fmax_eV_A` (number > 0, default 0.05): convergence threshold on the max
  per-atom force.
- `steps` (integer >= 1, default 200): optimizer step cap.

### [constraints]

- `fix_atoms_indices` (list of nonnegative integer atom indices): merges
  with FixAtoms the structure itself carries (e.g. POSCAR selective
  dynamics) into one fixed set. Every other constraint kind (RATTLE,
  energy-carrying or moving constraints) and any variable-cell dynamics is
  rejected explicitly. Fixed coordinates never move; the force budget
  applies to the free coordinates by default (`policy.force_metric =
  "active_dofs_max_atom"`), with raw forces, the projected driving force
  and the actual constrained displacement recorded per evaluation;
  `"all_atoms_max_atom"` is the explicit alternative.

### [structure]

- `file` (path, required): any structure ASE reads (extxyz, CIF, POSCAR,
  ...), resolved relative to the configuration file. Velocities in the file
  are kept; without them, momenta are thermalized at
  `dynamics.temperature_K` with `dynamics.velocity_seed`. FixAtoms
  constraints are supported (see `[constraints]`).

### [dynamics]

- `ensemble`: `nve` or `nvt`.
- `integrator`: `verlet` (default, NVE) or `langevin` (required for NVT).
- `timestep_fs` (number > 0, required).
- `steps` (integer >= 1, required).
- `temperature_K` (number >= 0, default 300.0): the bath target
  temperature for NVT, and the one-time velocity-initialization default
  when the structure carries no velocities (both modes). Supplied
  velocities/momenta may represent a different initial temperature; the
  bath still targets `temperature_K`. There is no separate initialization
  temperature field.
- `velocity_seed` (integer, default `run.seed`): seeds the one-time
  velocity initialization only; it never reseeds a resume.
- `friction_per_fs` (number > 0): required for NVT — the bath coupling
  (`friction = friction_per_fs / ase.units.fs` internally). An NVE run
  must not set it.
- `thermostat_seed` (integer): seeds the thermostat's dedicated NumPy
  `Generator` for a new NVT run (default `run.seed`). The thermostat
  stream is separate from the independent-check stream; a resume always
  restores the persisted stream and never uses this seed again. Changing
  algorithm, timestep, temperature or friction means a new run, not a
  resume.

NVT runs use ASE's Langevin with `fixcm=False` (the deprecated
`fixcm=True` does not strictly sample the correct NVT distribution;
FixCom is not a supported constraint). Adaptive mode supports
`ensemble = "nvt"` with `integrator = "langevin"`: the configured (TOML)
workflow always uses a fixed base model — no configuration key installs an
update callback.  Guarded online updates exist only in the Python
interface (`GuardedUpdater` + `UpdatePolicy` via
`EnergeticRunner(on_label=...)`), in both ensembles; resuming a consuming
run requires a stateful updater.

### [reference] / [surrogate]

- `backend` (string, required): a registered name — `pyramid backends`
  lists them. Builtin: `qe`, `qe-ase`, `pyscf` (engines), `mace`
  (surrogate), and the analytic toys `harmonic-reference` /
  `harmonic-surrogate`.
- Every other key is passed to the backend factory verbatim as a keyword
  option; unknown options fail at the factory with the option named.
- Path-like option values (keys ending in `_path`, `_file`, `_dir`, plus
  `model`, `density_source`, and `pseudos` filenames under `pseudo_dir`)
  resolve relative to the configuration file and are existence-checked at
  validate time. A MACE `model` that is a bare word (e.g. `small`) stays a
  foundation-model name; one that looks like a path must exist.
- Factories that declare `run_root` (QE) receive the run's `calculations/`
  directory from the workflow; factories that declare `event_log` receive
  the run's event log. Configuration files never name these themselves.
- Missing optional dependencies are reported at selection time with the
  matching extra (`pip install 'pyraimd2[mace]'` / `[pyscf]`).

### [policy] (adaptive only)

- `name`: only `energetic`.
- `force_budget_eV_A` (number > 0, required): the acceptance budget.
- `probe_steps_A` (two increasing positive numbers, default [0.02, 0.04]).
- `numerical_floor_eV_A` (number >= 0, default 0.0).
- `time_cap_fs` (number > 0, default 1.0).
- `transverse_cap` (number in [0, 1], default 0.1).
- `force_metric` (`active_dofs_max_atom` default, or
  `all_atoms_max_atom`): what the force budget measures. With FixAtoms,
  the default counts only the free coordinates (the raw all-atom residual
  stays recorded as a diagnostic); the all-atom metric is an explicit
  alternative, never a silent redefinition.

### [policy.calibration_pacing] (adaptive only; opt-in, 0.6 prototype)

Cost-aware calibration pacing: after `failure_streak_limit` consecutive
calibrations each failed to produce an accept, the probe investment of
the next recalibrations is deferred and those steps drive
reference-direct — the existing refusal path, with unchanged driving
forces and an unchanged gate.  Off by default; an absent section (or
`enabled = false`) is exactly the pre-0.6 behavior, and setting tuning
fields while disabled is an error.  It composes with neither online model
updates nor explicit direction callbacks (both are refused up front).

- `enabled` (boolean, default false).
- `failure_streak_limit` (integer >= 1, default 3): consecutive sterile
  calibrations before the first deferral.
- `wait_initial` (integer >= 1, default 1): opportunities skipped before
  the first forced retry calibration.
- `wait_max` (integer >= wait_initial, default 8): the backoff cap —
  failure rounds wait 1×, 2×, ... up to this bound, so recalibration is
  retried at a bounded interval, never suppressed permanently.

Every refused evaluation's decision (calibrate / defer, with the reason
and the remaining wait) is recorded as a `pacing_decision` event and is
visible under `pyramid inspect`.  Reference executions are always billed
by what actually ran; a deferral is visible as the ABSENCE of probes plus
a recorded decision, never as an invented saving.

### [verification] (adaptive only)

Independent checks over accepted evaluations; the segment parameters are
fixed for the run (change them with a fork, not a resume).

- `probability` (number in [0, 1], default 0.05): `0` disables checks.
  Setting `failure_probability` or `tilt` together with `probability = 0`
  is an error — a disabled segment carries no pretend parameters.
- `failure_probability` (number in (0, 1), default 0.05).
- `tilt` (number > 0, default ln 2).
- `seed` (integer, default `run.seed`).

### [checkpoint]

- `interval_steps` (integer >= 1, default 10).
- `keep_generations`: only 2 is accepted; the runtime keeps exactly two
  generations.

### [output]

Derived, regenerable views over the authoritative store:

- `trajectory_interval_steps` (integer >= 1, default 1): the run
  directory's `trajectory.extxyz` preview keeps every k-th committed
  evaluation (driving forces).
- `summary_interval_steps` (integer >= 1, default 10): `summary.json` /
  `summary.csv` are refreshed every k steps and always at run end.

`trajectory.db` and `events.jsonl` always record every committed
evaluation — they are the recovery record and are never thinned.

## Run directory layout

- `config.toml`: the original configuration, verbatim.
- `resolved_config.json`: every parameter actually in effect, absolute
  paths, units, schema version. `resume` rebuilds from this file.
- `manifest.json`: run identity, config/structure hashes, backend
  identities (fingerprints), software and Python versions.
- `trajectory.db`: ASE database of committed evaluations (base/reference/
  driving payloads, metadata).
- `events.jsonl`: the authoritative event log (single writer, locked).
- `checkpoints/<generation>/`: complete-step snapshots + `latest.json`.
- `models/<model_id>/`: immutable model artifacts for stateful updaters.
- `calculations/`: per-evaluation external calculation directories (QE).
- `summary.json`, `summary.csv`, `trajectory.extxyz`: derived outputs —
  safe to regenerate via `inspect`/`export` at any time.

## Resume semantics

- `resume --steps N` always means N **additional** steps; the current and
  target step numbers are printed before anything runs.
- Resume continues the same physics, model chain and check stream from the
  last valid checkpoint plus event replay. Reference/surrogate
  identity mismatches are refused; changing settings means a new run (or a
  library-level `fork`).
- Plain reference/surrogate runs resume from their complete-step
  checkpoints: the plain driver checkpoints at every
  `checkpoint.interval_steps` and on a stop request, and resume rebuilds
  the boundary from the last committed step. `export` works on them too.

## Export and missing data

`--force-source` states which forces the frames carry:

- `driving`: the forces that actually propagated the MD (always present).
- `reference`: reference engine labels. Evaluations without one (accepted,
  unchecked steps) are **missing data, not zero force**: the frame carries
  `forces_available=F` and an all-NaN forces array, and no `energy` key.
  Labeled frames also record `reference_label_id`.
- `base`: uncorrected surrogate predictions (same missing-data rule).

Every frame records `run_id`, `step_id`, `evaluation_id`,
`physical_time_fs`, `route` and `force_source`, so an exported file stays
interpretable without the event log. ASE reads labeled frames back with a
SinglePointCalculator (`get_forces()` / `get_potential_energy()`).

## Schema migration

- The configuration carries `schema_version`; this pyraimd2 reads version 1
  and rejects anything else with a pointer here.
- Unknown sections and fields are rejected. Use the fields documented above
  when preparing a version-1 configuration; no automatic migration from
  other schema versions is provided.
- `resolved_config.json` records the schema version next to the effective
  parameters, so old run directories stay interpretable.

## Current limitations

- Constraints: FixAtoms only — RATTLE/holonomic, energy-carrying and
  moving constraints are rejected explicitly, as is any variable-cell
  (NPT) dynamics.
- Released 0.4.2 supports NVE only. Version 0.5.0 adds
  fixed-cell NVT (ASE Langevin): plain reference/surrogate modes, and
  adaptive mode in both ensembles — fixed base model via TOML/CLI, or
  guarded online updates via the Python `GuardedUpdater` interface
  (re-anchoring supported; the update transaction publishes or rolls back
  atomically, and resume never retrains a committed update).
- `checkpoint.keep_generations` is fixed at 2 by the runtime.
- Model updates (online training) use the Python `GuardedUpdater` interface;
  resumable updates require state export and restore. Adaptive runs without
  an updater, including the supplied TOML examples, use a frozen surrogate.

## Examples

- [Harmonic adaptive MD](../examples/harmonic_adaptive/README.md): the runnable
  offline demo (same content as the `harmonic` init template).
- [QE + MACE skeleton](../examples/qe_mace_skeleton/README.md): configuration
  structure for a periodic material, with external inputs required before a run.
- [Bulk Si](../examples/si_bulk_qe_mace/README.md) and
  [Al(111) slab](../examples/al_surface_qe_mace/README.md): structure generators,
  input requirements, singlepoint/relaxation and short NVE examples.
- [Slurm submission](../examples/slurm/README.md): a generic batch template.
