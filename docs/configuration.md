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
  `--force`. Variants: `harmonic-nvt` (plain NVT),
  `harmonic-adaptive-nvt` (adaptive Langevin NVT with a fixed base model)
  and `harmonic-mts` (experimental fixed-model MTS NVE; the written
  structure carries fixed initial momenta).
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
- `pyramid resume RUN_DIR --steps N [--force-unlock]
  [--resource BACKEND.ROLE=PATH]`: continue a plain, adaptive or MTS MD run
  for N *additional* steps (for MTS, N counts inner steps and must be a
  multiple of `outer_ratio`); the current and target step numbers are
  printed. Settings come from the run's `resolved_config.json` — same
  physics, model chain and check stream. `--force-unlock` reclaims the
  writer lock left behind by a crashed process (use only when no live
  writer exists). `--resource` (repeatable) rebinds a declared file
  resource after the run was relocated; see "Resume semantics".
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
  `surrogate` (use the frozen surrogate), `mts` (experimental fixed-model
  multiple time stepping, NVE only — see *task.mode = "mts"* below). The
  `reference`/`surrogate` modes apply to singlepoint, relaxation and plain
  MD tasks, NVE or NVT.

Mode rules:

- `adaptive` requires `[reference]`, `[surrogate]` and `[policy]`, and only
  applies to `kind = "md"` — the energetic calculator is never placed
  under an optimizer, because it does not define a fixed potential surface.
- `reference` / `surrogate` require their own backend section and reject
  the other backend section plus `[policy]` and `[verification]` (unused
  sections are errors, not silently ignored).
- `mts` requires `kind = "md"`, both `[reference]` and `[surrogate]`, and
  `dynamics.integrator = "respa"`; it rejects `[policy]` and
  `[verification]` (the energetic policy and independent checks exist only
  in adaptive MD).

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
- `integrator`: `verlet` (default, NVE), `langevin` (required for NVT) or
  `respa` (fixed-model MTS; pairs exactly with `task.mode = "mts"`).
- `timestep_fs` (number > 0, required). With `respa` this is the INNER
  step.
- `steps` (integer >= 1, required). With `respa` it counts INNER steps and
  must be a multiple of `outer_ratio` — only complete outer steps exist.
- `outer_ratio` (integer >= 1): required with `integrator = "respa"` — the
  number of inner steps per outer step; refused with any other integrator.
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
- `max_wall_hours` (number > 0, optional): a SOFT per-process walltime
  budget for plain serial MD.  The clock anchors at the workflow entry —
  backend construction and the initial evaluation count against it — and
  the driver stops only at a complete-step boundary: when the remaining
  budget falls below the reserve for one more step (1.5x the last
  measured step wall plus a 300 s I/O margin; before any measurement, the
  backend's declared per-attempt timeout times its configured retries
  when available, else just the margin), the last complete boundary is
  checkpointed regardless of the interval and the run ends
  `stopped_early` with the budget numbers on the `run_end` event.  A
  running step is never interrupted, and a sudden slow step is not
  predicted by the estimate: this is a between-step scheduler, never a
  guarantee that an in-flight SCF returns before the wall (the engine's
  `timeout_s` bounds attempts; the scheduler's hard wall is external).  A
  resumed process re-arms the budget, so a job chain can hand a long
  trajectory across scheduler walls.  Adaptive MD and serial-recipe
  stages refuse the option explicitly.

NVT runs use ASE's Langevin with `fixcm=False` (the deprecated
`fixcm=True` does not strictly sample the correct NVT distribution;
FixCom is not a supported constraint). Adaptive mode supports
`ensemble = "nvt"` with `integrator = "langevin"`: the configured (TOML)
workflow always uses a fixed base model — no configuration key installs an
update callback.  Guarded online updates exist only in the Python
interface (`GuardedUpdater` + `UpdatePolicy` via
`EnergeticRunner(on_label=...)`), in both ensembles; resuming a consuming
run requires a stateful updater.

### task.mode = "mts" (experimental fixed-model MTS)

New in 0.8.0 as an experimental feature. Fixed-model multiple time stepping
for NVE: the slow residual `F_reference − F_fast` between the reference and
the fast potential is applied as symmetric outer half-kicks (r-RESPA) around
`outer_ratio` inner velocity-Verlet steps on the fast force, so the
reference is evaluated once per complete outer step. Both models are
frozen for the whole run — there are no anchors, policies or checks; those
belong to adaptive mode, which decides reference calls from a force-error
policy and is a different tool, not a replaced one.

Configuration:

- `task.kind = "md"`, `task.mode = "mts"`, `ensemble = "nve"` and
  `integrator = "respa"` are required together — the mode pairs exactly
  with the integrator, and either setting alone is an error.
- `timestep_fs` is the INNER step. `steps` counts inner steps and must be
  a multiple of `outer_ratio` — only complete outer steps exist, never a
  silent rounding.
- Both `[reference]` and `[surrogate]` are required (MTS integrates the
  slow residual between them); `[policy]` and `[verification]` are
  rejected.
- The structure must carry momenta: this mode continues a state and never
  thermalizes one, so a structure without velocities is refused before the
  first evaluation and `temperature_K`/`velocity_seed` never apply.

Refused up front, never silently ignored:

- NVT and NPT: `nvt` requires `integrator = "langevin"`, which conflicts
  with the required `respa` pairing; `npt` is not an ensemble choice.
- Constraints of any kind (from `[constraints]` or carried by the
  structure): no constraint algorithm is applied in this version, and
  ignoring them would integrate the wrong dynamics.
- Online model updates (the model is fixed; no configuration key installs
  an update callback in any mode) and adaptive step sizes (no such field
  exists — unknown fields are rejected).
- Backends that do not declare a content fingerprint, or that do not
  explicitly declare `force_consistent`, conservative forces and a known
  `energy_kind` — all checked before the first evaluation.

Run records, checkpoints and resume:

- One record is committed per COMPLETE outer boundary: the initial
  evaluation at step -1, then outer boundary k at inner step
  `k * outer_ratio`. A successful run of `steps = n * outer_ratio` commits
  n + 1 rows and makes exactly n + 1 reference evaluations and
  `steps + 1` surrogate predictions; rows carry the reference and
  surrogate boundary labels separately, with `driving` absent.
- `checkpoint.interval_steps` counts inner steps and checkpoints land only
  on complete outer boundaries; a completed or stopped run always leaves a
  checkpoint at the last committed boundary regardless of the interval.
- Readout and export adopt complete boundaries only: `pyramid inspect`'s
  `last_step`/`physical_time` and `pyramid export`'s frames reflect the
  last complete outer boundary. A committed tail evaluation whose boundary
  never completed (a crash mid-step) is listed separately as
  `last_evaluation` with `complete = false`, and the reference/inference
  calls it already paid for stay in the cost ledger.
- Each side keeps its own declared energy convention end to end: the
  boundary labels carry each backend's declared `energy_kind` and
  force-consistency verbatim into the store, the checkpoint and the
  exported frames. A changed declared convention refuses resume before any
  evaluation.
- `pyramid inspect` adds an MTS block (`inner_timestep_fs`,
  `outer_ratio`, `complete_outer_steps`, `complete_inner_steps`,
  `physical_time_fs`); the human rendering gains a progress line, e.g.
  `mts progress          : 32 complete outer steps = 128 inner steps
  (128.0 fs physical time)`.
- `pyramid export` has no `driving` source for MTS — an outer step has no
  single driving force. Export `--force-source reference` (reference
  boundary labels) or `--force-source base` (fast-potential predictions).
- `pyramid resume RUN_DIR --steps N` adds N INNER steps; N must be a
  multiple of `outer_ratio`. Resume continues the same integration
  settings (`timestep_fs`, `outer_ratio`), reference identity, model
  identity and declared energy conventions from the checkpoint — a
  mismatch is refused before anything is evaluated. Resource relocation
  (`--resource`) does not compose with MTS; resume in place.

Try it offline: `pyramid init --template harmonic-mts --output demo` writes
a complete analytic demo (128 inner steps of 1 fs, `outer_ratio = 4`); the
same demo lives in the source tree at
[examples/mts_nve/](../examples/mts_nve/).

### [reference] / [surrogate]

- `backend` (string, required): a registered name — `pyramid backends`
  lists them. Builtin: `qe`, `qe-ase`, `pyscf` (engines), `mace`
  (surrogate), the frozen correction wrappers `scaled` /
  `quadratic-corrected` (surrogates, see below), and the analytic toys
  `harmonic-reference` / `harmonic-surrogate`.
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

#### Correction wrappers (`scaled` / `quadratic-corrected`, surrogates)

Frozen, conservative corrections wrapped around a base surrogate — no
retraining.  `scaled`: `U_c = c U_b`, `F_c = c F_b` with one frozen
positive scalar.  `quadratic-corrected`: the static quadratic Taylor
correction of the reference-minus-base difference at one fixed center
`q0` (`U_c = U_b - dF0·u + 1/2 uᵀ dH u + E0`, `F_c = F_b + dF0 - dH u`),
with `delta_h` symmetrized and translation-projected at construction
(the acoustic sum rule on the correction matrix; the residuals are
recorded, and a net `delta_f0` force is recorded but never silently
removed).  Both wrappers declare no stress and never impersonate the
base model's.

- `base`: a `{"name", "kwargs"}` spec of the base surrogate backend;
  path-like values inside `kwargs` resolve against the configuration
  file's directory.
- `quadratic-corrected` additionally requires `species` (the fixed atom
  order and elements, validated on every prediction) and the correction
  content — inline `q0`/`delta_f0`/`delta_h`, or `parameters_npz` naming
  one `.npz` file with exactly those arrays (mutually exclusive; the
  file's path and content digest are recorded as provenance, never
  fingerprinted — the values are).  `energy_offset` anchors the corrected
  energy zero at `q0` (a constant changes no forces).
- The periodic chart is a fixed local coordinate domain: the pbc mask
  and cell are recorded at the first prediction (a non-periodic first
  call included) and every later call must match — a pbc change in
  either direction or a cell change is refused.  `q0` and the input
  positions must live in the same continuous coordinate representation:
  on every periodic axis the raw fractional displacement
  `(q - q0) @ cell⁻¹` must be strictly inside the half-cell around `q0`;
  reaching or crossing the boundary raises `CorrectionDomainError` for
  the upper layer — the run stops with the refusal recorded.  A genuine
  domain exit is a model problem, not a representation problem:
  translating `q0` and the positions together keeps `q - q0` fixed and
  cannot recover an out-of-domain point; continue only from correction
  parameters regenerated around a NEW calibration center (a new model),
  or stop and hand the refusal to the caller.  There is no automatic
  relocation API.  The wrapper never rounds, never applies a minimum
  image and never accepts an integer-offset shift, so no discontinuous
  energy/force pair is ever returned under a conservative declaration.
  Nothing depends on call history or process lifetime beyond the
  recorded environment, so a fresh-process resume reproduces identical
  corrections under an identical fingerprint.  NPT/variable-cell and
  general unwrapping are not supported.
- The correction content, `species`, `energy_offset`, the calibration
  note and the base identity all enter the fingerprint; the frozen
  parameters are read-only (a resume with edited correction bytes refuses
  on the identity mismatch).
- `uncertainty` forwarded by the wrappers is the BASE model's spread,
  unchanged: no recalibrated confidence of the corrected potential
  exists (the wrappers never recalibrate).

A minimal plain surrogate-MD configuration with a quadratic correction
loaded from an `.npz` (parameters of one H-O dimer `params.npz` holding
arrays `q0` (N,3), `delta_f0` (N,3), `delta_h` (3N,3N)):

```toml
[surrogate]
backend = "quadratic-corrected"
species = ["H", "O"]           # the fixed atom order of structure file
parameters_npz = "params.npz"  # relative to this configuration file
energy_offset = 0.0
[surrogate.base]
name = "harmonic-surrogate"
[surrogate.base.kwargs]
k = 1.0
r0 = 0.9
bias = 0.05
```

#### QE backends (`qe` / `qe-ase`)

Both QE paths share one `QeConfig`, so their options are identical; the
execution knobs below change how pw.x runs, never the physical recipe
(they are recorded in `resolved_config.json` but excluded from the
reference fingerprint).

- `pw_cmd` (string or list of strings, default `"pw.x"`): how pw.x is
  launched.  One shared argv contract everywhere — configuration
  validation, `--check-environment` and the executor all see the same
  normalized form.  A TOML list is literal argv, verbatim (`pw_cmd =
  ["mpirun", "-np", "4", "pw.x"]`; each element stays one argv word,
  including paths with spaces or parentheses).  A string naming an
  existing file is that literal path; any other string is split with
  POSIX `shlex` rules — quote a path containing spaces
  (`pw_cmd = "'/opt/QE 7.5/pw.x' -nk 2"`).  Nothing is shell-expanded:
  the executors never invoke a shell implicitly, so shell operators
  (`|`, `>`, `&&`, `$(...)`) would have no effect; such forms are not
  usable launch specifications, and `--check-environment` reports them
  `unverified` (the outer executable itself missing is `blocked`).
  For `qe-ase`, an explicit `command`
  string (ASE's FileIO layer) overrides `pw_cmd`; the preflight checks
  exactly the effective one.
- `disk_io` (string, optional): QE's own `disk_io` (INPUT_PW, QE 7.5).
  One of `high` / `medium` / `low` / `nowf` / `minimal` / `none`; an
  unsupported value fails at validation, before any SCF.  Absent (TOML)
  or `None` (Python) keeps QE's default — that is not the string
  `"none"`.  `nowf` still writes the XML and the converged charge
  density, so the *next* SCF can start from the density, but skips the
  wavefunction files (tens of GB on large cells with HDF5 builds).
  `minimal` writes only the XML; `none` writes neither — with those,
  warm-start chaining honestly reports that no reusable density exists.
  Note the distinction: `nowf` supports *starting the next SCF from the
  saved density*; it is not an in-place resume of an interrupted SCF.
  Both QE charge-density formats are recognized on warm starts and in
  density provenance: `charge-density.dat`, and `charge-density.hdf5`
  from HDF5 builds.  A density staged as warm-start *input* is never
  mislabelled as the attempt's own output: an attempt running
  `minimal`/`none` claims no produced density (its manifest keeps the
  true input origin), and the chain then falls back to the real source.
- `startpot_file` / `density_source`: warm starts from a verified
  density of known origin (see the engine docstrings).
- Validation has three explicit scopes:
  `pyramid validate run.toml` checks the configuration only (never the
  machine); `pyramid validate run.toml --check-environment` adds a
  read-only, zero-computation preflight of local runtime prerequisites
  (the effective `pw_cmd`/`command` argv[0] resolution — a direct
  executable with arguments passes statically, a missing one is
  `blocked`, launchers such as `srun`/`mpirun` and shell-style
  compositions stay `unverified` — pseudopotential file presence,
  optional packages probed via import metadata without importing, local
  model files, and wrapper backends checked through their declared base
  backend; unknown plugin backends report `unverified`, never `ready`);
  and `--probe-backends` explicitly
  evaluates the structure once per backend.  `--json` emits one
  machine-readable report object for any of the three scopes, including
  a usage conflict between `--probe-backends` and `--check-environment`.
- `startingwfc_file` (boolean, default `false`): also restart
  wavefunctions from a staged `.save` tree (`startingwfc = 'file'` is
  written only when this attempt actually staged one; the attempt record
  and ledger carry the decision with its reason, and the only recorded
  read observation comes from QE's own stdout — receipt-level
  information, never an authorization input).  Refused clearly when the
  combination cannot work: `disk_io = "nowf"`/`"minimal"`/`"none"`
  (those write no wavefunction files), the `[density] persist` registry
  (its published seed pack carries no wavefunctions), and the `qe-ase`
  adapter (not implemented there).  The wavefunction-restart read path is
  pinned by a regression fixture recorded from a real QE 7.5 native
  manual complete stage (`tests/data/qe75_warm_start_wfc_read.out`);
  the persistent density pack and its deletion permissions are
  not expanded for it.
- `density_source_policy` (`latest` default, or `fixed`): the
  density-chain trial order.  `latest` reuses the most recent successful
  density of the run and treats `density_source` as initialization (first
  evaluation, fresh process) plus fallback when the latest density is
  unusable — a continuous MD chain no longer re-seeds from the initial
  geometry every step.  `fixed` keeps the pre-0.7.2 order (the configured
  source wins every evaluation).  A failed or non-converged attempt never
  becomes the latest density, and an attempt whose `disk_io` wrote no
  charge density is never claimed as one.  New runs record the effective
  policy in `resolved_config.json`; runs saved by 0.7.1 or earlier carry
  no policy and resume with the legacy `fixed` order, so an old run's
  semantics never change under a new binary.  A resumed process
  re-initializes from `density_source` (recorded honestly in the density
  decision); recovering the previous process's latest density from
  persisted restart resources is a separate, later feature.

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
reference-direct — the existing refusal path, with the gate unchanged.
While a wait is active the current step's reference driving force is the
ordinary refusal formula, but future anchors and routes may differ from a
non-pacing run: pacing can change the trajectory within the same error
governance and does not promise an identical one.  Off by default; an
absent section (or `enabled = false`) is exactly the pre-0.6 behavior,
and setting tuning fields while disabled is an error.  It composes with
neither online model updates nor explicit direction callbacks (both are
refused up front).

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

- `interval_steps` (integer >= 1, default 10). With `task.mode = "mts"` it
  counts inner steps and checkpoints land only on complete outer
  boundaries.
- `keep_generations`: only 2 is accepted; the runtime keeps exactly two
  generations.

### [scratch] (opt-in unified temporary root, 0.7.1)

One unified managed tmp root for solver scratch, with a
durably-archived-then-reclaimed lifecycle.  Absent this section every
attempt behaves exactly as before (per-run `calculations/` scratch).

- `root` (path, required when the section is present): the unified tmp
  root; a relative path resolves against the configuration file's
  directory.  Every adapted attempt runs in its exclusive
  `tmp/<run-uuid>/<backend-role>/<request-uuid>/<attempt-id>/`
  directory underneath; the shared root is never recursively deleted.
- `retention` (`all` default, or `results`): `all` keeps the attempt's
  scratch on the unified root; `results` archives the verified result
  out of scratch and reclaims the attempt's subtree immediately — in
  this stage for standalone SCF; `results` with
  `startpot_file`/`density_source` is refused up front (a chained
  density lives in the scratch it would reclaim).
- Lifecycle records live outside the root in `scratch_records/`
  (authoritative, atomically updated); results and `pw.in`/`pw.out`
  stay in the run's persistent directories, so `inspect`/`export` never
  depend on reclaimed scratch.  Currently adapted backends: `qe` and
  `qe-ase`; others keep their existing behavior.  Inspect or retry a
  root with `pyramid scratch inspect --root PATH` and
  `pyramid scratch clean --root PATH [--dry-run]`.

### [density] (opt-in persistent density chain)

`persist = true` (the only field) wires a run-owned, persistent density
chain for one workflow shape: a plain serial reference MD run
(`task.kind = "md"`, `task.mode = "reference"`) on the `qe` or `qe-ase`
backend.  Every other combination — singlepoint, relax, surrogate or
adaptive modes, non-QE backends, serial-recipe stage configurations, and
`scratch.retention = "results"` — is refused at load/run time with the
supported scope named.  Absent the section (or `persist = false`) every
behavior is exactly as before, and old resolved configurations keep
their exact recorded identity.

- The run directory owns `restart/density/`: every successful evaluation
  that produced a new charge density publishes it as one immutable,
  content-verified generation; the attempt record notes the outcome, and
  a publication failure never breaks the delivered label.
- With `startpot_file = true` in `[reference]`, the next evaluation
  warm-starts from the registry's latest verified generation (the
  default `density_source_policy = "latest"` order: registry first, then
  the previous in-memory/configured sources); `density_source_policy =
  "fixed"` never consults the registry for the source — the configured
  external `density_source` wins every evaluation, is never deleted, and
  publications still land in the registry.  Without `startpot_file` the
  chain is save-only.
- Committed evaluations and checkpoints record the generation the run
  state actually depends on.  `resume` binds exactly the generation of
  the one authoritative restored boundary — the committed evaluation
  whose row supplied the restored positions/forces (a crash-window heal
  binds the healed evaluation's own record); only a zero-step resume,
  which restores the checkpoint's own arrays, reads the checkpoint field.
  A record that declares no reference (explicit null) or predates the
  field (legacy) initializes externally; a referenced generation that is
  missing or corrupt refuses the resume before anything is written or
  computed — never a silent swap to a newer generation.  The `resumed`
  event records the actual boundary evaluation and the bound density
  generation and content digest.
- With `[scratch]` (`retention = "all"`, as the persistent chain requires), an
  attempt's scratch is released only after a LATER ordinary calculation
  has independently read that exact published seed and succeeded — and
  "read" is proven by the consuming attempt's own raw output (QE's
  `The initial density is read from file` marker naming its staged save
  tree) together with the launch input's `startingpot = 'file'` and the
  archived output's digest.  The consumed seed's generation, content
  digest and reference settings are pinned at selection time and must
  agree across the consumer's pin, the producer's persisted receipt and
  the live registry manifest before anything is released — a
  contradiction or a missing identity keeps the producer with the reason
  named, and a damaged receipt is never rewritten to match.  A verified
  publication alone is not a deletion credential, and neither is an
  atomic fallback, a cache hit, a borrow from the producer's own tree,
  or a successful run whose output carries no read marker: a silent or
  unknown output format preserves every source with the reason
  recorded.  The producer attempt carries a
  pending release receipt on its authoritative record until then; an
  unproven terminal seed keeps its scratch (a protected resource, not a
  cleanup failure), as do failed publications and failed consumers.  A
  committed evaluation that produced no density releases its scratch
  directly.
- Old generations are reclaimed by the driver after each step commit and
  checkpoint retention update — and on demand through
  `pyraimd2.runtime.restart.execute_density_reclaim` (dry-run first:
  `plan_density_reclaim` reports per-generation keep/hold/reclaim reasons).
  To see the same decision picture from the shell without writing
  anything, run `pyramid density inspect RUN_DIR` (read-only JSON: every
  generation's keep / reclaim-candidate / hold decision with its concrete
  reasons, blocked references and the space report).  It is a snapshot of
  the current registry — `reclaim_candidate` there is a preview, not a
  deletion authorization, and the space report claims no bound on the
  run's whole-disk usage.  Only generations this run fully owns — once attached, validated,
  unreferenced — are deleted: the latest pointer, retained checkpoints'
  references, the committed recoverable boundary, in-flight inputs and
  unconsumed producer seeds are always kept; publish leftovers, corrupt or
  never-attached directories and anything not fully owned stay.  Each
  deletion writes a durable tombstone first, so an interrupted or repeated
  clean is resumable — a deletion stopped mid-tree completes only while a
  readable manifest still CONFIRMS ownership (directory name, durable
  tombstone and run_root/run_id must agree; payload already deleted needs
  no re-validation), while a re-referenced tombstone stays suspended, a
  manifest contradicting ownership (a foreign same-number tree) is held
  forever, and a missing or unreadable manifest is never resumed — and a
  reclaimed generation is never mistaken for corruption (its number is
  never reused).  A plan computed before the run moved on is refused as
  stale.

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
- MTS runs resume from the last complete-outer-boundary checkpoint;
  `--steps N` counts inner steps and must be a multiple of `outer_ratio`.
  Integration settings, backend identities and declared energy conventions
  are verified against the checkpoint before any evaluation.

### Relocating a run with declared file resources

A run whose backends declare file resources (the backend factory's
`file_parameters={"<option>": "<role>"}` declaration) can be **relocated**
— moved together with those resource files and resumed at the new
location:

```sh
pyramid resume "moved run" --steps 2 \
    --resource 'reference.potential=moved run/inputs/model.dat'
```

- `--resource BACKEND.ROLE=PATH` is repeatable and maps a baseline-declared
  `<section>.<role>` key to the file's new location; a missing `=`
  separator, an empty key, an empty path or a duplicated role is a usage
  error before anything runs.
- A relative PATH resolves against the **calling working directory**,
  never the run directory; the Python API
  (`resume_workflow(..., resource_paths=...)`) takes absolute paths.
- Every current file is re-read and compared byte-for-byte (SHA-256)
  against the run's baseline before any backend is built — a same-named
  file with different content is refused, as are unknown roles. Only the
  declared option slots are rebound, in memory: the run's config copy,
  manifest, baseline and history are never rewritten; each verified
  binding appends one receipt under `resource_bindings/` (a record, not a
  new trust baseline). The mapping is never remembered — every later
  restart while the recorded path is stale needs it again.
- Supported for a fixed model (byte-identical file at a new location) in
  the same environment (same pyraimd2 and backend code), for runs created
  with declared file resources; older runs without a baseline are never
  upgraded, and relocation does not compose with an online updater.
  The full walkthrough is
  [../examples/file_model_relocation/](../examples/file_model_relocation/).

## Export and missing data

`--force-source` states which forces the frames carry:

- `driving`: the forces that actually propagated the MD (present for plain
  and adaptive MD; refused for MTS runs, where an outer step has no single
  driving force — export `reference` or `base`).
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
