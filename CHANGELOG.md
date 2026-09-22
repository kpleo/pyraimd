# Changelog

## 0.7.5 (unreleased candidate)

Usability candidate on top of 0.7.4:

- `pyramid validate` now states explicitly that a successful run checks
  the configuration only — the environment is not checked.
- New `pyramid validate CONFIG --check-environment`: a read-only,
  zero-computation preflight of local runtime prerequisites (executable
  resolution for direct `pw_cmd`, pseudopotential file presence,
  optional-package availability probed without importing, local model
  file presence; wrapper/launcher commands and unconfirmable caches are
  reported `unverified`, never silently ready).  Exit status 0 only when
  every check this version knows passes; `blocked`/`unverified` exit 1.
- New `--json` for validate (all three scopes: configuration /
  environment / probe) emitting one machine-readable report object;
  handled configuration errors are reported as JSON as well.
- New `pyramid density inspect RUN_DIR`: read-only JSON view of the
  persistent density registry's keep / reclaim-candidate / hold
  decisions, blocked references and space report (plain serial
  reference MD scope; preview only, never a deletion authorization).
- Preflight hardening after independent review: wrapper backends
  (`scaled` / `quadratic-corrected`) are checked through their declared
  base backend; `pw_cmd` follows one shared argv contract (literal argv
  list, or POSIX `shlex` string parsing, never a shell — `~` is not
  expanded); `qe-ase` checks the effective `command`/`pw_cmd` priority;
  explicit shell wrappers stay `unverified`; run-directory occupancy is
  refused by presence alone with zero files created or modified (no
  SQLite connection, no WAL/SHM side files); the validate mode conflict
  reports one parseable error object under `--json`.

## 0.7.4 (2026-09-19)

Stable integration release on top of 0.7.3, absorbing the reviewed
upstream-contributed improvements that fit the general software (same
scope and guarantees as the 0.7.3 line documents).

### Added

- Frozen conservative correction wrappers around a base surrogate
  (`scaled` / `quadratic-corrected` backends): `U_c = c U_b, F_c = c F_b`
  with one frozen scalar, or the static quadratic Taylor correction of
  the reference-minus-base difference at a fixed center with the
  correction Hessian symmetrized and translation-projected (acoustic sum
  rule; net `delta_f0` recorded, never silently removed).  Corrections
  are value-fingerprinted with the required fixed atom order/elements
  (`species`); frozen parameters are read-only; correction content loads
  inline or from one `.npz` (`parameters_npz`, provenance recorded but
  never fingerprinted).  The periodic chart is a fixed atlas: the
  environment (pbc mask + cell) is recorded at the first prediction and
  verified unchanged on every call; positions and `q0` must share one
  continuous representation and stay strictly inside the half-cell
  around `q0` on periodic axes (no rounding, no minimum image, no
  integer-shift exception) — reaching or crossing the boundary is a
  controlled refusal for the upper layer, so a fresh-process resume
  reproduces identical corrections.  Stress is never impersonated from
  the base model.
- `dynamics.max_wall_hours` (plain serial MD only): a soft per-process
  walltime budget anchored at the workflow entry (initialization counts),
  stopping only at complete-step boundaries with a checkpoint written
  regardless of the interval; a resumed process re-arms it.  The reserve
  for one more step follows the last measured step wall (or the backend's
  declared timeout/retries before any measurement), never a hardcoded
  ceiling.  Adaptive MD and recipe stages refuse the option explicitly.
- `QeConfig.startingwfc_file` / `[reference] startingwfc_file`: an
  explicit, default-off wavefunction-restart knob — the input is written
  only when the attempt staged a `.save` tree, the receipt records the
  decision with its reason, and the only recorded read observation is
  parsed from QE's own stdout.  The read path is pinned by a regression
  fixture recorded from a real QE 7.5 native manual complete stage
  (`tests/data/qe75_warm_start_wfc_read.out`); the observation stays
  receipt-level and never authorizes anything.  Refused for
  `disk_io` modes without wavefunction files, the `[density] persist`
  registry combination (its seed pack carries no wavefunctions), and the
  `qe-ase` adapter.

### Notes

- A directory-based checkpoint density-chain resume was proposed
  upstream and is deliberately not integrated here: superseded by
  0.7.3's run-owned generation registry with committed-boundary binding,
  actual-read gating and ownership-checked reclaim (its acceptance
  scenarios are covered by the existing density-chain tests with
  stricter refusal semantics).

## 0.7.3 — 2026-09-17

An opt-in persistent QE density chain for plain serial reference MD:
every evaluation's charge density is published into a run-owned registry
as one immutable, content-addressed generation. With file starts and the
latest-source policy enabled, later calculations warm-start from the
published seed. Verified independent consumption, fixed checkpoint
retention and successful reclaim allow old seed generations and consumed
producer scratch to be removed (issue #7). Logs and metadata still grow.
Runs that leave
`[density]` unset behave exactly as 0.7.2.

### Added

- `[density] persist = true` (plain serial reference MD, `qe` and
  `qe-ase` backends, requires `[scratch] retention = "all"`): publish
  every produced charge density into `run/restart/density/` (charge
  density + schema XML, plus the PAW `paw.txt` when present), bind
  commits and checkpoints to the exact generation the state depends on,
  and warm-start following evaluations when `startpot_file = true` and
  `density_source_policy = "latest"` are set. Without file starts,
  publication is save-only; fixed external-source mode retains its
  configured source.
  Other workflow kinds and recipe/adaptive/surrogate combinations are
  refused at configuration time.
- Resume binds the one authoritative restored boundary: the committed
  evaluation whose row supplied the restored positions/forces decides
  the density reference (normal tail, crash-window heal and checkpoint
  fallback share the rule; only a zero-step resume, which restores the
  checkpoint's own arrays, reads the checkpoint field).  An explicit
  null means external initialization; a legacy record without the field
  keeps the old behavior; a referenced generation that is missing or
  corrupt refuses the resume before anything is written or computed —
  never a silent swap to a newer generation.  The `resumed` event
  records the actual boundary evaluation and the bound density
  generation and content digest.
- Delayed producer release: an attempt's scratch is released only after
  a later ordinary calculation independently read that exact published
  seed and succeeded, proven by the consuming attempt's own raw output —
  QE's `The initial density is read from file` marker naming its staged
  save tree, bound to the launch input (`startingpot = 'file'`) and the
  archived output digest — and only while the consumed seed's identity
  (generation, content digest, reference settings) agrees across the
  consumer's selection-time pin, the producer's persisted pending
  receipt and the live registry manifest; a contradiction or missing
  identity preserves the producer, and a damaged receipt is never
  repaired into an approval.  A verified publication alone, an atomic
  fallback, a cache hit, a borrow from the producer's own tree, or a
  successful run whose output reports no read never qualify: a silent or
  unknown output format preserves every producer with the reason
  recorded.  The producer attempt carries a persistent pending/consumed
  receipt; an unproven terminal seed keeps its scratch as a protected
  resource.
- Old-generation reclaim: `plan_density_reclaim` (dry-run with
  per-generation keep/hold/reclaim reasons) and
  `execute_density_reclaim` (run lock held, references re-read inside
  it, stale plans refused).  Only generations the run fully owns — once
  attached, validated, unreferenced — are deleted; the latest pointer,
  retained checkpoints' references, the committed recoverable boundary,
  in-flight inputs and unconsumed producer seeds are always kept, and
  anything foreign, external, corrupt, never-attached or leftover is
  never auto-deleted.  Every deletion is preceded by a durable tombstone
  in the registry state, so interrupted or repeated cleans are resumable
  — a deletion stopped mid-tree completes only while a readable manifest
  still confirms ownership (directory name, tombstone and run_root/run_id
  must agree; already-deleted payload needs no re-validation); a
  same-number foreign tree is held forever, and a missing manifest is
  never resumed — and reclaimed generations are never mistaken for
  corruption — their numbers are never reused.  The serial driver
  reclaims automatically after each step commit and checkpoint retention
  update.
- `examples/density_persist_qe/`: a documented minimal configuration and
  a fake-`pw.x` end-to-end demo (run, fresh-process resume, receipts,
  tombstones, dry-run and real reclaim).

### Compatibility and verified scope

- Opt-in only; runs without `[density] persist` are byte-for-byte the
  0.7.2 behavior.  Chains recorded by the development previews resume
  under the rules above.- Verified scope: QE 7.5 (HDF5 build), PAW, non-spin-polarized SCF
  restarts, exercised end to end on Si8 (fresh-process resume bound the
  actual boundary generation; the boundary seed's producer was released
  only after the first post-resume evaluation independently consumed
  that exact seed; old generations were reclaimed under tombstones).
  Fixed-retention fake-QE runs at 6/20/60 steps show the persistent seed
  set plateauing (latest + retained-checkpoint references + boundary +
  one unconsumed producer) while lightweight logs, per-attempt archives
  and registry metadata grow with the step count — no
  constant-whole-disk claim.  Other QE builds/formats/profiles are not
  claimed: an unproven profile keeps its producer scratch until a later
  calculation proves the seed independently sufficient.

### Packaging and upgrade

- The release ships `pyraimd2-0.7.3-py3-none-any.whl` and
  `pyraimd2-0.7.3.tar.gz`, built with `uv build` (the `uv_build`
  backend).  Upgrading from 0.7.2 is a plain reinstall into a clean
  virtual environment (`pip install pyraimd2-0.7.3-py3-none-any.whl`);
  existing run directories keep working under the same configuration
  files, and the wheel contains only the published package (no tests,
  examples or development files).

## 0.7.2

Four fixes from real adoption feedback, on both pw.x paths (subprocess
`QeEngine` and ASE-QE).  Configurations and engines that do not use the
new options behave exactly as 0.7.1, with one deliberate exception:
QE density chains now default to reusing the latest successful density
(see below), and runs saved by 0.7.1 or earlier resume with the legacy
order they were recorded with.

### Added

- `QeConfig.disk_io` / `disk_io = "..."` in `[reference]`/`[surrogate]`:
  QE's own write knob (INPUT_PW, QE 7.5 — `high`/`medium`/`low`/`nowf`/
  `minimal`/`none`), passed through verbatim by both QE paths and
  validated at construction, before any launch.  `nowf` keeps the
  converged charge density (next-SCF warm starts keep working) while
  skipping wavefunction files; `minimal`/`none` write no reusable
  density, and the density manifest then says so honestly instead of
  claiming a density that was never saved.  An execution knob: recorded
  in the resolved configuration, excluded from the reference fingerprint.
- `QeConfig.density_source_policy` / `density_source_policy = "..."`:
  the density-chain trial order, shared by both QE adapters.  `"latest"`
  (the new default) reuses the most recent successful density and treats
  `density_source` as initialization/fallback; `"fixed"` is the 0.7.1
  order.  New runs record the effective policy in
  `resolved_config.json`; older resolved records resume as `"fixed"`.

### Fixed

- `[scratch]` settings now reach the engines used by new single-point,
  plain MD and adaptive MD runs, matching resumed runs. Previously these
  entry points bypassed the shared configuration merge, so temporary
  files could remain in per-attempt directories despite a managed root.

- Continuous QE chains configured with an external `density_source`
  re-seeded every evaluation from the initial geometry; they now follow
  the latest successful density (issue #4).  A failed or non-converged
  attempt is never promoted, and a fresh process records honestly that
  it re-initialized from the configured source.
- Density claims follow what a run actually produced: both QE charge
  density formats are recognized (`charge-density.dat` and
  `charge-density.hdf5` from HDF5 builds — one shared rule on the
  production, manifest and loading sides of both adapters), and a staged
  warm-start input copy is never mislabelled as the attempt's own
  output.  With `disk_io = "minimal"`/`"none"` an attempt claims no
  produced density (QE writes none in these modes); the manifest keeps
  the true input origin and the chain falls back to the real source.
- Legacy recipe stages saved before `density_source_policy` existed
  adopt correctly again: the stored record's missing field is completed
  in memory with the fixed order the run provably used (the file is
  never rewritten), an unchanged stage continues with that legacy
  order, and an explicit policy change or any other changed reference
  setting is still refused.
- Model/UPF content identity no longer comes from a `(path, mtime,
  size)` cache: identical stat tuples across a same-size rewrite
  (coarse-timestamp filesystems, preserved mtimes, atomic replacement)
  used to serve a stale digest.  The identity is always computed from
  the file's current bytes, on both the UPF and the ASE model-file
  paths, and a transient unreadable state is never cached (issue #6).
  Note: on the generic undeclared ASE adapter path this re-reads the
  model file at each fingerprint read (a few times per adaptive
  evaluation); the declared `file_parameters` resource path identifies
  at construction and re-verifies content at the documented boundaries.
- `test_sigint_stops_at_a_step_boundary_and_resume_continues` signalled
  on a fixed delay after the event log appeared and could fire before
  the first step on a loaded machine; it now waits (bounded) for the
  first committed `step_completed` event — the real "runner owns the
  handler and a step boundary exists" condition.

## 0.7.1

A unified, managed temporary root for solver scratch with a
durably-archived-then-reclaimed lifecycle — labels no longer leave
wavefunction scratch accumulating behind them.  Everything is opt-in:
a configuration or engine without the new options behaves exactly as
0.7.0.

### Added

- `[scratch] root = "...", retention = "..."` (workflow) and
  `QeConfig(scratch_root=..., retention=...)`: one unified temporary
  root (a relative root resolves against the configuration file's
  directory), with every adapted attempt running in its exclusive
  `tmp/<run-uuid>/<backend-role>/<request-uuid>/<attempt-id>/`
  directory.  `retention = "all"` (default) keeps the attempt's
  scratch; `retention = "results"` archives the verified result out of
  scratch and reclaims the attempt's subtree immediately — currently
  supported for standalone SCF and refused up front in combination with
  `startpot_file` / `density_source`.  Both QE adapters (subprocess
  QeEngine and ASE-QE) share the one lifecycle; other backends keep
  their existing behavior.
- `pyraimd2.runtime.scratch`: the small lifecycle manager — allocation,
  durable archival (fsynced copies out of the scratch root), atomic
  state records outside the root (`scratch_records/`), idempotent
  reclaim with ownership and symlink guards, and `inspect_root` /
  `clean_pending` (dry-run first) for one view and safe retries.  The
  shared root is never recursively deleted; failed or unarchived
  attempts are always kept.
- `pyramid scratch inspect --root PATH` and
  `pyramid scratch clean --root PATH [--dry-run]`: the same manager
  from the CLI.
- The standalone QE label template (`examples/standalone_qe_label/`)
  now runs on the lifecycle: archived result files in the persistent
  run root, the attempt's scratch reclaimed, and the state printed.

## 0.7.0

Declared, content-verified file resources for ASE backends, and safe
relocation of runs that use them — resume a run that was moved together
with its resource files, from Python or the CLI.  No defaults change: a
run without declared file resources behaves exactly as 0.6.0.

### Added

- Declared file resources (`AseEngine`/`AseSurrogate`
  `file_parameters={"<option>": "<role>"}`): the content of a declared
  file parameter joins the backend's physical identity (full SHA-256,
  re-read at construction and at every fingerprint), so a changed file
  is a different backend rather than a silently different force field.
  Each run records an immutable `file_resources.json` baseline (schema,
  per-resource role/parameter/original path/content digest) and every
  checkpoint carries the baseline's digest; a resume verifies the
  baseline before any backend is rebuilt — missing, corrupt or
  mismatched baselines refuse.
- Relocation resume: `resume_workflow(..., resource_paths=...)` (Python)
  and `pyramid resume RUN_DIR --steps N --resource BACKEND.ROLE=PATH`
  (repeatable; a relative PATH resolves against the caller's working
  directory) rebind declared resources after the run was moved together
  with those files.  Only baseline-declared keys are accepted; every
  current file is re-verified byte-for-byte before any computation;
  only the declared option slots are rebound, in memory, without
  rewriting the run's history; each verified binding appends a receipt
  under `resource_bindings/`.  Supported for a fixed model
  (byte-identical file) in the same environment for runs created with
  declared resources; older runs are never upgraded and relocation does
  not compose with an online updater.  New example:
  `examples/file_model_relocation` (plus the
  `examples/backends/pyraimd2_filemodel` plugin) walks through fresh run
  → relocate → refused resume → mapped resume → export in two processes.
- `Path`-valued declared file parameters are accepted without
  conversion (identity and computation read the path as-is).

### Fixed

- The resume help and API reference now state the actual plain
  reference/surrogate and fixed-model adaptive resume support.

## 0.6.0

Opt-in calibration pacing for adaptive MD, with resumable rule state and
outcome-accurate reporting.  No defaults change: a run without the new
section is byte-identical to 0.5.0.

### Added

- `[policy.calibration_pacing]` (`enabled`, `failure_streak_limit`,
  `wait_initial`, `wait_max`; default off): after that many consecutive
  calibrations each failed to produce an accept, the probe investment of
  the next recalibrations is deferred and those steps drive
  reference-direct through the existing refusal path.  The wait retries
  on a bounded doubling backoff and is cancelled immediately when a
  refused step's retained-correction error exceeds the run's own force
  budget.  Every refused evaluation records a keyed `pacing_decision`
  event (calibrate / defer, reason, remaining wait) before any probe
  spend; `pyramid inspect` separates completed, deferred, unavailable
  and pending calibration outcomes.  Scope: fixed base model, fixed
  cell, the default single-direction NVE/NVT paths; it composes with
  neither online model updates nor explicit direction callbacks (both
  are refused up front).  New example: `examples/calibration_pacing`
  (analytic, core-only) demonstrates enabling, the decision log, stop
  and continue, the on/off cost comparison and the offline error check.
  See `docs/configuration.md`.

### Fixed

- A committed calibration's pending segment state is committed with the
  evaluation (shared by the row, the commit event and the live state),
  so a hard exit after the commit and before the next decision resumes
  with exactly the continuous run's decision order and rule state.
- The pacing section of `inspect` no longer counts planned calibrations
  as completed; a frozen decision whose evaluation never committed is
  reported pending.

## 0.5.0

### Added

- Plain fixed-cell NVT MD (ASE Langevin, `fixcm=False`) in reference and
  surrogate modes, with complete-step checkpoints and resume of the
  thermostat's random stream. `dynamics`
  gains `ensemble`, `integrator`, `friction_per_fs` and `thermostat_seed`;
  `temperature_K` is both the bath target and the one-time
  velocity-initialization default when the structure has no velocities.
- Adaptive MD supports `ensemble = "nvt"` (Langevin) with a fixed base
  model: the decision consumes the actual constraint-processed
  configuration of each stochastic step, the bath's random quantities are
  drawn once per step and recorded for exact resume, and the independent
  check stream is role-derived (`role-derive-v1`) so it never shares a
  generator with the bath or velocity streams. Re-anchoring is supported
  and recorded per segment. New template: `harmonic-adaptive-nvt`.
- Guarded online updates under adaptive NVT through the Python
  `GuardedUpdater` interface (NVE and NVT): the label-consumption/update
  transaction keeps its atomic publish-or-rollback contract across process
  crashes — a committed evaluation whose consumption never persisted is
  re-delivered to the restored updater exactly once, and committed
  training is never re-executed.  New runnable example:
  `examples/guarded_nvt.py` (fresh run, stop, new-process resume,
  purpose-split cost report).
- The serial `relax → NVT → NVE` recipe: `load_completed_state(run_dir)`
  reads the authoritative completed boundary (MD: the last true
  STEP_COMPLETED record, verified against its committed row by the
  record's own digest format; relax: the converged terminal state), and
  `run_serial_recipe` drives the stages with per-stage run identities, a
  short `workflow.json` manifest, durable stage identity persisted before
  the first computation, finished stages adopted without recomputation,
  and crashed MD stages resumed to their configured totals.  Examples:
  `examples/periodic_lj` (offline plugin) and `examples/si_bulk_qe_mace`
  (QE reference + frozen MACE surrogate).

### Fixed

- A stage's source identity (the parent boundary digest and the
  materialized input structure) is persisted before the stage's first
  computation — a hard exit mid-stage no longer leaves `source: null` in
  the manifest — and a finished stage is adopted only while its bound
  source, materialized input and completion facts still match the
  authoritative records: an upstream run extended outside the recipe
  refuses with the existing results preserved, instead of presenting the
  stale chain as current.
- A stop request received on a stage's final step now ends the recipe
  invocation before the next stage starts (the finished stage is still
  persisted done first), with a concise stopped / steps-completed /
  how-to-continue record instead of a misleading failure traceback; the
  next explicit invocation continues, and the stop check is
  invocation-scoped so a stale stopped RUN_END cannot poison a completed
  resume. The plain MD driver now records a stop received on its final
  step, matching the adaptive runner.
- Stage wall-time completeness is judged by invocation pairing
  (RUN_START / RESUMED against RUN_SUMMARY) with the reason recorded: a
  hard-killed invocation leaves its unrecorded time marked unknown rather
  than letting the recorded resume leg pose as the complete total.
- QE process launches write durable `attempt_receipt` phases (prepared /
  started from the actual process-creation fact / not_launched) under a
  stable attempt identity. The ledger, `inspect` and the recipe manifest
  distinguish confirmed executions, confirmed successes and unresolved
  attempts — including staging directories no event accounts for — and
  report whether the cost record is complete, so an empty `pw.out` is no
  longer read as a completed SCF nor silently dropped from the total.
- The Si recipe guide documents the exact prepare → edit → validate → run
  ordering with a `--prepare-only` mode (no computation, inference or
  downloads), the precise deferred-validate scope, and how to probe a
  materialized stage input through a standalone probe config with a new
  run id/directory.
- An abandoned event log (a constructor or a failed resume dropping the
  writer without closing it) no longer leaks an OS handle: garbage
  collection releases the handle while the lock file stays as crash
  evidence, since lock removal remains a deliberate `close()`.
- The ASE-native QE attempt span is settled at its terminal exit, so the
  density-manifest write is counted once in the declared
  staging/process/validation span instead of being frozen out.
- Plain MD resume verifies each persisted step-summary format by its own
  semantics (0.4.x records carry none; the previous development batch's
  complete-boundary JSON digest; 182cc8d's unmarked dual digest whose
  boundary digest never covered the bath stream; the current array digest
  plus boundary
  digest covering the thermostat stream) instead of rejecting older
  records as corrupt, and crash-healed step records now carry the same
  boundary digest as normal commits.
- Stores opened by the MD workflows are closed on every controlled
  failure path (setup, resume, adaptive), not left to garbage collection.
- A saved-but-uncommitted model candidate is completed on resume (after
  validating the parent model, source label IDs and recipe bindings)
  instead of retraining into an immutable-artifact conflict; explicit
  user direction callbacks keep precedence over the deferred
  calibration's persisted displacement.

### Changed

- The NVT statistics acceptance now identifies the ASE Langevin propagator
  as Vanden-Eijnden–Ciccotti (not BAOAB) and carries its measured
  timestep bias (+0.020% position variance, -0.042% mean kinetic energy at
  dt=0.5 fs, from a deterministic discrete-Lyapunov covariance check)
  instead of claiming zero bias; sampling tolerances are derived from the
  actual discrete process at the run's budget rather than a fixed 7% gate.

## 0.4.2

Second correctness patch on top of 0.4.1. No new features.

### Fixed

- Nested ASE mixing calculators (Sum of Sum of ...) fingerprint their full
  composition; a wrapper whose children are unidentifiable reports an honest
  unknown instead of a trusted empty-parameter hash.
- Attempt sinks are checked for identity: a run log is connected for the
  duration of each call and restored afterwards, mismatched sinks are refused
  before launch, and the no-log direct API keeps working.
- The cost ledger counts every launched attempt once, with failed, killed and
  post_processing_failed all counted as failed physical executions.
- Native QE output-read failures surface as contractual errors with the
  original cause and a terminal attempt record, never UnboundLocalError.
- Plain-MD failed requests close exactly one task record under one id,
  including the initial evaluation.
- Calibration counters are restored only for pre-proposal deferred
  calibrations, keeping segment history [1,2,3] across resume.
- All user-visible trajectory outputs (CLI export, automatic trajectory,
  summary CSV, low-level frames API) read through the same committed-row view;
  orphaned rows stay as audit records only.

### Changed

- numpy is held below 2.5: ASE 3.29.0 sets array shapes through an API
  deprecated in NumPy 2.5, and the pin removes that compatibility noise
  instead of suppressing warnings.

## 0.4.1

Correctness fixes for resume, export, model updates, backend identity and
execution accounting. Dynamics remain fixed-cell NVE.

### Fixed

- Resume now binds positions, driving forces, reference labels and update
  re-anchoring to the verified committed row; orphaned pre-commit rows remain
  as audit records but never drive a trajectory.
- Array sidecars carry dtype/byte-order/shape/content identity at all three
  entry points (event, model artifact, checkpoint); 0-D states keep their
  scalar shape.
- The initial frame keeps its original complete-step momenta on export; only
  genuine mid-step force evaluations get the second half-kick reconstruction.
  Export, inspect and resume share one commit/phase-aware read
  interface per task kind: relax iterations and singlepoints export
  correctly, uncommitted steps are never counted or exported as complete, and
  inspect separates the last committed boundary from the last evaluation.
- Training-success logging, state_dict, artifact and publish form a single
  rollback domain; a recalibrated-but-uncommitted proposal restores its
  segment history exactly once.
- Single-layer ASE mixing calculators fingerprint their direct children
  and weights. Automatic identity detection for nested or custom wrappers
  requires additional verification.
- Engines accepting request_id expose an explicit attempt sink; a run log is
  connected for the duration of each call, and sink-less combinations are
  refused before launch instead of undercounting retries.
- One explicit execution per compute, then the whole result set is read — no
  implicit second run for missing stress; executable-missing failures record
  zero attempts; post-processing failures terminate the attempt record as
  `post_processing_failed`; density staging is timed over its real interval
  and cache hits are timed per access.

### Compatibility

- 0.4.0 event logs and checkpoints remain readable; logs without the
  `physical_attempt_v1` marker keep their previous ledger semantics.

## 0.4.0

Configuration-driven, resumable runs with an explicit evaluation contract,
a complete cost record and guarded model updates. Includes bulk Si and Al(111)
examples using QE + MACE.

### Added

- TOML configuration (`schema_version = 1`) and a `pyramid` CLI:
  `init`, `validate`, `run`, `resume`, `inspect`, `export` and `backends`.
  Units are part of field names, relative paths resolve against the
  configuration file, and unknown fields are rejected with a remedy.
  `--help`/`--version` work offline; validation runs no SCF and downloads
  nothing unless `--probe-backends` is given.
- Workflow layer (`pyraimd2.workflows`) driving three tasks — `singlepoint`,
  `relax` (ASE FIRE/BFGS under a fixed model) and `md` — in reference-only
  or surrogate-only mode, plus adaptive MD. The CLI and Python API use the
  same code path.
- Complete-step checkpoints with atomic generation snapshots and sha256
  manifests. Adaptive and plain runs resume in a new process from the last
  valid generation plus ordered event replay — same physics, model chain and
  check stream, without re-thermalizing, re-training or re-drawing checks.
  `resume --steps N` always means N additional steps; `fork` starts an
  inheriting run with a fresh check stream.
- Single-writer JSONL event log with idempotent appends, persistent
  run/evaluation/segment/model/task/attempt/label identities, and a per-task
  cost ledger distinguishing logical requests, actual executions, failed
  attempts and cache hits.
- Evaluation contract: `EvaluationContext` with explicit physical time from
  the integrator, energy-kind and force-consistency metadata on results, and
  capability declarations checked before any expensive calculation.
- `pyraimd2.backends` registry: lazy builtin factories (`qe`, `qe-ase`,
  `pyscf`, `mace`, analytic `harmonic-reference`/`harmonic-surrogate`) and
  third-party plugins through the `pyraimd2.backends` entry-point group, with
  kind and capability checks at creation time.
- QE execution support: explicit `xc`/`dispersion` recipe (default PBE+D3),
  pseudopotential content hashes in the reference fingerprint, combined
  return-code/output/convergence success determination, process-group cleanup on
  timeout, bounded classified retries, and manifest-verified density warm
  starts with per-attempt copies.
- Guarded model updates: `GuardedUpdater` (candidate → guard-set validation →
  publish or roll back to the exact parent state), an immutable
  `ModelRegistry` artifact store recording parent model, label set, recipe and
  training cost, label-consumption dedup by durable label ID, and
  `update_rejected` records. `LegacyCallbackAdapter` keeps record-only
  callbacks from triggering pseudo-updates.
- `FixAtoms` support: probe directions, error norms and driving forces are
  projected; the force budget measures the free coordinates by default
  (`policy.force_metric = "active_dofs_max_atom"`, with
  `"all_atoms_max_atom"` as the explicit alternative); raw forces and the
  actual constrained displacement are recorded per evaluation. Other
  constraint kinds and variable-cell dynamics are rejected explicitly.
- Exact-match verification label cache (a cache hit counts the check without
  a physical reference execution), offline verification replay
  (`replay_verification`), and run inspection/export tooling
  (`inspect_run`, `summary_csv`, extxyz export with an explicit force source
  and NaN-marked missing labels — never zero-filled).
- Runnable offline example `examples/harmonic_adaptive/` (identical to the
  `init` template), a QE + MACE configuration skeleton
  `examples/qe_mace_skeleton/`, an installable example backend plugin
  `examples/backends/pyraimd2_harmonic/`, and user documentation:
  `docs/configuration.md`, `docs/api.md`.
- Material examples with structure generators, input requirements and
  runnable configurations: `examples/si_bulk_qe_mace/` (8-atom bulk Si) and
  `examples/al_surface_qe_mace/` (Al(111) slab with bottom layers fixed).
  Both include surrogate singlepoint/relaxation, plain and adaptive NVE,
  resume and export. The offline periodic example `examples/periodic_lj/`
  uses the `pyraimd2_lj` backend plugin. A generic Slurm template is available
  in [examples/slurm/](examples/slurm/README.md).

### Fixed

- ASE adapters return raw (`apply_constraint=False`) energies consistent with
  the already-raw forces and stress; constraints are applied once, at the
  workflow layer.
- QE calculations use unique per-call directories with absolute input paths;
  density-start fallback writes an explicit atomic-start input and keeps the
  failed attempt for diagnosis. Output parsing groups the final energy, its
  force block and stress from one complete SCF block, accepts Fortran `D`
  exponents, and rejects non-finite results.
- Committee surrogate state loading validates the full specification before
  touching any state; non-CPU committee data paths are refused.
- The energetic calculator validates immutable physical state (masses,
  constraints, identity) even when ASE's cache short-circuits a calculation,
  and a rejected evaluation never leaves stale results.
- Model changes invalidate same-geometry cached results, pending proposals,
  anchors and calibrations through model-generation keying; a reference
  settings change mid-run is refused before the next expensive call.

### Changed and compatibility notes

- Version now has a single source in `pyproject.toml`;
  `pyraimd2.__version__` reads installed package metadata.
- Python 3.12+ required; the core install remains NumPy + ASE only.
- The `Engine`/`Surrogate` protocols, positional result construction, the
  store row format (old rows remain readable), and the legacy switching API
  (`Runner`, `SwitchingCalculator`, scheduled/conformal policies) are
  unchanged.
- QE working-directory layout changed to `<label>-NNNNNN/attempt-N/` (run
  artifacts only, no API change), continuing across processes so resume never
  collides; pseudopotentials inside `pseudo_dir` are written as basenames in
  ATOMIC_SPECIES lines (long absolute paths overflow QE's card-line buffer).
  With smearing, the QE engine now declares `free_energy` (the `!`
  value is the variational free energy); it was previously mislabeled `energy`.
- Each side's reported scalar must be consistent with its own forces, and a
  cross-kind combination (e.g. a smeared QE `free_energy` reference with a
  potential-energy MACE surrogate) is allowed when both sides strictly
  declare force-energy consistency and conservative forces — equal kind
  strings are no longer required, and an unverified (`unknown`) side can
  never make such a combination pass.
- Tensor updater states persist as digest placeholders plus npz sidecars
  (model artifacts, checkpoints and the event log), so committee
  fine-tune states survive publish, checkpointing and fresh-process
  resume.
- Scope in 0.4.0 and 0.4.1: NVE only, FixAtoms only,
  `checkpoint.keep_generations` fixed at 2, and adaptive mode only for
  `task.kind = "md"`.

## 0.3.0

- Present Pyramid as **Python wrapped Ab initio Molecular Dynamics**, an
  extensible framework connecting reference engines, fast interatomic
  potentials, solvers and molecular dynamics.
- Add `pyraimd2.energetics` for directional residual response, empirical force
  forecasts, the directional residual-work coefficient, signed work and
  independent accepted-force checks.
- Add `EnergeticCalculator` and `EnergeticRunner` for reference-triggered
  anchoring and energetic MD decisions, with optional `on_label` adaptation.
- Add generic `AseEngine` and `AseSurrogate` adapters for configured ASE
  calculators, including force-consistent energy and optional stress settings.
- Add a harmonic energetic-loop example and an optional PySCF molecular example.
- Keep the base installation to NumPy and ASE. Provide `mace`, `pyscf`, `all`,
  `dev` and `builders` extras for optional dependencies; require Python 3.12 or
  later.
- Add the `pyramid` command alongside `pyraimd2`, retaining the `pyraimd2` Python
  import and existing switching workflows.
- Replace the overview with the energetic workflow and add architecture and
  method guides.
