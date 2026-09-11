# Changelog

## Unreleased

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

### Fixed

- An abandoned event log (a constructor or a failed resume dropping the
  writer without closing it) no longer leaks an OS handle: garbage
  collection releases the handle while the lock file stays as crash
  evidence, since lock removal remains a deliberate `close()`.
- The ASE-native QE attempt span is settled at its terminal exit, so the
  density-manifest write is counted once in the declared
  staging/process/validation span instead of being frozen out.
- Plain MD resume verifies each persisted step-summary format by its own
  semantics (0.4.x records carry none; the previous development batch's
  complete-boundary JSON digest; the current array digest plus boundary
  digest covering the thermostat stream) instead of rejecting older
  records as corrupt, and crash-healed step records now carry the same
  boundary digest as normal commits.
- Stores opened by the MD workflows are closed on every controlled
  failure path (setup, resume, adaptive), not left to garbage collection.

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
