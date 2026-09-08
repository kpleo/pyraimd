# API reference

Public entry points of the `pyraimd2` package, grouped by layer. Everything
listed here is importable from the named module and covered by the test suite;
anything not listed is internal and may change without notice. Units are fixed
everywhere: Å, eV, eV/Å, fs, K, ASE Voigt stress.

For protocols and design rationale read
[architecture.md](architecture.md) first; for TOML fields and CLI semantics
read [configuration.md](configuration.md). This page is the map between them.

Typical imports are lazy: `import pyraimd2` pulls in nothing beyond the
version, and the CLI imports workflow modules per command. Backends with
optional third-party dependencies (MACE, PySCF) are only imported when
selected.

## `pyraimd2.workflows` — run orchestration

The same functions the CLI calls; using them from Python is identical to the
command line.

- `run_workflow(config, *, verbose=True, handle_sigint=True) -> WorkflowResult`
  — validate, set up the run directory and execute the configured task
  (`singlepoint`, `relax` or `md` in any supported mode).
- `resume_workflow(run_dir, extra_steps, *, force_unlock=False, verbose=True, ...)`
  — continue a run for *additional* steps from its persisted state; adaptive
  and plain reference/surrogate runs are both resumable.
- `validate_setup(config, *, probe=False) -> dict` — everything checkable
  without running: schema, structure, paths, backend construction, capability
  contract. `probe=True` additionally evaluates the structure once per backend.
- `export_run(run_dir, *, force_source="driving", output=None, force=False) -> dict`
  — export the committed trajectory to extxyz; `FORCE_SOURCES` lists the
  valid sources. Missing labels are NaN-marked, never zero-filled.
- `write_template(template, output_dir, *, force=False) -> Path` — write a
  runnable `run.toml` + `structure.extxyz`; `TEMPLATES` lists the available
  templates.
- `create_configured_backend(section, config, *, run_dir=None, event_log=None)`
  / `build_backends(config, ...)` — construct configured backends through the
  registry (factories declaring `run_root`/`event_log` receive them here).
- `load_structure(config) -> Atoms` — read and sanity-check the configured
  structure, merging `[constraints]` FixAtoms.
- `prepare_run_directory(config, *, engine, surrogate) -> Path` — create the
  run directory with the config copy, `resolved_config.json` and manifest.
- `frames_from_store(store, run_id, *, force_source, ...)` — export frames in
  step order from the authoritative store.
- Result/record types: `WorkflowResult`, `RunOutputs`. Errors: `WorkflowError`,
  `ExportError`.

Configuration loading lives one layer down: `pyraimd2.config.load_config(path)
-> PyramidConfig`, `load_resolved_config(path)`, `ConfigError`,
`CONFIG_SCHEMA_VERSION`.

## `pyraimd2.backends` — factory registry

Backend names usable from TOML resolve here.

- `available_backends() -> dict` — registered names with declared kind and
  origin (`builtin` or `entry-point:<distribution>`); imports nothing.
- `create_backend(name, *, kind=None, require=None, **options)` — create a
  backend by name, importing its module now; `kind` (`"engine"`/`"surrogate"`)
  and `require` (an `EngineCapabilities`/`SurrogateCapabilities` requirement)
  are enforced before any calculation.
- `backend_factory(name)` — the registered factory callable itself.
- `backend_capabilities(backend)` — declared capabilities of a created
  backend, unknown-safe.
- `assert_capabilities_satisfy(caps, require, *, name="backend")` — the
  requirement check, raising on undeclared capabilities.
- `BackendRegistration`, `ENTRY_POINT_GROUP` (`"pyraimd2.backends"`) — the
  plugin contract: a distribution registers factories under this entry-point
  group, declaring `backend_kind` on each factory.
- Errors: `BackendRegistryError`.

Builtin names: `qe`, `qe-ase`, `pyscf` (engines), `mace` (surrogate),
`harmonic-reference`, `harmonic-surrogate` (analytic, offline).

## `pyraimd2.engines` — reference backends

- Protocols and results: `Engine` (`compute(atoms) -> EngineResult`, raising
  `EngineError` when no valid label can be produced), `EngineResult` (energy,
  forces, optional stress, wall time, plus energy-kind/force-consistency
  metadata), `EnergyKind` (`energy` / `free_energy` / `unknown`).
- Contracts: `EngineCapabilities` (undeclared means unknown — never treated as
  supported), `engine_capabilities(engine)` (all-unknown fallback),
  `CapabilityMismatchError`.
- Implementations: `PyscfEngine` (molecular closed-shell RKS), `QeEngine` with
  `QeConfig` (periodic fixed-cell through `pw.x`; `QeEngineError` states
  whether a retry can help), `AseEngine` (any configured ASE calculator),
  `AseQeEngine` (ASE-native Espresso path).

## `pyraimd2.surrogate` — fast potentials

- Protocols and results: `Surrogate` (`predict(atoms) ->
  SurrogatePrediction`), `TrainableSurrogate` (adds `finetune(labels) ->
  TrainReport`), `SurrogatePrediction` (energy, forces, optional stress and
  per-atom uncertainty), `SurrogateCapabilities`, `surrogate_capabilities`
  fallback.
- `assert_compatible_energy_contract(engine_caps, surrogate_caps)` — fail
  before any expensive call when declared energy conventions conflict.
- Implementations: `MaceSurrogate` (single MACE model, lazy calculator),
  `CommitteeSurrogate` (K-member committee with uncertainty and fine-tuning),
  `AseSurrogate` (any configured ASE calculator).

## `pyraimd2.runtime` — run services

Persistence, identity and accounting behind resumable runs.

- Identity: `EvaluationContext` / `EvaluationPhase` (one logical evaluation:
  run/step/evaluation ids, phase, explicit physical time, model id),
  `fingerprint_of`, `model_id_for`, `atoms_input_hash`.
- Events and costs: `EventLog` (single-writer append-only JSONL with an
  exclusive lock) / `EventLogError`, `summarize_tasks(events)` (ledger:
  logical vs actual vs failed vs cache-hit), `inspect_run(run_dir)` /
  `format_inspection(info)` / `summary_csv(store, run_id)`.
- Recovery: `CheckpointManager` (atomic generation snapshots with hash
  manifests; reads fall back to the last valid generation) /
  `CheckpointError` / `ResumeError`.
- Models and labels: `ModelRegistry` (immutable per-model artifacts) /
  `ModelRegistryError`, `StatefulUpdater` (the resumable updater protocol),
  `LabelCache` / `label_key` (exact-match verification cache).

## `pyraimd2.energetics` — pure numerics

Array-level quantities of the method, independent of MD and I/O:
`estimate_responses` / `DirectionalResponse` / `DegenerateResponseError`
(local directional force response), `Forecast` (predicted residual force and
signed work), `residual_work` / `integrate_residual_work` (signed work along a
segment), `IndependentCheckBound` (accepted-force check accounting).

## `pyraimd2.loop` — dynamics and updates

`EnergeticCalculator` (ASE calculator implementing the energetic policy) and
`EnergeticRunner` (fixed-cell Velocity Verlet with checkpointing, resume and
fork) drive adaptive MD; `EnergeticRunSummary` reports a run. Model updates go
through `GuardedUpdater` + `UpdatePolicy` (candidate → guard-set validation →
publish or roll back); `LegacyCallbackAdapter` adapts 0.3.0-era `OnlineUpdater`
callbacks. The pre-0.4 switching interface (`Runner`, `RunSummary`,
`SwitchingCalculator`, and the scheduled/conformal policies in
`pyraimd2.switch`) remains available unchanged.
