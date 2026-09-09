# Changelog

## 0.4.0

The first workflow-stable release line: configuration-driven, resumable runs
with an explicit evaluation contract, a complete cost record and guarded model
updates. Materials validation landed with the two QE + MACE recipes below
(QE 7.5, MACE-MPA-0 medium at float64 on CPU).

### Added

- TOML configuration (`schema_version = 1`) and a `pyramid` CLI:
  `init`, `validate`, `run`, `resume`, `inspect`, `export` and `backends`.
  Units are part of field names, relative paths resolve against the
  configuration file, and unknown fields are rejected with a remedy.
  `--help`/`--version` work offline; validation runs no SCF and downloads
  nothing unless `--probe-backends` is given.
- Workflow layer (`pyraimd2.workflows`) driving three tasks — `singlepoint`,
  `relax` (ASE FIRE/BFGS under a fixed model) and `md` — in reference-only,
  surrogate-only and adaptive modes, from the CLI and the Python API through
  the same code path.
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
- QE productization: explicit `xc`/`dispersion` recipe (default PBE+D3),
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
- Materials recipes with inputs, settings, commands and full costs:
  `examples/si_bulk_qe_mace/` (8-atom bulk Si: surrogate relax, plain NVE in
  both modes, adaptive NVE with 4/7 accepted evaluations at checks p = 1.0,
  interrupted runs resumed in a new process), `examples/al_surface_qe_mace/`
  (Al(111) slab with bottom layers fixed: plain modes and plain
  checkpoint/resume), and the no-external-software periodic CI example
  `examples/periodic_lj/` with its `pyraimd2_lj` backend plugin. A generic
  Slurm template lives in `hpc/templates/`; site profiles are not shipped.

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
  `pyraimd2.__version__` reads installed package metadata. Development
  pre-releases carry the `0.4.0.devN` marker.
- Python 3.12+ required; the core install remains NumPy + ASE only.
- The `Engine`/`Surrogate` protocols, positional result construction, the
  store row format (old rows remain readable), and the legacy switching API
  (`Runner`, `SwitchingCalculator`, scheduled/conformal policies) are
  unchanged; the 0.3.0-era examples continue to run.
- QE working-directory layout changed to `<label>-NNNNNN/attempt-N/` (run
  artifacts only, no API change), continuing across processes so resume never
  collides; pseudopotentials inside `pseudo_dir` are written as basenames in
  ATOMIC_SPECIES lines (long absolute paths overflow QE's card-line buffer).
  With smearing, the QE engine now honestly declares `free_energy` (the `!`
  value is the variational free energy); it was previously mislabeled `energy`.
- The energy contract is refined per the independent review (section 5):
  each side's reported scalar must be consistent with its own forces, and a
  cross-kind combination (e.g. a smeared QE `free_energy` reference with a
  potential-energy MACE surrogate) is allowed when both sides strictly
  declare force-energy consistency and conservative forces — equal kind
  strings are no longer required, and an unverified (`unknown`) side can
  never make such a combination pass. Validated on the Al(111) recipe:
  QE's variational free energy differentiates to its forces within
  3.6e-5 eV/A (central differences, preset tolerance 5e-4), and a short
  adaptive run with resume completes with a ledger-consistent cost record.
- Tensor updater states persist as digest placeholders plus npz sidecars
  (model artifacts, checkpoints and the event log), so committee
  fine-tune states survive publish, checkpointing and fresh-process
  resume; verified end-to-end with a real MACE-MPA-0 committee on the
  Al(111) recipe (one accepted update, byte-exact restore).
- Current 0.4.0 limits by design: NVE only (NVT is 0.4.1), FixAtoms only,
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
- Remove obsolete development-stage notes, missing design-document references
  and a historical analysis driver that depended on unpublished run files.
