# PYRAIMD-2 — Phase B Design Document

Status: **draft v0.1** (development constitution — changes require discussion, not silent edits)

## 1. Mission

Rebuild the 2021 PYRAIMD concept — on-the-fly machine-learned acceleration of ab initio
molecular dynamics with DFT fallback — on top of foundation interatomic potentials, with a
*statistically calibrated* switching criterion, and demonstrate it on genuine finite-temperature
MD at the ~10³-atom scale.

Phase B is also the software foundation for Phase A (learned fictitious-dynamics /
"AI Car–Parrinello" framework). Every architectural choice below must keep that door open:
the surrogate slot and the switch slot will later host a learned energy functional E_θ(R, z)
and an extended-Lagrangian propagator.

## 2. What Phase B is NOT

- Not another from-scratch MLIP trainer (MACE/DeePMD already own that).
- Not a re-implementation of VASP MLFF / DP-GEN from zero (they own uncertainty-gated
  from-scratch learning). Our delta: *foundation-model prior* + *calibrated* switching.
- Not a production MD engine. ASE drives the loop; LAMMPS (ML-IAP) is the production channel.
- Not tied to one DFT code. The engine is a swappable ASE-calculator factory.

## 3. Lessons imported from PYRAIMD v1 (failure-mode → rule)

The v1 audit found three commit-level fatal bugs in the hybrid path (stale signatures after a
virial refactor, swapped model kwargs, upgraded data written but never reloaded), zero test
coverage on the ML/hybrid machinery, pervasive `__globals__` injection, `os.system(mpirun)`
with unchecked exit codes, and CWD-coupled tests that could not even be collected. Rules:

1. **No hidden global state.** All state lives in explicit dataclasses passed as arguments.
2. **Every subprocess exit code is checked**; failures raise, never silently reuse stale files.
3. **Restart correctness is a tested requirement**: everything the loop learns is in `Store`,
   and a restart from `Store` alone reproduces the run.
4. **No CWD-dependent tests.** Unit tests are hermetic (fake engine); the only test that
   touches a real quantum code is a marked integration test.
5. **The switching decision is a first-class object** with logged rationale, not a threshold
   buried in a loop.
6. Units are explicit (ASE units everywhere: eV, Å, eV/Å, fs); conversions happen only at
   engine boundaries and are covered by tests.
7. Virial/stress is part of the force-response contract from day one (v1 broke exactly here).

## 4. Architecture

```
                 ┌────────────────────────────────────────────┐
                 │                 Runner                     │
                 │  ASE MD loop (NVE/NVT), one step at a time │
                 └──────────────┬─────────────────────────────┘
                                │ atoms (R, v, cell)
                                ▼
                     ┌─────────────────────┐   score, quantile
                     │       Switch        │──────────────┐
                     │ committee + online  │              │
                     │ conformal decision  │              │
                     └──────┬──────────────┘              │
                  trust     │            distrust          │
                            ▼                              ▼
                 ┌───────────────────┐        ┌────────────────────────┐
                 │     Surrogate     │        │        Engine          │
                 │ foundation model  │        │ DFT backend (PySCF /   │
                 │ + online fine-tune│        │ GPAW / QE), subprocess │
                 └─────────▲─────────┘        │ supervised, checked    │
                          │ fine-tune on      └───────────┬────────────┘
                          │ accumulated labels            │ E, F, σ
                          │                               ▼
                          │                     ┌──────────────────┐
                          └─────────────────────│      Store       │
                            labels              │ ASE db (SQLite), │
                                                │ append-only,     │
                                                │ restart-complete │
                                                └──────────────────┘
```

### 4.1 Components

- **Runner** — owns the MD integrator (ASE VelocityVerlet / Langevin). Per step: asks Switch,
  routes to Surrogate or Engine, applies the returned forces, logs everything. Knows nothing
  about model internals.
- **Surrogate** — wraps a foundation model (MACE-MP-0 first; interface is model-agnostic).
  Exposes `predict(atoms) -> EnergyForcesStress` and `finetune(batch) -> TrainReport`.
  Fine-tuning is head-only by default (backbone frozen) to bound cost and forgetting.
- **Switch** — the decision engine. Two coupled signals:
  1. *Committee disagreement*: K surrogate heads → predictive spread σ_F on per-atom forces.
  2. *Online conformal calibration*: a running calibration window converts σ_F into an
     empirical miscoverage estimate against actually-observed DFT errors; switch fires when
     the predicted error quantile exceeds the user's accuracy budget ε_acc.
  Target property: long-run fraction of accepted steps with true force error > ε_acc is ≤ α
  (user-set, default 0.05), *under exchangeability of the calibration window* — the honest
  limit of conformal under MD time correlation; we state it, measure its violation, and
  report effective vs nominal coverage.
- **Engine** — factory producing a configured ASE calculator for a supervised DFT call.
  Local dev: PySCF (molecules). Cluster: GPAW / QE / CP2K (periodic). Contract:
  `compute(atoms) -> EnergyForcesStress`, raises `EngineError` on any failure.
- **Store** — ASE db (SQLite), append-only. Each row: atoms + engine labels + surrogate
  prediction + switch rationale + wall times. The loop can be killed and resumed from Store
  alone (rule 3).
- **Experiment** — a versioned config (TOML + pydantic) naming the system, engine, surrogate,
  switch parameters, and MD settings. One command reproduces a run.

### 4.2 The per-step contract

```
atoms_t  → Switch.assess(atoms_t, surrogate) -> Decision(route, score, quantile)
route=ML : Surrogate.predict -> (E,F,σ) ; MD step
route=DFT: Engine.compute -> (E,F,σ) ; Store.append(labels) ;
           maybe Surrogate.finetune(Store.recent(batch_policy)) ; MD step
```

Fine-tuning trigger policy (initial): fine-tune every N_label new DFT labels
(default N_label=8) or when the conformal quantile drifts beyond a dead band.
Batch policy: recent window + reservoir sample of older labels (bounded forgetting).

## 5. Why calibrated switching is the contribution

- v1 used R²(descriptor_new, descriptor_ref) ≥ 0.99999: not an error metric, no guarantees.
- VASP MLFF / FLARE use Bayesian predictive variance: principled but heuristic in practice;
  miscalibration silently breaks the loop (documented in 2025 UQ-MLIP reviews).
- Our claim: conformal calibration on top of committee disagreement gives a *distribution-free,
  assumption-light* coverage statement for the switching decision, with online updates tracking
  distribution drift along the trajectory. If effective coverage ≈ nominal coverage across
  systems, that is the paper's quantitative core.

## 6. Module layout

```
src/pyraimd2/
├── engines/    # Engine protocol + pyscf / gpaw / qe implementations
├── surrogate/  # Surrogate protocol + MACE wrapper (+ committee heads)
├── switch/     # Decision, CommitteeScorer, OnlineConformal
├── loop/       # Runner (ASE MD integration), StepLog
├── store/      # Store (ASE db wrapper), restart logic
├── experiment/ # Experiment config (pydantic), entry points
└── cli.py
tests/
├── unit/       # hermetic; FakeEngine, FakeSurrogate drive all loop/switch/store logic
├── regression/ # H2O NVE energy-drift, restart-equality; model download allowed
└── integration/# marked; requires real GPAW/QE (cluster only)
examples/
└── minimal_loop.py   # environment verification: MACE-MP-0 + PySCF on H2O
```

## 7. Testing strategy

- Unit: FakeEngine (analytic potential) + FakeSurrogate (controllable error) exercise every
  branch of Switch/Runner/Store, including restart-from-Store equality and engine-failure
  raising.
- Regression: 1 ps NVE on H₂O with the surrogate alone, total-energy drift < 1 meV/atom/ps
  (with default settings); restart mid-run reproduces trajectories bit-for-bit.
- Integration (cluster): GPAW-labeled 100-step adaptive run on bulk Si; effective conformal
  coverage within 0.05 ± 0.03 of nominal.
- Lint/type: ruff clean; mypy on `switch/` and `store/` (the statistical core).
- CI (later): GitHub Actions, unit+regression on CPU; integration excluded.

## 8. Resource policy (local dev / cloud HPC for scale)

The local machine is for development only. Anything beyond small, fast tests runs on
cloud GPU HPC (several available; pick an idle one).

- Local: development, unit tests, tiny regression runs — systems ≤ ~50 atoms, runtimes
  of seconds to a few minutes (e.g. the M1 200-step H₂O loop, 7 s).
- Cloud HPC: longer MD, larger cells, DFT labeling at scale, fine-tuning beyond trivial
  head-only dev runs, integration suites, M3 flagship, all paper production runs.
- Dev surrogate: MACE-MP-0 "small". Dev DFT labels: PySCF (molecules) locally;
  PySCF-pbc tiny cells locally; GPAW/QE on HPC when periodic labeling at scale begins.
- Operational notes: the repo syncs to HPC via git; the HPC env needs the CUDA torch
  build (local macOS uses CPU/MPS wheels) — handled once at first HPC setup and pinned
  in uv.lock / an extra index. Scheduler, queue names, and idle-node selection are
  recorded in `docs/hpc.md` once access is confirmed.
- Every test must still pass locally on CPU; HPC/GPU code paths are config-gated with
  local fallbacks.

## 9. Milestones and acceptance criteria

- **M0 — scaffold** (this week): repo, deps, design doc, `examples/minimal_loop.py` runs
  (MACE-MP-0 MD + PySCF reference on H₂O). ✅ when the example prints a passing report.
- **M1 — minimal closed loop**: Engine(PySCF) + Surrogate(MACE-MP-0, frozen) + scheduled
  Switch + Store + Runner; 200-step adaptive MD on H₂O locally (a tiny periodic cell via
  PySCF-pbc optional; GPAW Si on the cluster is deferred to M3 per §8). ✅ when:
  restart-equality test passes; DFT fraction logged; engine failures raise, never silently
  reuse stale results.
- **M2 — calibrated switch**: committee + online conformal; coverage measured on Si and a
  held-out molecule. ✅ when: effective coverage within ±0.03 of nominal on both systems;
  ablation vs committee-only and vs fixed-threshold reported.
- **M3 — flagship demo** (superseded 2026-08-19: TBG+Li replaced by the battery
  electrolyte/interface demonstration — see docs/design-m3.md, the normative spec):
  two-tier demo — bulk 1 M LiPF₆/EC:DMC transport vs experiment, and Li(100)|electrolyte
  initial-SEI interface MD with conformal reactive-event detection; QE labels from small
  cells on Neimeng A. ✅ when: the §6 acceptance criteria in design-m3.md are met.
- **M4 — paper**: draft with (i) framework, (ii) calibrated-switching theory + measured
  coverage, (iii) electrolyte/interface flagship, (iv) baselines (scheduled switching,
  zero-shot foundation, classical FF, BAMBOO-class numbers). Target: JCP / npj Comput.
  Mater.

## 10. Open questions (to resolve during M1–M2)

1. Committee construction: K separate fine-tuned heads vs MACE ensemble vs last-layer
   dropout — trade accuracy of σ_F against 3–5× eval cost.
2. Conformal under MD time correlation: sliding-window size; adaptive α; how badly does
   effective coverage degrade on fast-drifting trajectories (measure, don't assume).
3. Fine-tuning scope: head-only vs LoRA on backbone; forgetting of the foundation prior
   (monitor zero-shot error on a fixed reference set as a forgetting gauge).
4. Nonconformity score: max per-atom force error vs energy-weighted norm — which tracks the
   DFT error better across systems.
5. Stress/virial through the surrogate at scale (needed for NPT; NVT first).

## 11. Phase A hooks (do not build yet, do not close off)

- `Surrogate.predict` already returns a structured result; Phase A replaces it with
  `EnergyFunctional(R, z)` + an extended-Lagrangian propagator behind the same Runner.
- `Engine` already isolates the quantum code; Phase A needs Hamiltonian/density-matrix
  labels from the same boundary (GPAW/CP2K), so keep the label payload extensible.
- Store rows must tolerate extra per-step observables (dipole, charges) without schema
  migrations.
