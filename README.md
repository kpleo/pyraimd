# PYRAIMD

**Runtime-assured AI-accelerated ab initio molecular dynamics.**
A conformal runtime supervisor routes every MD step between a fine-tuned
foundation-potential committee and direct DFT — accelerating ab initio MD
while keeping a finite-sample statistical guarantee on the force accuracy of
every accepted step.

Successor of the original PYRAIMD (2021) project.

## The problem

- Ab initio MD (AIMD) is the accuracy reference for reactive and
  far-from-equilibrium chemistry, but its cost caps accessible time and
  length scales.
- Machine-learned interatomic potentials (MLIPs), including foundation
  models, are orders of magnitude faster — but they fail **silently** out of
  distribution, and MD trajectories inevitably drift out of distribution.
- Classical active-learning schemes retrain offline on a schedule or on
  heuristic uncertainty thresholds: they offer no per-step guarantee about
  the trajectory that was actually produced.

The question PYRAIMD answers: *can an MD engine decide, at every single
step and with a statistical guarantee, whether the surrogate's prediction
is accurate enough — and pay for DFT exactly when, and only when, it is
needed?*

## How it works

Each step, the engine evaluates a small committee of surrogates and computes
an honest per-atom force spread `s`. A sliding window of the most recent
DFT-labeled steps calibrates that spread against **realized** errors `e`
through a normalized nonconformity ratio `r = e / (s + δ)`; the conformal
quantile `q̂` (with finite-sample correction) turns the spread into an error
bound `B(s) = q̂·(s + δ)`. If `B(s) ≤ ε_acc` the step is integrated with the
committee; otherwise the step is labeled by DFT and the label joins the
calibration window — and periodically fine-tunes the committee online.

Design principles:

- **Guarantee first.** The supervisor never silently trusts the surrogate;
  when nothing accurate can be certified it refuses (all-DFT) rather than
  degrade. Under approximate exchangeability, the long-run fraction of
  accepted steps with true error above `ε_acc` is ≈ α.
- **Honest uncertainty.** Committee spread uses the RMS estimator
  `σ_i = sqrt(mean_k |F_ik − F̄_i|²)` over mixed-backbone members
  (e.g. MACE-MP-0b3 + MACE-MPA-0). Cross-backbone diversity keeps the spread
  calibrated (residual miscalibration is absorbed by the conformal layer).
- **Bounded forgetting.** Every fine-tune restarts from a frozen snapshot of
  the foundation's readout heads with seeded perturbation and a seeded
  bootstrap resample — deterministic given (seed, labels).
- **Replayable science.** Every decision, prediction, and label lands in an
  append-only store; runs resume bit-for-bit, and any routing policy can be
  replayed offline against recorded labels.

## Components

- `pyraimd2.surrogate` — committee surrogate over MACE foundation models
  (single- or mixed-backbone), readout-head fine-tuning, energy referencing.
- `pyraimd2.switch` — the conformal switch (`ConformalSwitch`), a scheduled
  ablation switch, and the offline replay harness.
- `pyraimd2.engines` — DFT engines: Quantum ESPRESSO (`QeEngine`, with a
  hardened SCF protocol for large insulating liquid cells) and PySCF.
- `pyraimd2.loop` — the ASE-driven runner, the switching calculator, and the
  online updater (calibration ingest + periodic fine-tune).
- `pyraimd2.store` — the append-only trajectory/label store (ASE db),
  restart-safe.

## Repository layout

- `src/pyraimd2/` — the framework (components above).
- `experiments/` — production drivers (single-node adaptive loop for HPC).
- `examples/` — minimal local loops.
- `hpc/neimeng/` — structure builders, input generators, and Slurm templates
  for the production campaigns (electrolyte bulk/interface, W thermal spike).
- `docs/` — design documents (`design-m2.md`, `design-m3.md`), the HPC
  campaign logbook (`hpc.md`), and the research log.
- `tests/unit/` — fast unit tests (no model downloads, no DFT).

## Quickstart

```bash
uv sync
uv run pytest tests/unit -q        # fast suite
uv run python examples/minimal_loop.py
```

Production campaigns run on Slurm clusters; see `hpc/neimeng/` for the
structure builders, input generators, and Slurm templates used by the
campaigns.

## Status and roadmap

- M0–M2: framework, conformal switch, online fine-tuning, H₂O validation
  campaign (correct-refusal and calibrated-acceptance regimes both
  demonstrated).
- M3 (in progress): two flagship demonstrations — interfacial slow chemistry
  at a Li-metal|liquid-electrolyte interface, and defect production in W
  thermal-spike surrogates — with the runtime guarantee active throughout.

## License and citation

Released under the MIT License (see `LICENSE`). A citation entry will be
added with the associated publication.
