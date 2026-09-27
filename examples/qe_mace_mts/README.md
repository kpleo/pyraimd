# QE + MACE MTS compare-chain skeleton

A configuration *shape* for carrying the offline accuracy-verification
chain to a real setup: Quantum ESPRESSO as the reference engine and a
MACE model as the fast potential, over a small periodic silicon cell
(public sample data — a 2-atom diamond cell with explicit initial
momenta, teaching values only).

Three configurations share ONE structure (same initial state, momenta
included), ONE reference recipe (identical `[reference]` section) and ONE
physical-time span:

- `run_reference.toml` — plain reference-driven NVE (16 fs at 1 fs,
  illustrative), states at every fs;
- `run_mts.toml` — fixed-model MTS candidate with the plain MACE fast
  model (inner 1 fs, outer_ratio 4, complete boundaries at
  t = 0, 4, 8, 12, 16 fs — a subset of the reference grid);
- `run_mts_scaled.toml` — the same MTS arm with the `scaled` wrapper;
  `scale = 1.0` is a **placeholder** — replace it with your own
  `pyramid calibrate-scale` result (see `examples/calibrate_scale/`)
  and record the provenance in `calibration_note`.

For the backend option shape (`qe` recipe fields, `mace` options,
correction wrappers) see `examples/qe_mace_skeleton/` — which is an
ADAPTIVE-mode skeleton with `[policy]`/`[verification]`, not an MTS
template; MTS composes with neither — and `docs/configuration.md`.

## What works where

```sh
pyramid validate run_reference.toml      # schema, structure, backend contract
pyramid validate run_mts.toml
pyramid validate run_mts_scaled.toml
pyramid validate run_mts.toml --check-environment   # strictly read-only preflight
```

These static checks never execute anything. A missing `pw.x`, missing
pseudopotentials under `pseudos/`, or the placeholder `models/user.model`
not existing is a normal user-environment state — validation reports it;
nothing here pretends an environment check has already passed, and
validation never downloads or loads a model.

Real work is separate and explicit, on compute-authorized nodes only:

```sh
pyramid validate run_mts.toml --probe-surrogate   # one real fast-model call
pyramid validate run_reference.toml --probe-backends  # launches pw.x once per section
pyramid run run_reference.toml
pyramid run run_mts.toml
pyramid run run_mts_scaled.toml
pyramid compare runs/reference runs/mts --json
pyramid compare runs/reference runs/mts-scaled --json
```

The steps, timestep, cutoffs, k-points and outer ratio are illustrative:
do your own convergence checks, prepare your own paired calibration data
from independent reference evaluations, and choose the reference grid so
every complete candidate boundary has an exact reference time point (the
reference may be denser, never coarser than the candidate's boundaries).
`pyramid compare` reads the COMPLETED run directories only and reports
metrics — thresholds and their meaning are yours.
