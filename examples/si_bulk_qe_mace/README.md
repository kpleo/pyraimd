# Bulk Si with a QE reference and MACE surrogate

An 8-atom conventional diamond-Si cell. Two ways to run it:

- **The serial recipe** (`recipe/` + `run_recipe.py`) — the recommended path:
  fixed-MACE FIRE relaxation, an adaptive Langevin NVT stage (QE reference
  against the frozen MACE surrogate, 12 steps at 300 K), and a fixed-MACE
  NVE stage continuing with the NVT boundary's complete momenta. Idempotent
  stop/continue, per-stage run identities, provenance and costs in
  `workflow.json`.
- **The single-run configurations** (`run-*.toml`) — singlepoint, relax,
  and 6-step NVE in reference-only, surrogate-only and adaptive modes, for
  studying one task at a time.

These short runs demonstrate the workflow. "Completed NVT" means the
configured 12 steps ran — it is not a claim of thermal equilibrium or of
unbiased canonical sampling; the NVE stage propagates the frozen surrogate
potential only, not all-reference AIMD. Choose converged settings and an
appropriate force budget before using any result for a materials study.

## Required inputs and setup

Install Pyramid (with the `mace` extra) and make Quantum ESPRESSO `pw.x`
available, then provide three inputs — paths below are examples, not
prefilled values:

- **MACE weights**: a MACE-MPA-0 medium checkpoint, e.g.
  `./mace-mpa-0-medium.model` relative to the configuration file (the
  recipe runs CPU inference with `float64`).
- **Pseudopotentials**: `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE PAW,
  pslibrary 1.0.0) in a directory such as `./qe_pseudos/`; set
  `[reference] pseudo_dir` to it. Pseudopotential filenames under
  `[reference.pseudos]` stay basenames within `pseudo_dir`.
- **A launch command**: set `[reference] pw_cmd` to your `pw.x` invocation,
  e.g. `["pw.x"]` serially or an MPI launcher matching the allocated
  resources. A generic [Slurm template](../slurm/README.md) is available.

Relative input paths resolve against the TOML file's directory. The
reference and adaptive configurations use PBE + Grimme D3, `ecutwfc = 50`
Ry, `ecutrho = 400` Ry, a 4×4×4 k-point grid and `conv_thr = 1e-8` Ry —
the previously run-through starting point, not a convergence study.

## The serial recipe

```sh
python run_recipe.py --output results/si-recipe   # relax + NVT + NVE
python run_recipe.py --output results/si-recipe   # continue, idempotent
pyramid inspect results/si-recipe/nvt
pyramid export results/si-recipe/nve --force-source driving
```

The recipe is idempotent: re-running the same command skips finished
stages and resumes a crashed MD stage to its configured total (pass
`--force-unlock` deliberately to reclaim the lock of a stage whose process
died). Each stage has its own run id, event log and cost ledger under the
output directory; `workflow.json` records per-stage status, parent
provenance and purpose-split reference costs (anchors, probes, checks,
refusals, retries). The NVT stage's momenta are initialized exactly once
at 300 K (`velocity_seed = 7`, recorded in the manifest and the run's
RUN_START streams block); the NVE stage keeps the complete momenta and has
no bath settings. Configuration checks without any computation:
`pyramid validate recipe/nvt.toml` (after providing the inputs above;
the NVT stage's structure is materialized by the controller at run time).

## Settings and files

- `make_structure.py` writes `structure.extxyz`: an 8-atom diamond cell at
  a = 5.431 Å with one atom displaced from its ideal site.
- `recipe/relax.toml` — fixed-MACE FIRE, `fmax_eV_A = 0.05`, 100-step cap.
- `recipe/nvt.toml` — adaptive Langevin NVT: 12 steps, `dt = 1.0` fs,
  300 K, `friction_per_fs = 0.01`, checkpoint interval 4, check
  probability 1.0 (every accepted force paired-checked), force budget
  0.10 eV/Å, probes 0.02/0.04 Å, numerical floor 0.001 eV/Å, time cap 2 fs,
  transverse cap 0.1.
- `recipe/nve.toml` — fixed-MACE NVE, 12 steps, `dt = 1.0` fs.
- `run-singlepoint.toml` / `run-relax.toml` / `run-md-*.toml` — the
  single-run variants (6 NVE steps each).

## Single-run commands

```sh
python make_structure.py
pyramid validate run-md-adaptive.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
pyramid run run-md-adaptive.toml
# Continue the 6-step adaptive run for 2 additional steps.
pyramid resume runs/si-adaptive --steps 2
pyramid inspect runs/si-adaptive --json
pyramid export runs/si-adaptive --force-source driving
```

Add `--probe-backends` to `pyramid validate` to evaluate the initial
structure once with each backend. Ordinary validation checks setup without
SCF or inference. Run directories must be new; use `resume` to continue an
existing run. `Ctrl-C` requests a stop at the next complete-step boundary.

`pyramid inspect` reports reference execution costs, including anchors,
probes, checks and retries. Acceptance counts alone do not measure savings:
directional probes add reference calculations, and every acceptance is
checked with the supplied probability. See the
[configuration guide](../../docs/configuration.md) for output fields and
missing-label export semantics.
