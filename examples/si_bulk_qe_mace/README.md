# Bulk Si with a QE reference and MACE surrogate

An 8-atom conventional diamond-Si cell with configurations for surrogate
singlepoint evaluation, FIRE relaxation, and 6-step NVE in reference-only,
surrogate-only and adaptive modes. The example includes resume and export.
These short runs demonstrate the workflow; choose converged settings and an
appropriate force budget before using the results for a materials study.

## Required inputs and setup

Run the commands below from this directory, after installing Pyramid with the
`mace` extra and making Quantum ESPRESSO `pw.x` available.

- Place `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE PAW, pslibrary 1.0.0) in a
  directory such as `./qe_pseudos/`. Set `[reference] pseudo_dir` in
  `run-md-reference.toml` and `run-md-adaptive.toml` to that directory.
- Place a MACE-MPA-0 medium checkpoint at `./mace-mpa-0-medium.model`, or
  change `[surrogate] model` in each configuration that uses MACE. The
  example uses CPU inference with `float64`.
- Set `[reference] pw_cmd` in both QE configurations to your execution
  command, for example `["pw.x"]` for a serial run or an MPI launcher
  matching your allocated resources. A generic
  [Slurm template](../slurm/README.md) is available.

Relative input paths resolve against the TOML file's directory. Keep the
pseudopotential filename under `[reference.pseudos]` as a basename within
`pseudo_dir`; the engine then writes short QE species-card entries.

## Settings and files

- `make_structure.py` writes `structure.extxyz`: an 8-atom diamond cell at
  a = 5.431 Å with one atom displaced from its ideal site.
- `run-singlepoint.toml` and `run-relax.toml` use the frozen MACE surrogate.
  Relaxation uses FIRE with `fmax_eV_A = 0.05` and a 100-step cap.
- `run-md-reference.toml`, `run-md-surrogate.toml` and
  `run-md-adaptive.toml` initialize velocities at 300 K with seed 7 and
  run NVE for 6 steps at 1.0 fs, checkpointing every 3 steps.
- The reference and adaptive configurations use PBE + Grimme D3,
  `ecutwfc = 50` Ry, `ecutrho = 400` Ry, a 4×4×4 k-point grid and
  `conv_thr = 1e-8` Ry.
- Adaptive settings: force budget 0.10 eV/Å, probe displacements
  0.02/0.04 Å, numerical floor 0.001 eV/Å, time cap 2 fs,
  transverse cap 0.1, and check probability 1.0. Every accepted force
  evaluation is checked; these values are illustrative accuracy settings.

## Run, resume and export

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

Add `--probe-backends` to `pyramid validate` to evaluate the initial structure
once with each backend. Ordinary validation checks setup without SCF or
inference. Run directories must be new; use `resume` to continue an existing
run. `Ctrl-C` requests a stop at the next complete-step boundary.

Each configuration starts from the same `structure.extxyz`. Relaxation does
not automatically feed its final geometry into MD. To use that geometry,
export `runs/si-relax`, select the final frame and set `[structure] file` in
a new MD configuration to it.

`pyramid inspect` reports reference execution costs, including anchors,
probes, checks and retries. Acceptance counts alone do not measure savings:
directional probes add reference calculations, and every acceptance is checked
with the supplied probability. See the
[configuration guide](../../docs/configuration.md) for output fields and
missing-label export semantics.
