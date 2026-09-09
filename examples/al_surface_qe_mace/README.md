# Al(111) slab with fixed layers, a QE reference and MACE surrogate

A 17-atom Al(111) slab: 2×2 surface cells, four layers and one adatom.
The bottom two layers (indices 0–7) are fixed with `FixAtoms`. Configurations
cover surrogate singlepoint evaluation and FIRE relaxation, plus 6-step NVE
in reference-only, surrogate-only and adaptive modes, with resume and export.
The short trajectories and example budgets do not establish thermodynamic
convergence or speed-ups.

## Required inputs and setup

Run the commands below from this directory, after installing Pyramid with the
`mace` extra and making Quantum ESPRESSO `pw.x` available.

- Place `Al.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE PAW, pslibrary 1.0.0) in a
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

- `make_structure.py` writes `structure.extxyz` with vacuum along z and a
  small displacement of the free atoms. The TOML files fix indices 0–7.
- `run-singlepoint.toml` and `run-relax.toml` use the frozen MACE surrogate.
  Relaxation uses FIRE with `fmax_eV_A = 0.05` and a 100-step cap.
- `run-md-reference.toml`, `run-md-surrogate.toml` and
  `run-md-adaptive.toml` initialize velocities at 300 K with seed 7 and
  run NVE for 6 steps at 1.0 fs, checkpointing every 3 steps.
- The reference and adaptive configurations use PBE + Grimme D3,
  `ecutwfc = 50` Ry, `ecutrho = 400` Ry, a 4×4×1 k-point grid,
  Marzari–Vanderbilt smearing with `degauss = 0.02` Ry, and
  `conv_thr = 1e-8` Ry.
- Adaptive settings: force budget 0.15 eV/Å on free coordinates,
  probe displacements 0.02/0.04 Å, numerical floor 0.001 eV/Å,
  time cap 2 fs, transverse cap 0.1, and check probability 1.0.
  Every accepted force evaluation is checked. Choose accuracy settings
  for your system before extending the trajectory.

## Energy consistency in adaptive mode

With electronic smearing, QE reports variational `free_energy`, while MACE
reports `energy`. This combination is supported when each scalar is
consistent with its own forces and both backends declare
`force_consistent=True` and `forces_conservative=True`. Equal energy-kind
names are not required; an unknown declaration cannot satisfy a cross-kind
requirement. `pyramid validate` checks declarations, not numerical
force-energy derivatives. Keep the reference settings fixed within a run.
See the [method guide](../../docs/energetic_force_error.md) for endpoint-work
semantics and accuracy limits.

## Run, resume and export

```sh
python make_structure.py
pyramid validate run-md-adaptive.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
pyramid run run-md-adaptive.toml
# Continue each 6-step run for 2 additional steps.
pyramid resume runs/al-reference --steps 2
pyramid resume runs/al-adaptive --steps 2
pyramid inspect runs/al-adaptive --json
pyramid export runs/al-adaptive --force-source driving
```

Add `--probe-backends` to `pyramid validate` to evaluate the initial structure
once with each backend. Run directories must be new; use `resume` to continue
an existing run. `Ctrl-C` requests a stop at the next complete-step boundary.
`pyramid inspect` includes reference costs from anchors, probes, checks and
retries; acceptance counts alone do not measure reference savings.

Each configuration starts from the same `structure.extxyz`. To start MD from
the relaxed geometry, export `runs/al-run-relax`, select the final frame and
set `[structure] file` in a new MD configuration to it.

## Force reporting with fixed atoms

The relaxation criterion uses forces projected onto free coordinates. Its
reported `final_fmax_eV_A` can differ from `raw_all_atom_fmax_eV_A`, which
includes the fixed layers' reaction forces. Adaptive MD also measures its
force budget on free coordinates by default (`active_dofs_max_atom`);
`all_atoms_max_atom` is an explicit alternative. Raw forces remain recorded,
while fixed atoms have zero driving force and displacement. See the
[configuration guide](../../docs/configuration.md) for force sources and
missing-label export semantics.
