# Recipe B — Al(111) slab with fixed bottom layers, QE reference + MACE (WP08)

A 17-atom Al(111) 2x2 slab (4 layers + 1 adatom) with the bottom two layers
fixed (`[constraints] fix_atoms_indices = [0..7]`), through the constrained
workflow: singlepoint → relax → 6-step NVE in the two plain modes
(reference-only, surrogate-only) → plain resume → export. This is the
constrained-workflow materials users run every day; short by design, no
thermodynamic conclusions.

## Adaptive mode: not supported for this recipe in 0.4.0

Do **not** run `run-md-adaptive.toml` expecting a calculation: the adaptive
combination is rejected at `pyramid validate` by the WP01 energy-convention
contract. QE with Marzari-Vanderbilt smearing on a metallic slab reports
its energy as `free_energy`, while the MACE surrogate reports `energy`;
anchored bookkeeping may not mix the two quantities, so validation stops
before any SCF. The recorded rejection message is in
`docs/development_reports/wp08_evidence/al/contract-evidence.txt`. The
config is kept in the directory so the limitation is reproducible:

```sh
pyramid validate run-md-adaptive.toml   # exits with the contract error (by design)
```

Lifting the limitation requires the refined reference-functional contract
sketched in INDEPENDENT_REVIEW_20260909 §5 (verified force–energy
consistency per side, explicit reference functional identity) — a deliberate
change, not a config tweak. Until then the supported workflow for this
recipe is the plain one below.

## Inputs needed on the target machine

- `Al.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE kjpaw PAW, pslibrary 1.0.0) placed in
  `./qe_pseudos/`.
- A MACE-MPA-0 medium model file at `./mace-mpa-0-medium.model`.
- `pw.x` (QE 7.x) and a python with `pyraimd2` + `mace`/`torch` on PATH.

## Reference settings (identical across all modes)

- PBE + Grimme D3, ecutwfc 50 Ry / ecutrho 400 Ry, k-points 4x4x1,
  Marzari-Vanderbilt smearing degauss 0.02 Ry (metallic slab), conv_thr
  1e-8 Ry.
- Structure: Al(111) 2x2x4 + adatom (make_structure.py), bottom two layers
  (indices 0-7) fixed, 300 K, 1.0 fs, 6 steps.

## Commands (one node, serial; adjust pw_cmd ranks to the node)

```sh
python make_structure.py
pyramid validate run-md-reference.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
# plain resume: the 6-step reference-only run continues +2 to 8 steps
pyramid resume runs/al-reference --steps 2
pyramid inspect runs/al-reference --json
pyramid export runs/al-reference --force-source driving
```

## Note: stages are independent demonstrations

Every config reads the same `structure.extxyz` written by
`make_structure.py`; the relaxed final structure is **not** fed into the MD
configs. To start MD from the relaxed geometry, export the relax final
frame and point `[structure] file` at it explicitly.

## Note on force reporting with FixAtoms

The optimizer's convergence criterion uses constraint-projected forces
(fixed atoms excluded). The recorded WP08 relax run converged with
free-atom (8–16) max |F| = 0.048 < 0.05 eV/A while the *raw* all-atom value
was 0.055 eV/A — the difference is the fixed layers' reaction forces, not
unconverged free atoms. Current releases report the criterion-consistent
value as `final_fmax_eV_A` and keep the raw value as
`raw_all_atom_fmax_eV_A`; the recomputation is in
`docs/development_reports/wp08_evidence/analysis/al_relax_final_fmax.txt`.

## Note on pseudo_dir and QE line lengths

QE parses ATOMIC_SPECIES card lines with a limited buffer (~80 chars); a
long absolute pseudopotential path silently truncates into an unreadable
filename (the reported "file not found" then shows garbage). Stage the
pseudopotentials to a short node-local path (`/tmp/ps`) before running and
keep `pseudo_dir = "/tmp/ps"` in the configs; the engine writes basenames
for pseudopotentials inside pseudo_dir (WP08 engine fix), so both the
namelist and the species lines stay short.
