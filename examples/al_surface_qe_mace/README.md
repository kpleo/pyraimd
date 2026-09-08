# Recipe B — Al(111) slab with fixed bottom layers, QE reference + MACE (WP08)

A 17-atom Al(111) 2x2 slab (4 layers + 1 adatom) with the bottom two layers
fixed (`[constraints] fix_atoms_indices = [0..7]`), through the full
workflow: singlepoint → relax → 6-step NVE in all three modes → interrupted
resume → export. This is the constrained-workflow materials users run every
day; short by design, no thermodynamic conclusions.

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
- Adaptive: force budget 0.15 eV/A on the free coordinates
  (active_dofs_max_atom), probes 0.02/0.04 A, checks at p = 1.0,
  checkpoint every 3 steps.

## Commands (one node, serial; adjust pw_cmd ranks to the node)

```sh
python make_structure.py
pyramid validate run-md-adaptive.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
pyramid run run-md-adaptive.toml
pyramid resume runs/al-adaptive --steps 2
pyramid inspect runs/al-adaptive --json
pyramid export runs/al-adaptive --force-source driving
```

## Note on pseudo_dir and QE line lengths

QE parses ATOMIC_SPECIES card lines with a limited buffer (~80 chars); a
long absolute pseudopotential path silently truncates into an unreadable
filename (the reported "file not found" then shows garbage). Stage the
pseudopotentials to a short node-local path (`/tmp/ps`) before running and
keep `pseudo_dir = "/tmp/ps"` in the configs; the engine writes basenames
for pseudopotentials inside pseudo_dir (WP08 engine fix), so both the
namelist and the species lines stay short.
