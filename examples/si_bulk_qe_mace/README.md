# Recipe A — bulk Si with QE reference + MACE surrogate (WP08)

Small bulk-Si run (8-atom conventional cell) through the full Pyramid
workflow: singlepoint → fixed-model relax → 6-step NVE in all three modes
(reference-only, surrogate-only, adaptive) → resume → export.
Short by design: it demonstrates mechanics and records costs; it carries no
thermodynamic conclusions.

## Inputs needed on the target machine

- `Si.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE kjpaw PAW, pslibrary 1.0.0) placed in
  `./qe_pseudos/` — the same pseudopotential validated in
  `tests/data/qe_si_scf.out` and `hpc/neimeng/inputs/si_bulk.scf.in`.
- A MACE-MPA-0 medium model file placed at `./mace-mpa-0-medium.model`
  (the `mace_mp` foundation checkpoint; any local path works — the configs
  reference it recipe-relative).
- `pw.x` (QE 7.x) and a python with `pyraimd2` + `mace`/`torch` on PATH.

## Reference settings (identical across all modes)

- PBE + Grimme D3 (QeEngine fixed recipe), ecutwfc 50 Ry / ecutrho 400 Ry,
  k-points 4x4x4, conv_thr 1e-8 Ry, default local-TF mixing.
- Structure: diamond conventional cell a = 5.431 A, one atom displaced
  (make_structure.py), 300 K seeded velocities, 1.0 fs timestep, 6 steps.
- Adaptive: force budget 0.10 eV/A, probes 0.02/0.04 A, checks at p = 1.0
  (every accepted step verified), checkpoint every 3 steps.

## Commands (one node, serial; adjust pw_cmd ranks to the node)

```sh
python make_structure.py
pyramid validate run-md-adaptive.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
pyramid run run-md-adaptive.toml
# resume demonstration: the 6-step adaptive run continues +2 to 8 steps
pyramid resume runs/si-adaptive --steps 2
pyramid inspect runs/si-adaptive --json
pyramid export runs/si-adaptive --force-source driving
```

`pw_cmd` in the `run-md-*.toml` files is `["mpirun", "-np", "28", "pw.x"]`
for a 28-core node — edit the rank count for other shapes. Costs: every
anchor/probe/refusal/check SCF lands in `events.jsonl` and `summary.json`.

Interrupted-run variant (as exercised in WP08): copy `run-md-adaptive.toml`,
change `run.id`, `run.directory` and `steps = 4`, run it, then
`pyramid resume <new run dir> --steps 2` to reach 6 steps. The WP08 record
used a separate 4-step config (`runs/si-adaptive-short`); the main run
above resumes 6 -> 8.

## Note: stages are independent demonstrations

Every config reads the same `structure.extxyz` written by
`make_structure.py`. The relaxed final structure is **not** fed into the
MD configs — singlepoint, relax and the MD modes each start from the same
displaced initial geometry (the MD demonstration does not need a
pre-relaxed start; if you want one, export the relax trajectory frame and
point `[structure] file` at it explicitly).

## Note on pseudo_dir and QE line lengths

QE parses ATOMIC_SPECIES card lines with a limited buffer (~80 chars); a
long absolute pseudopotential path silently truncates into an unreadable
filename (the reported "file not found" then shows garbage). Stage the
pseudopotentials to a short node-local path (`/tmp/ps`) before running and
keep `pseudo_dir = "/tmp/ps"` in the configs; the engine writes basenames
for pseudopotentials inside pseudo_dir (WP08 engine fix), so both the
namelist and the species lines stay short.

## What this demonstration does and does not show

It shows the integration running on a real material (surrogate acceptances,
re-anchoring, checks, interrupted resume) — not a speedup. On the recorded
6-step point, reference-only cost 7 SCF / 80.3 s while adaptive cost 19
reference executions / 302 s (short runs are probe-dominated); the 57%
acceptance rate is not a 57% DFT saving. Run records backing these numbers
are in `docs/development_reports/wp08_evidence/`.
