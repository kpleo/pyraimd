# WP08 evidence pack (recovered 2026-09-09)

Lightweight records backing the WP08 materials claims and the
INDEPENDENT_REVIEW_20260909 §4 corrections. Recovered read-only from the
Neimeng A cluster (`~/cloud_projects/pyraimd2/experiments/wp08/`, jobs
7625052/7625058 and the probe series) on 2026-09-09; no new MD/SCF work
was submitted for this pack (added actual SCF count: 0).

## Contents

- `si/`, `al/` — one directory per recipe:
  - `configs/`: the recipe TOMLs as submitted (run-*.toml, including
    `run-md-adaptive-short.toml` used for the interrupted-run demo),
    `make_structure.py`, and the input `structure.extxyz`.
  - `runs/<run>/`: verbatim run records — `config.toml`,
    `resolved_config.json`, `manifest.json`, `events.jsonl`,
    `trajectory.db`, and `summary.*`/`trajectory.extxyz`/
    `export-driving.extxyz` where the run produced them.
  - `inspect-*.json`, `al/contract-evidence.txt`: the inspect dumps and
    the adaptive energy-contract rejection message WP08 cited.
- `analysis/` — recomputations from the records above:
  - `scf_recount.md` / `scf_recount.json`: per-run/per-purpose SCF
    recount (58 logical / 57 actual) and the fingerprint/cache evidence.
  - `al_relax_final_fmax.txt`: Al final-frame fmax split fixed (0–7) vs
    free (8–16) — 0.054590 raw all-atom, 0.048156 free.
  - `al_relax_final_frame.extxyz`: the FIRE final frame with the raw MACE
    force array (source of the recomputation).
  - `si_adaptive_final_frame.extxyz`: the si-adaptive step-8 frame.

## Identity quick reference

- Software: pyraimd2 0.4.0.dev0 (run-time dev checkout); remote env
  Python 3.12.13, torch 2.6.0 CPU, mace 0.3.16, ase 3.29.0, numpy 2.5.1;
  QE 7.5.
- Reference identity (run_start `reference_id` / manifest fingerprint):
  Si `qe-pbe-d3:d23205a4a5469f4f`, Al `qe-pbe-d3:b5880bfa70564c2d`
  (PBE+Grimme-D3, 50/400 Ry, kpts 4x4x4 Si / 4x4x1 Al, conv_thr 1e-8;
  Al adds metallic mv smearing degauss 0.02 Ry).
- Pseudopotentials (resolved configs): `Si.pbe-n-kjpaw_psl.1.0.0.UPF`,
  `Al.pbe-n-kjpaw_psl.1.0.0.UPF` staged under `/tmp/ps` at run time.
- Surrogate identity: `mace-mp:.../macempa0mediummodel:cpu:float64`
  (MACE-MPA-0 medium, local cache file; the ~MB-scale model file itself is
  not part of this pack).

## Provenance policy

Run records under `runs/` and `configs/` are verbatim copies and therefore
keep run-time site paths inside file contents (account home directory,
`/tmp/ps` pseudo staging). Submission profiles — filled sbatch scripts and
the site launcher wrapper (`pwx.sh`, including the 2026-09 `mpirun
--map-by :OVERSUBSCRIBE` workaround) — are deliberately not included; they
stay off-repo per the sensitive-profile rule. Wavefunctions, `.save`
directories and the GB-scale `calculations/` trees remain on the cluster
and were not copied.

## Notes on the records

- `al-run-relax/trajectory.db` stores the final frame twice (rows 17 and
  18, both step 16, same evaluation_id) — a WP08-era relax-driver quirk;
  `run_summary.n_evaluations = 17` counts it once. The fmax recomputation
  uses the last row; both rows carry identical forces.
- The WP08-era store records relax `driving` as the *raw* force array
  (constraint projection for driving/reporting arrived with the R-stage
  review fixes), which is exactly the array the review asked to recompute.
- `si-adaptive-short/events.jsonl` seq 59–62 documents the directory-
  collision failure (0.05 s, no pw.x started), the `run_end failed`, and
  the successful retry after the engine fix.
