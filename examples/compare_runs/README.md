# Comparing two completed runs offline — `pyramid compare`

**What this shows.** How to judge "is the cheaper run accurate enough"
without private scripts: run a reference NVE trajectory and a candidate
(scaled surrogate + fixed-model MTS) from the SAME initial structure, then
compare the two already-completed runs offline at identical physical times.
Everything runs on the builtin analytic harmonic backends — the atoms are
test particles in a toy well, not a real material.

`pyramid compare REFERENCE_RUN CANDIDATE_RUN` is strictly read-only: it
constructs no backend, evaluates nothing, and opens the trajectory
databases through read-only connections — compared run directories stay
byte-identical (no grown database, no sidecar files), and an empty,
damaged or schema-less database is refused up front rather than
initialized. Only committed complete-step states count as trajectory
points (a crashed tail evaluation is reported in the coverage counts,
never compared). Candidate time points must each match a UNIQUE reference
time point within 1e-9 fs — there is no interpolation and no snapping to a
coarser grid, so the reference may be denser but the candidate may not
carry unmatched points. Positions are compared as stored (continuous
coordinates, no alignment); the position/velocity RMS is normalized per
atom, and velocities use `v = p/m * ase.units.fs`. Hamiltonian drift is
reported per trajectory (zeroed at each run's own start) only when every
complete state carries a reference energy label with
`energy_kind="energy"` and `force_consistent=true`; it is descriptive —
this version defines no energy-drift threshold.

By default the command reports metrics only (`criteria_status:
"not_requested"`) and never concludes "reliable" or "recommended".
Thresholds are yours: pass `--max-position-rms A` and/or
`--max-velocity-rms A_PER_FS` to get per-criterion pass/fail (exit 0 when
all requested criteria pass, 1 when one is exceeded or required records
are missing, 2 for out-of-scope or incompatible inputs; `--json` prints
one structured object, errors included).

**Teaching setup (fixed, do not read as a recommendation).** Reference
arm: plain NVE on `harmonic-reference` (k = 1.0, r0 = 0.9), 16 steps of
1 fs → states at every fs. Candidate arm: MTS (inner 1 fs, outer_ratio 4)
with `scaled` (scale = 1.2) around `harmonic-surrogate` (k = 0.8) →
complete outer boundaries at t = 0, 4, 8, 12, 16 fs, a strict subset of
the reference grid. The used fast field is 0.96 of the reference, so the
two trajectories genuinely diverge. These are toy numbers demonstrating
the wiring — not evidence about any material and not suggested values.

## Steps

```sh
python make_structure.py                  # writes structure.extxyz (with momenta)

pyramid validate run_reference.toml
pyramid validate run_mts_scaled.toml

pyramid run run_reference.toml            # run-reference/  (plain NVE, 1 fs grid)
pyramid run run_mts_scaled.toml           # run-scaled-mts/ (MTS boundaries every 4 fs)

# metrics only — no pass/fail conclusion without explicit thresholds
pyramid compare run-reference run-scaled-mts

# explicit criteria: per-criterion pass/fail, exit 0/1
pyramid compare run-reference run-scaled-mts --max-position-rms 1e-4

# one structured object (errors are structured too) for scripts
pyramid compare run-reference run-scaled-mts --json
```

## Expected output shape

The human report names both runs and drivers, restates the scope, verifies
the shared initial state, and summarizes the matched window, both sides'
coverage, the whole-window RMS maxima and the per-trajectory Hamiltonian
drift:

```
compare: reference compare-runs-reference-demo (plain-nve) vs candidate compare-runs-scaled-mts-demo (mts-nve-respa)
  scope             : same-initial-state pointwise comparison of fixed-cell NVE trajectories (stored continuous coordinates; no alignment, no interpolation)
  initial state     : t = 0.0 fs, positions match (max |dq| = 0.000e+00 A)
  initial momenta   : match (max |dp| = 0.000e+00 ASE units)
  matched times     : 5 points, 0.0 .. 16.0 fs (reference complete states 17, candidate 5)
  coverage reference: 17 complete states of 17 committed evaluations; 0 incomplete tail evaluation(s)
  coverage candidate: 5 complete states of 5 committed evaluations; 0 incomplete tail evaluation(s)
  position rms      : max ... A over the matched window (per-atom normalization; per-time array in the JSON report)
  velocity rms      : max ... A/fs over the matched window
  H drift reference: max |(H(t)-H(0))|/N = ... eV/atom over 17 states (descriptive; per-run energy zero)
  H drift candidate: max |(H(t)-H(0))|/N = ... eV/atom over 5 states (descriptive; per-run energy zero)
  criteria          : not requested — metrics reported only, no pass/fail conclusion
```

With `--json` the same report is one object (`schema_version: 1`): the
`time_axis` block shows `"matched_points": 5` against
`"reference_complete_states": 17`, and `position_rms_A` /
`velocity_rms_A_fs` carry the per-time arrays next to the whole-window
`max`. A failed requested criterion exits 1 with `criteria_status:
"failed"`; an out-of-scope or incompatible pair (different atoms, cells,
initial states, or unmatched candidate times) exits 2 with a structured
`error` object (`reason`: `unsupported_scope`, `incompatible_inputs`,
`missing_information` or `usage`).

Swapping the arms is a quick way to see the matching rule: with the dense
reference arm as CANDIDATE, its 1 fs points find no match on the 4 fs grid
and the command refuses — interpolation would invent accuracy the data
does not have.

The same comparison from Python:

```python
from pyraimd2.workflows import compare_runs

report = compare_runs("run-reference", "run-scaled-mts",
                      max_position_rms_A=1e-4)
print(report["criteria_status"], report["position_rms_A"]["max"])
```
