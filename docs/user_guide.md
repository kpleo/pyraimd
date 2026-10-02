# User guide

The ordered path through Pyramid, from a wheel-only install to verifying a
cheap trajectory against a trusted reference. Details live in
[configuration.md](configuration.md) (every TOML field and CLI command)
and [api.md](api.md) (the Python entry points); this page is the map.

## 1. Install

There is no PyPI release: download the wheel (and/or source) of the
version you actually use from the project's GitHub Releases, then install
it locally. Optional extras come from the same local wheel file:

```sh
pip install pyraimd2-0.8.5-py3-none-any.whl        # the file you downloaded
pip install 'pyraimd2[mace] @ file:///path/to/pyraimd2-0.8.5-py3-none-any.whl'   # MACE surrogate backend
pip install 'pyraimd2[pyscf] @ file:///path/to/pyraimd2-0.8.5-py3-none-any.whl'  # PySCF reference backend
```

The core install (NumPy + ASE only) runs every analytic-backend example
without any external compute software. Optional backends are
imported only when selected in a configuration; a Quantum ESPRESSO
reference additionally needs `pw.x` and pseudopotentials on your machine.

## 2. The offline verify chain (analytic backends)

Everything below works without any external program. The
`harmonic-compare` template emits the complete chain in five files —
three configurations around one shared structure, plus a README:

```sh
pyramid init --template harmonic-compare --output demo
```

Follow `demo/README.md`: validate all three configs, run the reference
plain-NVE arm and the two fixed-model MTS arms (unscaled and scaled),
then `pyramid compare` each completed candidate against the reference.
The template's numbers are demonstration values, not recommendations.

The same chain exists for real backends as a project skeleton:
`pyramid init --template qe-mace-compare --output proj` writes the
QE + MACE three-arm layout (five files incl. its README). It is NOT
directly runnable — edit the QE command, pseudopotentials, local model
path and physical parameters first, run the static checks, and only then
run real compute explicitly on compute-authorized nodes.

## 3. Your own material

Replace the toy structure with your own, carrying explicit initial
momenta (`structure.extxyz` with a `momenta` column) — one file shared by
the reference and candidate configurations, so both arms start from
exactly the same state and no arm re-thermalizes. Keep the reference
recipe identical to the settings you trust (same `xc`/dispersion,
pseudopotentials, cutoffs, k-points); `examples/qe_mace_skeleton/` shows
the QE + MACE option shape, and `examples/qe_mace_mts/` is the three-config
MTS skeleton. Point the surrogate at an explicit local model file.

## 4. Calibrate the scale (offline, closed form)

If you use the `scaled` wrapper, its coefficient comes from your own
paired calibration data — never from inside a run. Prepare one `.npz`
(`reference_forces_eV_A` and `fast_forces_eV_A` of shape
`(n_frames, n_atoms, 3)`, already paired in identical configurations and
atom order, plus `frame_ids`, `reference_id`, `fast_model_id` and
`force_unit = "eV/angstrom"`; see `examples/calibrate_scale/`) from
independent reference evaluations you run explicitly on compute-authorized
nodes, then:

```sh
pyramid calibrate-scale --pairs pairs.npz --output scale.json
```

Copy `scale` into `[surrogate] backend = "scaled"` and record the
file's hash in `calibration_note`. The reported residual RMS is a
training metric over the fitted set only — no generalization bound, and
no MTS outer ratio follows from it.

## 5. Validate before running

```sh
pyramid validate run.toml
pyramid validate run.toml --check-environment
```

`validate` checks schema, structure, paths and the backend contract
without evaluating anything; `--check-environment` is a strictly
read-only local-prerequisites preflight. A missing executable,
pseudopotential or model file is a normal user-environment state that
validation reports — it never downloads anything. The probes
(`--probe-backends`, `--probe-surrogate`) are different: they perform one
REAL backend evaluation. Run those, and the MD runs themselves, only
explicitly on compute-authorized nodes.

## 6. Run and resume

```sh
pyramid run run.toml
pyramid run run_mts.toml
pyramid resume runs/mts --steps 32        # MTS: a multiple of outer_ratio
```

One directory per run; resume continues from the persisted complete-step
(or complete outer-boundary) state with the same physics, model identity
and check stream. For the comparison, the reference arm must cover at
least the candidate's physical-time span, sampled so every complete
candidate boundary has an exact reference time point — the reference may
be denser, never coarser than the candidate's boundaries.

On clusters, `examples/prepared_qe_launcher/` (source tree only, not in
the wheel/sdist) is an OPTIONAL Slurm helper for allocations making many
QE calls under a per-call re-setup wrapper: it prepares the environment
once per allocation and reuses it per call through `reference.pw_cmd`.
The plain default `pw_cmd = ["pw.x"]` has no per-call setup cost. The
prepared state is valid only inside its own allocation — resume in the
same job works as usual; a resubmission must re-prepare, and the changed
command identity then goes through the same checkpoint/cache
compatibility checks as any configuration change.

Driving `EnergeticRunner` directly from Python instead of the CLI: always
finish with `runner.close()` — it releases the event-log writer lock and,
when the runner was asked to handle SIGINT, restores the previous SIGINT
handler, so a finished runner never keeps catching Ctrl-C meant for later
code in the same process. `EnergeticRunner` has no `__enter__`/`__exit__`,
so use try/finally or `contextlib.closing`:

```python
runner = EnergeticRunner(atoms, surrogate, engine, store, "run", ...)
try:
    runner.run(100)
finally:
    runner.close()

# or: with contextlib.closing(EnergeticRunner(...)) as runner: ...
```

## 7. Inspect and compare

```sh
pyramid inspect runs/reference
pyramid compare runs/reference runs/mts
pyramid compare runs/reference runs/mts-scaled --json
```

`compare` reads COMPLETED trajectories only, reports pointwise
position/velocity RMS over the matched window and per-trajectory
Hamiltonian drift, and applies pass/fail ONLY to thresholds you pass
yourself (`--max-position-rms`, `--max-velocity-rms`). Without thresholds
it reports metrics and never declares a run "reliable".

On timing: `inspect`'s `timing` block reports durations of the run's own
RUN_SUMMARY records explicitly — `last_reported_run_wall_time_s` and
`summed_reported_run_wall_time_s` over `reported_segments` valid segments
(`invalid_or_missing_summary_durations` counts records with missing or
invalid durations, never backfilled). After a run of a 20 s segment and a
30 s resume segment this reads **last 30 / sum 50** — and that is NOT an
end-to-end 50 s: queue waits, startup overhead and gaps between separate
invocations are outside the reported records, a crashed segment that
produced no summary is not counted, and old logs with partial records sum
only what is there (no valid summaries at all shows both as null). The
top-level `wall_time_s` remains, for backward compatibility, the last
RUN_SUMMARY record's raw value. A meaningful speed comparison needs the
same resources over the same physical interval, measured deliberately —
never read a whole-trajectory speedup off a single field.

## 8. Export and clean up

```sh
pyramid export runs/mts --force-source reference
pyramid export runs/mts --force-source base
```

Exports carry an explicit force source (an MTS outer step has no single
driving force, so `driving` is refused there); missing labels are marked,
never zero-filled. If your runs use the managed scratch root
(`[scratch]` or a QE `scratch_root`), `pyramid scratch --root PATH
inspect` shows task states and retention reasons, and `clean` retries
only the archived cleanup-pending attempts — failed or unarchived
attempts are always kept. The persistent-density chain (QE) is inspected
with `pyramid density inspect RUN_DIR`; reclaim candidates are previews,
never deletions.
