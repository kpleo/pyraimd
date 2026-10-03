# calibrate-scale — fit the frozen scale for the `scaled` wrapper

**What this shows.** How the `scaled` surrogate wrapper's coefficient is
produced OFFLINE, before any run: one closed-form force least-squares fit
over paired reference/fast forces, no backend constructed, nothing
launched. The data is analytic teaching data (toy atoms in a harmonic
well, reference forces exactly 1.25 times the fast ones) — not material
data, not a fitted recommendation.

## Steps

```sh
python make_pairs.py                          # writes pairs.npz (3 frames x 2 atoms)
pyramid calibrate-scale --pairs pairs.npz --output scale.json
```

Expected: `scale = 1.25` with the after-scaling training residual RMS at
zero (the teaching relationship is exactly collinear; real calibration
data is not, and the report then shows nonzero training metrics).

## The pairs file protocol

One `.npz` (loaded with `allow_pickle=False`):

- `reference_forces_eV_A`, `fast_forces_eV_A`: float arrays of shape
  `(n_frames, n_atoms, 3)`, already paired — every index is the SAME
  configuration in the SAME atom order on both sides. Preparing that
  pairing is the user's job: independent reference evaluations of the
  configurations the fast model was run on (or vice versa). The tool
  never discovers, infers, reorders or splits data: ALL frames enter the
  fit.
- `frame_ids`: unique Unicode strings, one per frame — the explicit
  declaration of the pairing.
- `reference_id`, `fast_model_id`: non-empty strings naming the two
  sides (any stable label you choose; they are recorded verbatim).
- `force_unit`: must be exactly `eV/angstrom`.

## Using the result

`scale.json` (schema_version 1) records the scale, numerator,
denominator, frame/atom counts, per-atom training residual RMS before
and after scaling, and the input's basename + content sha256. Copy
`scale` into your candidate configuration, and note the record's hash
for provenance:

```sh
shasum -a 256 scale.json
```

```toml
[surrogate]
backend = "scaled"
scale = 1.25                  # from scale.json — frozen for the whole run
calibration_note = "scale.json sha256:<paste the digest>"
base = { name = "mace", kwargs = { model = "models/user.model" } }
```

Nothing here edits run.toml, evaluates a model or starts a run: real
calibration data comes from reference evaluations the USER runs
explicitly on compute-authorized nodes, and the wrapper then keeps the
scale constant in production (it multiplies energy and forces together).
The training residual RMS is a metric over the fitted set only — it is
not a generalization bound, and no MTS outer ratio follows from it.

## From an existing MTS run: `export-pairs`

If you already have a completed fixed-model MTS run (`task.mode = "mts"`),
its store holds the reference and fast-model forces at the same
configurations — exporting them needs no new reference evaluation, no
model loading and no array hand-conversion:

```sh
# one uncalibrated analytic MTS run to see the whole chain (offline):
pyramid init --template harmonic-compare --output demo
pyramid run demo/run_mts.toml

# export three saved states by their evaluation ids (1 is the initial
# configuration, stored at step -1; the outer boundaries follow)
pyramid export-pairs demo/runs/mts --evaluation-ids 1 2 3 --output pairs.npz

# the same fit as above — here it returns 1.25, the analytic teaching
# coefficient of the template (reference k = 1.0 over fast k = 0.8)
pyramid calibrate-scale --pairs pairs.npz --output scale.json
```

`export-pairs` is strictly read-only: the run directory is never
modified, only committed complete-step states pair, correction-wrapped
runs (`scaled` / `quadratic-corrected`) are refused because their stored
labels are not raw base forces, and the output must live outside the run
directory. All exported frames enter the fit; whether to hold out a
separate validation set is your own later choice — nothing auto-splits,
applies the coefficient for you, or recommends an outer ratio. The NPZ
also carries positions/cell/numbers and a `provenance_json` record
(source file hashes, identities, per-frame evaluation/step/row binding)
— provenance-by-record, not a certification that the labels are
physically accurate.
