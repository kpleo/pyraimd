# Supplementary Data 1: force-error and residual-work data

Numerical dataset and reproduction package version **1.2.0**.
Previously bundled arrays and values are preserved; this version
adds the molecular and controlled records described below.

This package contains numerical extracts from existing atomistic records and
explicitly identified derived source data. It supports recalculation of local
directional responses, signed residual work, discrete force-accuracy horizons,
numerical sensitivity checks, and the specified molecular and tungsten comparisons.
It contains neither complete electronic-structure outputs nor a complete simulation
restart environment. No new simulations were performed to assemble it.

## Data and code

The analysis scripts are included in the [parent directory](../README.md).
Data, code, requirements and expected outputs are available in the same GitHub
checkout; no separate download is required. The original data and code deposits
are retained as archival identifiers in `metadata.json`. No separate data license
is specified in that metadata; the analysis code uses the MIT license.

`metadata.json` describes systems, units, settings, coverage and exclusions.
`schema.json` lists every NPZ array's shape and dtype and every CSV column.
NPZ archives contain only finite numeric or boolean arrays and can be read with
`numpy.load(filename, allow_pickle=False)`. Atom and record indices are zero-based.
Cells use three row vectors. Forces are in eV/angstrom, energies in eV, positions
and cells in angstrom, time in fs, velocities in angstrom/fs, and masses in atomic
mass units. Integer flags use 0/1. JSON metadata supplies chemical symbols and
categorical definitions separately from the arrays.

## Interface coverage

The interface contains 474 atoms, formula C114H186F18Li39O114P3, in a fixed periodic
cell. Two origin configurations each have two velocity directions. All four paths
retain their complete 33-state primary trajectory from 0 to 4 fs at 0.125 fs.
The reference data contain 18 development geometries (two origins plus 16 signed
probes), all 40 future endpoints, and six additional numerical-check calculations.
Origins shared by two paths are intentionally repeated in the paired archives.

- `interface/probes_0.npz` and `probes_1.npz`: one origin each, two directions per
  origin, and central displacements at 0.02 and 0.04 angstrom. Origin positions,
  base/reference forces, normalized full-configuration directions, actual signed
  displacements and paired probe forces are retained. The actual displacements
  preserve coordinate rounding; the maximum component difference from the intended
  signed probe is 9.883e-9 angstrom.
- `interface/motion_0.npz` through `motion_3.npz`: only `time_fs` and unwrapped
  `positions_A`, as required by the prediction CLI. Extra metadata is kept separate.
- `interface/labels_0.npz` through `labels_3.npz`: times, matching coordinates,
  base/reference force vectors and energies, kinetic energies and fixed group IDs.
  The row counts are 11, 11, 13 and 11. Every path includes the origin and future
  times 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3 and 4 fs. Path 2 also includes
  reference labels at 0.625 and 0.875 fs.
- `interface/metadata/system.npz`: atomic numbers, atom indices, masses, periodic
  flags, cell and fixed groups. `reference_atom_index` is the identity permutation:
  every coordinate and force array retains the same atom order.
- `interface/metadata/initial_states.npz`: four initial positions, initial velocities
  converted from stored momenta, kinetic energies and the path/probe/direction map.
  The initial kinetic energy is approximately 36.683975 eV per path, corresponding
  to the recorded 600 K normalization with 1419 degrees of freedom.
- `interface/metadata/probe_record_mapping.npz`, `label_record_mapping.csv` and
  `reference_convergence.csv`: explicit mapping to the 64 reference calculations,
  with stored final SCF accuracy estimates, thresholds and printed energies.
  Reference families 0, 1 and 2 mean development, future and additional check;
  each family has its own record numbering. Positive/negative probe index arrays
  have axes `(probe archive, local direction, displacement scale)`.

Path-to-CLI mapping:

- Path 0: `probes_0.npz`, `motion_0.npz`, `labels_0.npz`, direction `0`.
- Path 1: `probes_0.npz`, `motion_1.npz`, `labels_1.npz`, direction `1`.
- Path 2: `probes_1.npz`, `motion_2.npz`, `labels_2.npz`, direction `0`.
- Path 3: `probes_1.npz`, `motion_3.npz`, `labels_3.npz`, direction `1`.

The base force is the retained frozen committee prediction. Its origin correction
is `c = F_reference(0) - F_base(0)` and remains fixed for a whole path. Dynamics use
velocity Verlet, fixed cell, unwrapped coordinates, and no thermostat or constraints.
Reference labels use Quantum ESPRESSO PBE+D3 PAW, 60/600 Ry wavefunction/density
cutoffs, Gamma sampling, 942 bands, Fermi-Dirac smearing at 0.02 Ry, and an SCF
threshold of 1e-8 Ry, with the paired tightened check at 1e-10 Ry. Reference energies
are the fixed-smearing variational E-TS consistent with the printed forces.

The empirical prediction uses a 0.25 eV/angstrom force budget, numerical floor
0.00025 eV/angstrom, 1-fs time cap and 0.1 transverse-fraction cap. Its discrete
admitted-prefix horizons are 0.375, 0.25, 0.25 and 0.375 fs. All 40 future endpoints
are retained: 22 satisfy the geometric/time domain and 10 are admitted by the
prefix rule. Labels outside either criterion remain diagnostic data. Finite probe
estimates are empirical coefficients, not certified global derivative bounds.

## Interface numerical checks and derived records

- `interface/checks/reference_checks.npz` retains all six additional reference
  calculations with positions, paired forces/energies, kinetic energies, times,
  timesteps and SCF thresholds. Kind codes are 0 tightened origin, 1 tightened
  endpoint, 2 added primary-path quadrature point and 3 half-timestep endpoint.
- `interface/checks/scf_tightened_2.npz` contains the origin and 1-fs endpoint with
  tightened reference labels. The standard residual-work CLI reconstructs its
  correction from the tightened origin, so its work is **reanchored**.
- `interface/checks/half_step_labels_0.npz` and `half_step_labels_2.npz` contain the
  unchanged standard origin plus each half-timestep 1-fs endpoint. These permit
  endpoint-work comparisons for 0.0625 versus 0.125 fs integration. The two-state
  trapezoid is not the fine quadrature of the half-timestep path.
- `interface/checks/dynamics_energy_0.npz` through `dynamics_energy_3.npz` contain
  base and kinetic energies at all 33 primary and 65 half-timestep states, plus the
  derived correction projection `c dot (X-X0)`. Recalculate anchored total energy
  as `H = base_energy - correction_projection + kinetic_energy`, and subtract its
  origin value. `timestep_energy_summary.csv` compares the maximum absolute drifts
  at common primary-grid times. Half-step drift ratios are approximately one quarter.
  The scalar half-timestep series does not imply reference labels at all its times.
- `interface/derived/endpoints.csv` contains recalculated values for every future
  endpoint, including force error, endpoint work, prediction, envelope, domain and
  admission flags, and both Hamiltonian diagnostics.
- `interface/derived/quadrature.npz` contains all 29 complete signed atomic work
  integrals. Each row identifies path, endpoint and grid spacing. Atomic contributions
  sum to the total; `group_ids` groups the same atoms. Three incomplete grids are
  listed in `interface/checks/numerical_checks.json`: paths 0, 1 and 3 lack labels
  at 0.625 and 0.875 fs for the 0.125-fs grid ending at 1 fs. Missing labels are not
  interpolated or counted as successful quadratures.
- `interface/checks/numerical_checks.json` contains recalculated SCF/timestep
  sensitivities and the path-2, 1-fs numerical reference values. At that endpoint,
  `W = 0.052808734295215834 eV`, the 0.125-fs trapezoid is
  `Q = 0.0527638285524305 eV`, and `Q-W = -4.4905742785331104e-5 eV`.
  This endpoint is outside the primary admitted prefix.

Signed endpoint work is
`W = delta(U_reference) - delta(U_base) + c dot (X-X0)`.
Residual forces are `R = F_base + c - F_reference`, and the trapezoid is
`Q = sum_j (R_j+R_(j+1))/2 dot (X_(j+1)-X_j)`.
Fixed-c tightening changes only the reference-energy increment. Reanchoring adds
`(F_reference_tight(0)-F_reference_standard(0)) dot (X-X0)` to that change.
The stored fixed-c energy difference is zero at the retained output precision;
the reanchored work change is approximately -3.437896e-5 eV. These are operational
numerical-resolution checks, not total error bounds.

Groups retain initial chemical identities: atoms 0–35 lithium slab, 36–245 ethylene
carbonate, 246–449 dimethyl carbonate, and 450–473 salt. Atomic and group force-work
partitions are pathwise signed integrals, not independent atomic electronic energies
or a causal decomposition. Atom 157 is carbon; its path-2, 1-fs fine-grid integral is
approximately 0.000644168653 eV.

## Recalculate with the public analysis code

Use Python 3.10 or later with NumPy. From this `data/` directory, the code
is in its parent directory. Install dependencies following [the main README](../README.md),
then run:

```sh
python ../run_demo.py --data .
```

All output files are newly calculated; use a fresh output directory for the
individual CLIs because they do not overwrite results.

The demonstration checks the controlled reference-call counts, the path-2
directional coefficient and horizon, endpoint work, force integral and carbon-157
contribution against stored reference values. A successful run prints
`"status": "passed"`; its intermediate outputs are temporary. The commands below
retain the individual analysis outputs instead.

```sh
CODE=..
python "$CODE/directional_response.py" estimate interface/probes_1.npz --output results/response_1.json
python "$CODE/directional_response.py" predict interface/motion_2.npz --response results/response_1.json --direction 0 --budget 0.25 --floor 0.00025 --time-cap 1 --transverse-cap 0.1 --output results/predictions_2.json
python "$CODE/residual_work.py" analyze interface/labels_2.npz --prediction results/predictions_2.json --output results/work_2.json
python "$CODE/residual_work.py" analyze interface/labels_2.npz --end 1 --spacing 0.5 --output results/work_coarse.json
python "$CODE/residual_work.py" analyze interface/labels_2.npz --end 1 --spacing 0.25 --output results/work_medium.json
python "$CODE/residual_work.py" analyze interface/labels_2.npz --end 1 --spacing 0.125 --output results/work_fine.json
python "$CODE/residual_work.py" analyze interface/checks/scf_tightened_2.npz --end 1 --output results/work_reanchored.json
python "$CODE/residual_work.py" analyze interface/checks/half_step_labels_2.npz --end 1 --output results/work_half_step.json
```

Repeat the first three commands with the path mapping above to recover the other
directions. Without `--spacing`, integration uses all supplied reference-labelled
points, which become nonuniform after 1 fs. It does not infer a converged integral
over an unlabelled dense trajectory. The code's controlled-dynamics calculation is
self-contained and does not require this data package. The v1.2.0 additions below
include check outcomes for the forward molecular paths and four retained
statistical histories. The original 302-record water replay does not gain missing
decision-time force vectors or historical fitted states.

## Molecular source data

`water/reference_frames.npz` contains **all 302 stored records of one isolated H2O
molecule**, with three-atom coordinates, cell, periodic flags, atomic identities,
masses, velocities, reference energies and full reference force vectors. It is not
a bulk liquid-water dataset. The retained collection implementation uses a frozen
MACE-MP-0 small potential, 300 steps of velocity Verlet at 0.5 fs with an initial
300 K kinetic normalization, and PySCF RKS PBE/def2-SVP reference labels (SCF
tolerance 1e-9). Records 0 and 1 are the same initial state: an explicit initial
copy and the integrator's initial callback. Both are preserved. `integration_step`
and `time_fs` reconstruct this convention, ending at step 300 and 150 fs; they are
not independent stored timestamps. `record_index` maps directly to both CSVs.

`water/decisions_conformal.csv` and `decisions_scheduled.csv` each contain all 302
stored decision-time scalar errors, spreads and route flags at the common
0.25 eV/angstrom budget. They reproduce 0 violations among 227 accepted decisions
and 21 among 226, respectively. These are chronological replays over a shared
fully labelled trajectory, not two separately integrated production trajectories.

`accepted` means the approximate force route was selected; `accepted_violation`
means that route's error exceeded the budget; `updated_after_record` indicates a
parameter update after that record. `qhat` is the error/spread calibration factor
and `bound_eV_A` is the decision bound. Their `_state` columns mean 0 finite,
1 positive infinity (cold start), and 2 undefined. In states 1 and 2 the numeric
entry is a zero placeholder, never a physical zero bound. The scheduled policy
does not supply a calibration bound.

The scalar decision records are processed comparison source data. The decision-time
approximate force vectors and intermediate fitted states are not present, so the
scalar force errors cannot be independently reconstructed from reference frames
alone. Counting violations and checking the recorded decisions require no refitting.

## Tungsten source data

The fixed comparison comprises **24 distinct 432-atom geometries and 27 reference
labels**, with a fixed 40-atom core and 392-atom matrix. Labels include four separate
reference evaluations of the common initial geometry; none is discarded or averaged.

- `tungsten/geometries.npz`: coordinates, atomic numbers, cell, periodic flags,
  core mask, times, segment codes and representative record indices. Segment codes
  are 0 shared initial geometry, 1 bulk 300 K, 2 bulk 3000 K, 3 bulk 6000 K and
  4 localized spike. Segments are separate paths, so the archive is not a single
  uniformly sampled trajectory.
- `tungsten/reference_labels.npz`: all 27 reference energies/force arrays, record
  indices, and explicit `geometry_index` mappings to the coordinates. The retained
  reference is Quantum ESPRESSO PBE+D3 with 3074 bands and Marzari-Vanderbilt
  smearing at 0.01 Ry.
- `tungsten/base_predictions.npz`: stored predictions for all 24 geometries,
  including both members' forces/energies, equal-weight means and per-atom spreads.
  Members are MACE-MP-0b3 medium and MACE-MPA-0 medium, with the second member's
  recorded seeded readout perturbation 0.01. These arrays are numerical predictions,
  not parameter files.
- `tungsten/stock_pair.npz`: the two unperturbed members' stored predictions and
  reference forces at geometry indices 0 and 20 (shared origin and 50-fs spike).
- `tungsten/d3_pair.npz`: additive D3 forces/energies at those same two geometries,
  the PBE+D3 reference forces and the derived difference `F_PBE+D3 - F_D3`.
  Removing an additive force term is not a new self-consistent PBE calculation.

To recover a label residual, use its `geometry_index` to select the mean prediction
and subtract its reference vector. The maximum atom norm is
`max(sqrt(sum(residual**2, axis=-1)))`; the vector RMS is
`sqrt(mean(sum(residual**2, axis=-1)))`. Select the fixed `core_mask` or its complement
for the two groups. Use energy increments only within a stated segment/reference
convention; no absolute energy-offset fit is supplied.

## Fig. 3 velocity interventions and probes

`water_velocity/trajectories.npz` has axes `(path, time, atom, component)` for
vector arrays, with six paths, 20 times and three atoms. Path order is seeds
2026090501 and 2026090502, each at velocity factors 0.5, 1 and 2. These seeds use
initial records 0 and 151 in `water/reference_frames.npz`. The `time_fs` axis is
0, 0.5, ..., 9.5 fs. Atomic identities, masses, positions, full-step momenta,
surrogate/reference forces, reference energies and recorded maximum-atom errors
are included. `momenta_sqrt_u_eV` retains the original ASE momenta, in
sqrt(atomic-mass-unit times eV). `velocities_A_fs` is derived as momentum/mass
times `ase_fs_in_internal_time` from `settings.json`. No velocity or fit state
is generated by model inference.

`probes.npz` supplies two anchor positions, full-configuration unit directions,
steps 0.001 and 0.002 angstrom, and signs [-1,+1]. The retained residual derivative
array has axes `(anchor, scale, atom, component)` and units eV/angstrom², with
residual convention `F_surrogate - F_reference`. The eight reference-force and
energy arrays have axes `(anchor, scale, sign, ...)`. Probe positions are explicitly
named `reconstructed_positions_A`: they are calculated from the retained anchor,
unit velocity and signed step. Each reconstructed coordinate set matches its
original reference geometry digest in `probe_mapping.csv`. The digest uses
SHA256 of the atomic-number list, a vertical bar, and semicolon-separated Bohr
coordinates printed to ten decimal places (no spaces around coordinate commas).
The recorded Bohr conversion is in `settings.json`.

Individual **surrogate forces at the signed probes were not retained**. The
derivative vectors therefore support independent norms and two-spacing
comparisons, but not a new reconstruction from both signed surrogate force
arrays. The reference forces alone do not fill this gap.

The retained figure-source CSVs are:

- `motion_cases.csv`: six path summaries; blank crossing times mean no observed
  exceedance through 9.5 fs. The horizons are [2,0.5,0,1,0.5,0] fs. Zero means
  refusal of the first forecast step, not an initial force-budget violation.
- `motion_points.csv`: 120 saved force-error values, polygonal path lengths,
  adjacent-state residual secants and empirical envelope values.
- `motion_forecasts.csv`: all 14 retained forecast times, lengths and bounds,
  including the first failed forecast points. No longer forecast series is inferred.
- `motion_directional_probes.csv`: four recorded maximum-atom derivative norms.

`settings.json` includes the recorded initial residual, growth coefficient and
horizon for each path, plus original foundation-file/checkpoint hashes. The
coefficient is a **retained empirical value**: its historical same-model residual
vectors were not saved, so independent refitting is unavailable. The saved
forecasts match later path coordinates; the records do not independently
timestamp a pre-reference commitment. Neither finite secants nor local probes
certify force errors between saved states.

## Fig. A2: 48-path molecular comparison

`molecular_comparison/paths.csv` defines the zero-based path order: temperatures
300, 600 and 1200 K; within each, seeds 2026090541–2026090544; within each seed,
policies reference, periodic, calibrated and horizon. Seeds start at records
0,151,0,151 respectively. The initial velocity factors are 1, sqrt(2) and 2.
These are initial kinetic normalizations with six nontranslational degrees of
freedom, not equilibrium temperatures. There are four independent fresh
directions, paired across policies and speeds.

`trajectories.npz` retains every one of the 50 states per path, 0–24.5 fs at
0.5-fs spacing. Vector arrays have axes `(48,50,3,3)`. It contains positions,
full-step momenta, reference/surrogate/driving forces, independently retained
pre-check force proposals, reference and kinetic energies, error/spread/bound
scalars, and acceptance, check, request, new-reference and violation flags.

- `accepted` and `proposal_accepted` retain the pre-check force choice.
- `checked` is the saved independent-check outcome, restricted to accepted states.
- `reference_requested = (~accepted) | checked` includes refused states and
  verification calls. This is the policy cost, not all hidden measurement labels.
- `accepted_violation = accepted & (maximum_atom_error > 0.10)`.
- `new_reference` counts hidden measurement evaluations as well as policy calls:
  49 new labels and one reused initial label per path, 2352 new evaluations total.
- The surrogate is never evaluated in the reference-only policy. Its force and
  proposal arrays use zero placeholders with `surrogate_available` and
  `proposal_available` false. Its recorded surrogate error/spread zero is also
  unavailable, not evidence of a perfect surrogate.
- Scalar bound fields ending `_state` use 0 finite, 1 positive infinity and
  2 undefined; the associated numeric value is zero for states 1 and 2.

`metrics.csv` supplies all 48 retained per-path outputs. In particular,
`online_requests` is the reference count for Fig. A2a, `OH_bond_RMSE_A` is
Fig. A2b, `accepted/states` gives Fig. A2c, and `risk` gives Fig. A2d.
Multiply fractions by 100 to plot percentages. Bond RMSE pools the two O–H
bonds and all 50 paired time points against the reference trajectory with the
same seed and speed. Angle RMSE is in degrees. Empty risk/bound entries denote
no accepted states. All 12 reference paths and three horizon paths have undefined
accepted risk; zero reference self-deviation is expected.

`anchors.csv` retains the current reference error, empirical coefficient and
announced horizon at each recorded horizon-policy anchor. `forecasts.csv` links
to `anchor_index` and gives each recorded elapsed time, cumulative configuration
path length and `e0 + kappa*length` bound. Coefficients have units eV/angstrom².
These files preserve existing estimates without supplying missing fitting vectors.

`settings.json` records the 0.10-eV/angstrom budget, 0.1 check probability,
period 4, calibrated multiplier 1, initial 32 calibration pairs (spread, error),
and frozen foundation/checkpoint identifiers. The four-direction screening rule
requires at least 20% acceptance, at most 5% observed accepted violations and
at most 90% reference requests in every direction at a speed. No speed passes
the horizon rule; calibration passes at the original speed. The per-path
sequential upper bound is 100% whenever the acceptance count is nonzero.
Empirical observation of zero violations is not a tighter confidence certificate.

## Eight-path hot-forward table

`hot_forward/` uses the same archive and table schemas, with eight paths of
40 states each over 0–19.5 fs. `paths.csv` maps seeds 2026090521 and 2026090522
at the two known origins, each with reference, periodic, calibrated and horizon
policies. The velocity factor is 2, the selected period is 8, and the calibrated
multiplier is 1. There are 312 new reference evaluations and eight reused initial
labels. `metrics.csv` supplies the hot-forward table's acceptance, violation,
reference-request and maximum reference-Hamiltonian-drift entries. Drift is in
meV per molecule. Counts and all numerical values retain the original outcomes.
The horizon accepts no states on either path; it follows the reference forces.

Both molecular collections omit fitted weights and full historical calibration
states. `model_sha256` and `checkpoint_sha256` are recorded file identities,
not a claim that weights are bundled. Checks of saved proposal equality and
decision arithmetic cannot independently establish the temporal ordering or
independence of the original random draws. Force work in these finite-timestep
molecular records uses trapezoidal residual power; its discrepancy from the
reference-Hamiltonian change is retained in `max_work_balance_residual_meV`.

## Oscillator and statistical supplementary sources

`controlled/` uses reduced oscillator units (unit mass and surrogate stiffness);
these quantities are not eV, angstrom or fs.

- `oscillator_trajectories.csv` contains all 80 recorded initial conditions,
  trajectory-level costs and violations. Its zero-based row order defines
  `trajectory_index` in the following NPZ files.
- `oscillator_intervals.npz` contains all 160,000 original interval outcomes:
  trajectory/step indices, route codes, endpoints in time, starting position and
  velocity, applied bias, maximum interval error and violation flag, envelope
  diagnostics and relative energy change. Decode routes using `settings.json`.
  Selecting the unreferenced route reproduces the interval-error distributions
  and the class-horizon, fixed-streak and out-of-class comparisons. Stored maximum
  interval errors include interior extrema; no new oscillator is propagated.
- `oscillator_segments.npz` contains all 10,518 fixed-model segments, including
  their anchor positions, remaining anchor residual, bias, total work and
  anchor-residual/curvature contributions. Check
  `W = residual_at_start*(x_end-x_start) + (k-1)*(x_end-x_start)^2/2`.
- `oscillator_trajectory_work.csv` contains the 80 work-energy summaries.
  `oscillator_work_components.csv` contains the 16 Fig. 3d bars. Its
  `trajectory_index` is explicitly **one-based display order**, 1–16, and its
  seed maps to the held-out class-horizon trajectory in the 80-row table.
  Contributions are divided by each trajectory's initial reference energy before
  averaging. The means are 0.154209422508043 and 0.36513137934334783.
- `closed_form_checks.json` retains positive, zero, negative and indefinite
  curvature examples, including repeated matching and a nonquadratic remainder.
- `verification_replicates.npz` retains all 20,000 original terminal outcomes and
  first-crossing times. SeedSequence entropy is 20260905 and spawn key is
  `(replicate,)`. A first-crossing value of -1 means none was observed.
- `verification_traces.csv` contains all 1024 steps for each of the first four
  repetitions, including binary A/Y/Z and cumulative N/V/D. These support the
  displayed bound trajectory and independent martingale arithmetic.
  **Full histories of the other 19,996 repetitions were not retained here.**
- `verification_summary.json` preserves the original counts, separate marginal
  Clopper–Pearson intervals and deterministic finite-time probability bracket,
  including the exact numerator and denominator of its analytical gap bound.
  The original events are 682 and 1012; the latter finite-sample proportion
  exceeds its 0.9^29 comparator. This outcome is retained without resampling.
- `settings.json` records the original physical/statistical parameters, category
  codes and reference-cost accounting.

To check all these records without running a simulation, from `data/` run:

```sh
python ../supplementary_records.py --data .
```

The checker verifies exact count identities; force, work, forecast and geometry
arithmetic use absolute tolerances of 1e-12 in their stated units, with 1e-9 meV
for energy-drift metrics that subtract extensive energies. It uses no relative
tolerance. The exact analytical probability-gap arithmetic uses 40 decimal
digits and is compared to its retained value at 1e-35 relative tolerance.
Reported Clopper–Pearson endpoints are retained original outputs; the checker
does not rerun their interval construction or the deterministic state recursion.

## Tungsten hashes, data traceability and license

`tungsten/model_identity.json` distinguishes SHA256 of each original model file
from SHA256 of the effective model's parameter/buffer tensors. The evaluation
retains the seeded 1% second-member readout/products perturbation; the stock
control uses the unperturbed files. Tensor hashing iterates sorted state-dictionary
names and appends a compact JSON [name,dtype,shape] header and contiguous array
bytes for each entry. The hashes were copied from the original authoritative
identity records without loading a model. Historical remote trajectory-generation
binary hashes are unavailable; foundation-pretraining overlap is unknown.

`source_records.json` provides source-collection IDs, original basenames, byte
sizes and SHA256 digests for the numerical extracts and model identities. It
contains no machine paths, raw execution logs or working notes. The source
digests identify original records; the package checksums independently protect
the portable exports. `schema.json` inventories every delivered NPZ array and
CSV column. Check `SHA256SUMS` in this directory, or the parent manifest for the
entire code-and-data package.

The code remains under the parent MIT license. `metadata.json` deliberately
retains `data_license: null`. Existing DOI identifiers refer to earlier deposits;
the v1.2.0 additions have not been represented as already deposited or published.

The fixed comparison excludes the later reference-only records 42 and 44, for which
corresponding frozen-comparison predictions were not produced. At the final retained
capture, 20 other requested reference records had failed; their indices are in
`metadata.json`. Thus the fixed comparison originally lacked 22 requested labels,
of which two were later completed outside this selection. Neither the 686-atom
comparison nor complete tungsten trajectories/initial velocities are supplied.

## Integrity and limits

`SHA256SUMS` covers every delivered file except itself, using paths relative to this
directory. Verify before creating any local result files:

```sh
shasum -a 256 -c SHA256SUMS
```

The numeric extracts preserve the existing stored values; NPZ compression changes
container bytes only. Derived records are labelled as such. The package excludes
licensed software, pseudopotential files, executable programs, parameter weights,
logs and full electronic-structure outputs. Recalculation from the supplied arrays
is supported; rerunning the complete generating simulations requires additional
materials outside this deposit.
