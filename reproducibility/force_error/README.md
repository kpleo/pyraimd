# Standalone force-error analysis tools

Version **1.0.0** applies to these standalone analysis tools. The Pyramid
framework remains at **0.3.0**; installing Pyramid is not required here.
The three analysis scripts are distributed unchanged, with a small demonstration
runner and scalar reference results.

The controlled demonstration recomputes an analytical harmonic oscillator using
the Python standard library. The optional material demonstration reconstructs
directional response, force-error predictions and signed residual work from
supplied source-data arrays. **Source-data reconstruction does not run a new
ab-initio simulation.** These scripts do not generate material trajectories,
train or download models, or invoke electronic-structure software. New ab-initio
simulations require a separately configured engine, physical inputs and potential.

The existing [code archive](https://doi.org/10.5281/zenodo.22537316) provides the
software DOI. Material inputs are supplied separately as **Supplementary Data
1** (the `force_error_data` archive). This GitHub folder contains the analysis
software and demonstration checks.

## Installation and requirements

Use Python 3.10 or later. An ordinary CPU is sufficient; no GPU or special
hardware is required. Run the commands below from this directory.

The controlled demonstration needs no dependency installation:

```sh
python -S run_demo.py
```

For material-array analysis, install NumPy in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The requirement is
`numpy>=1.24,<3`; keep all three analysis scripts together. For the dependency
version used in validation, install `numpy==2.5.2` with Python 3.12.13.
Only Python 3.12.13 and NumPy 2.5.2 were checked for this demonstration.

Installation time is **estimated at 1–3 minutes** with Python already installed,
a compatible NumPy wheel and a typical broadband connection. This estimate was
not measured and excludes Python installation and source-data downloads.

One demonstration run on 7 September 2026 used **CPython 3.12.13, NumPy 2.5.2,
macOS 26.6.2, arm64, Apple M3 Max**. Measured elapsed time was **0.428 seconds**
in total: 0.256 seconds for the controlled calculation and scalar validation,
and 0.169 seconds for the optional material-array analysis and validation. These
are timings from one run in an existing environment, including subprocess
startup and temporary JSON I/O; they exclude installation and data download.
Runtime varies with CPU, storage and system load.

## Demonstration and expected results

`run_demo.py` invokes the unchanged scripts, compares selected scalars with
[expected_results.json](expected_results.json), and prints compact JSON with
`"status": "passed"`, the runtime platform and versions, and measured elapsed
seconds. Full intermediate JSON outputs are temporary and are removed when the
run finishes. A missing input or a failed numerical comparison exits nonzero.
The no-data mode does not import NumPy; its controlled subprocess uses `-S` to
disable site packages even when the material demonstration is requested.

The controlled solver uses exact harmonic flow between reference calls, fixed
parameters and seeds stored in `controlled_dynamics.py`, and 80 trajectories in
total. The following counts refer to the 16 held-out trajectories for each
policy (32,000 intervals per policy):

```text
class_horizon: n_reference=2353, n_unreferenced=29647, n_interval_violations=0
fixed_streak:  n_reference=2031, n_unreferenced=29969, n_interval_violations=1849
fixed_streak_k=18; trajectory_count=80
```

The fixed-streak baseline is selected using calibration reference cost before
its held-out force errors are measured. Violations count complete unreferenced
intervals, including interior extrema. These controlled-oscillator results do
not establish a guarantee for a material trajectory.

For the optional demonstration, download and extract the source data **outside
the code checkout**. Replace `PATH` below with either the extracted
`force_error_data` directory or its `interface` subdirectory:

```sh
python run_demo.py --data PATH
```

The data collection contains `interface/probes_0.npz`, `probes_1.npz`,
`motion_0.npz` through `motion_3.npz`, and `labels_0.npz` through `labels_3.npz`.
The demo uses only `probes_1.npz`, `motion_2.npz` and `labels_2.npz`. It reestimates
the response, predicts motion using direction 0, then analyzes labels with
`--end 1 --spacing 0.125`. Prediction and measurement geometry digests must match.
Expected scalar results are:

```text
directional_residual_work_coefficient_A2_eV = 9.137521529045872
force_accuracy_horizon_fs = 0.25
time_fs = 1.0
maximum_force_error_eV_A = 0.12230184406118094
endpoint_work_eV (W) = 0.052808734295215834
force_integral_eV (Q) = 0.0527638285524305
carbon atom 157 force_integral_eV = 0.0006441686530815945
```

Atom indices are zero-based. The carbon value is the contribution of the single
carbon atom at index 157 in the supplied atom order, not a sum over all carbon
atoms. The runner checks it whenever that atomic entry is available and reports
`"status": "unavailable"` otherwise. Without `--data`, material analysis is
reported as `"status": "not_requested"`.

Integer results must agree exactly. Floating-point results use an absolute
tolerance of `1e-12` in each quantity's stated units and zero relative tolerance.
The different values of W and Q are expected: W uses endpoint energies, whereas
Q is a trapezoidal force integral at 0.125 fs spacing. The 1 fs work endpoint
extends beyond the admitted 0.25 fs forecast horizon and is a diagnostic value.

## Use the individual scripts

To retain full outputs, choose new output filenames for each run; the individual
scripts refuse to overwrite existing files. Set `DATA` to your extracted
`interface` directory, then run:

```sh
python controlled_dynamics.py --output results/controlled.json
python directional_response.py estimate "$DATA/probes_1.npz" --output results/response.json
python directional_response.py predict "$DATA/motion_2.npz" --response results/response.json --direction 0 --budget 0.25 --floor 0.00025 --time-cap 1 --transverse-cap 0.1 --output results/predictions.json
python residual_work.py analyze "$DATA/labels_2.npz" --prediction results/predictions.json --end 1 --spacing 0.125 --output results/work.json
```

The force budget and numerical floor are in eV/angstrom, the time cap is in fs,
and the transverse-fraction cap is dimensionless. `--direction` is zero-based
within an origin's probe archive. The two origins pair with motions/labels 0–1
and 2–3, respectively; local direction indices are 0 and 1 for each pair.

Input archives are numerical NumPy arrays without pickled objects. Coordinates
are unwrapped and in angstrom; forces are in eV/angstrom; energies are in eV.
For N atoms, D directions at one origin and T times, the schemas are:

- **Probes:** `origin_positions_A`, `origin_base_forces_eV_A` and
  `origin_reference_forces_eV_A` have shape `(N,3)`; `unit_directions` has shape
  `(D,N,3)`; `probe_steps_A` has shape `(2,)` with increasing positive steps.
  `plus_displacements_A`, `minus_displacements_A`, `plus_base_forces_eV_A`,
  `minus_base_forces_eV_A`, `plus_reference_forces_eV_A` and
  `minus_reference_forces_eV_A` each have shape `(D,2,N,3)`.
- **Motion:** only `time_fs` of shape `(T,)` and `positions_A` of shape `(T,N,3)`.
  Include every integration state, starting at time zero with strictly
  increasing times. No future reference labels enter the prediction step.
- **Labels:** `time_fs` and `positions_A`, plus `base_forces_eV_A` and
  `reference_forces_eV_A` of shape `(T,N,3)`, and `base_energy_eV` and
  `reference_energy_eV` of shape `(T,)`. Optional `kinetic_energy_eV` has shape
  `(T,)`; optional integer `group_ids` has shape `(N,)`.

Normalize probe directions over all 3N components. Actual saved displacements
must match the signed steps within `1e-7` angstrom per component. Paired forces
must share coordinates, cell, atom order and reference settings. Use one fixed
base potential and origin correction per trajectory. Each requested quadrature
time must have a reference label; missing points are not interpolated. Reference
energies must be the thermodynamic potential whose derivatives yield the forces,
including when electronic smearing is used.

The sign convention is `R = F_base + c - F_ref`, with
`c = F_ref(0) - F_base(0)`. The endpoint work is
`W = delta(U_ref) - delta(U_base) + c dot delta(X)`; Q integrates R along the
supplied coordinates. Positive and negative atomic contributions retain their
signs. The finite-probe envelope is an empirical estimate whose accuracy needs
independent reference assessment.

`residual_work.py verify` additionally computes accepted-force violation bounds
from CSV columns `accepted,checked,violation`; use `--help` for its options.
Flags are 0 or 1; an unchecked violation may be empty, while a checked one must
be observed. Refused forces cannot be marked checked. Checks must be drawn
independently after the accepted force is fixed, with the check probability and
positive tilt chosen in advance. A CSV alone cannot establish that independence.

## License

These tools use the repository's [MIT License](LICENSE), copied without changes:
Copyright (c) 2026 Kang Peng.
