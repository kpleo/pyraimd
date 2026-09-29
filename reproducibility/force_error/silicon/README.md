# Silicon numerical source data

This compact archive contains saved numerical results only. Its checker uses
NumPy and the Python standard library; it performs no model inference, DFT,
training, dynamics, or large-scale test. All arrays retain float64 precision.

## Contents and scope

- `size_series/Si{64,216,512,1000}_{ideal,displaced}/directions.npz`: all eight
  fixed geometry conditions, all 32 directions per condition (256 total), saved
  `u` and `H_delta_u`, coordinates, cell, atomic numbers, force correction, and
  per-direction coefficients. `statistics.json` retains the reported means,
  sample standard deviations, CVs, direction quantiles, conditional bootstrap
  intervals, covariance, summary-CSV values, budget ratios and frozen size
  transfer results. The bootstrap itself is not rerun by the quick checker.
- Size data use the frozen MACE pair: base MACE-MP-0b3 medium and reference
  MACE-MPA-0 medium. They are **not DFT data and not a DFT thermal ensemble**.
  Ideal diamond and controlled Gaussian-displaced configurations are fixed
  geometries, not equilibrium samples. The direction sample is a COM-free
  isotropic sphere; confidence intervals condition on geometry and model pair.
  The stored response uses h=0.04 angstrom. Four directions per condition have
  scalar half-h checks at 0.02 angstrom; their indices are stored explicitly.
  Raw displaced-force labels for the size responses are not present here;
  the saved response arrays support coefficient and participation recomputation.
- `thermal/Si64_anchor{1,2}/probes.npz`: two distinct preselected Si64 thermal
  preparations, five PBE/base E/F labels each (center, +h, -h, +h/2, -h/2),
  actual post-COM-removal velocity, normalized velocity direction u, masses,
  frozen correction c, response and coefficients. h=0.04 angstrom; the frozen
  prediction uses the small h=0.02 angstrom response. The preparations came from
  base-model MD; they are not an independently sampled PBE thermal ensemble.
- Each thermal directory has `budget_0p02_eV_A.npz` and
  `budget_0p04_eV_A.npz`: all four independent PBE endpoint labels, with base
  E/F, coordinates, actual velocity, frozen time and nominal prediction, plus
  same-time half-step base values. These are fixed-time corrected-base MD
  endpoints, not measured first threshold crossings. Two budgets on each path
  are paired. The four endpoint DFT labels were not used to refit C.
- `md_dt_0p05_fs.npz` and `md_dt_0p025_fs.npz` contain only the short saved
  numerical-control traces (19/37 and 21/41 frames), including positions,
  velocities, E/F, kinetic energy, corrected potential, energy defects and
  endpoint indices. No DFT labels exist for the half-step endpoints: half-step
  comparisons quantify integration differences, not DFT reference accuracy.
- `development/Si64_displaced/`: 3 original baseline labels plus 8 reference
  sensitivity labels. Four NPZ files hold center/plus/minus triplets, with the
  baseline center repeated in the half-displacement triplet (11 unique labels).
  Older labels without settings/cell fields are normalized using their saved
  physical input settings and common development cell. Each cutoff or k-grid
  comparison uses its own setting's center; h/2 work is scaled by four.
  Development sensitivity does not establish universal thermal convergence.

## Reference and units

The thermal reference is fixed S0: **PBE, no added D3, 50/400 Ry wavefunction/
charge-density cutoffs, 2x2x2 unshifted k-point grid, fixed occupations,
SCF threshold 1e-10 Ry**. The numerical sensitivity settings are 70/560 Ry
and 3x3x3 k points at the separate development geometry. The Si pseudopotential
is Si.pbe-n-kjpaw_psl.1.0.0.UPF; no software, model weights or pseudopotential
files are included. The base model for the PBE comparisons is MACE-MP-0b3 medium.

`numbers=14` denotes Si. `coords_A` and row-vector `cell_A` are in angstrom;
thermal coordinates are unwrapped. E is in eV, F and c in eV/angstrom,
velocity in angstrom/fs, time in fs, mass in atomic mass units. h is the
signed Euclidean norm along the normalized full 3N-vector u, not a per-atom
step. H_delta_u and Q are in eV/angstrom^2; C and chi in angstrom^2/eV;
participation numbers are dimensionless. `ase_time_unit_per_fs` records the
numeric conversion used for kinetic-energy reconstruction, without importing ASE.
Physical-setting flags in NPZ are numeric: functional_PBE=1, dispersion_D3=0,
occupations_fixed=1. Numeric JSON has no free-text provenance values.

## Reconstruction

Define r=F_DFT-F_base and c=r(center). At either displacement scale,
H_delta_u=[r(-h)-r(+h)]/(2h), Q=sum(u*H_delta_u),
S2=sum_i |H_delta_u_i|^2 and g2=max_i |H_delta_u_i|^2.
Then C=Q/g2, chi=Q/S2, n_infty=S2/g2,
n2=S2^2/sum_i |H_delta_u_i|^4, and C_rms=N*chi.
For size data, replace DFT by the MACE pair reference.
Mean C is the mean of directionwise chi*n_infty; covariance is retained.

For each endpoint with displacement dx from its anchor:

- W=(E_DFT-E_DFT_anchor)-(E_base-E_base_anchor)+sum(c*dx).
- epsilon_actual=max_i |F_DFT_i-F_base_i-c_i|.
- P0=C_frozen*epsilon_target^2/2; tau=epsilon_target/(sqrt(g2)*||v_actual||).
- Pe=C_frozen*epsilon_actual^2/2 is a retrospective residual conversion.
- Signed nominal error is 100*(P0-W)/|W|; signed converted error is
  100*(Pe-W)/|W|. Absolute percentage errors are their magnitudes.

All four outcomes are retained. The nominal absolute errors are below 1%
for these four points; the retrospective conversion errors are approximately
1.335%, 2.454%, 1.899%, and 3.644%. These are different quantities; the latter
must not be described as an a priori below-1% prediction. No universal or
long-time accuracy claim follows from these endpoints.

To check a standalone copy with an existing NumPy environment:

```sh
python check_silicon_source_data.py --data .
```

In an existing uv project, use `uv run --no-sync python` followed by the checker
location and `--data` followed by this directory. No dependency installation
is needed. The checker verifies manifest coverage/hashes, numeric-only data,
all direction coefficients and summary means, thermal finite differences,
all four E/F endpoint outcomes, both integration controls, and the development
  reference table. It only reads the archive and prints numerical results.

`theory/finite_n_summary.csv` retains the saved harmonic benchmark (eight model
families, 55 cases, 715 metric rows), including all seven three-dimensional
periodic-lattice comparison points; `finite_n_predictions.csv` retains 23 finite-N
comparison points. `finite_n_integrals.json` contains saved spectral and iid
chi-square_3 finite-N integral values, numerical quadrature changes and sign
fractions. These files support the finite-N benchmark and chi-square integral
comparisons; they are a conditional harmonic/Gaussian-ray benchmark, not silicon
DFT evidence. The checker verifies their cross-file consistency only; it reruns
neither Monte Carlo nor quadrature. Heavy sample arrays are omitted.
Units use k=1 and equal masses; atom RMS means sqrt(sum_i|r_i|^2/N).
Matching numeric pairing_group values identify paired draws; embedded local
models reuse their active draws across total sizes. Saved confidence intervals
use normal/delta approximations for means/SD/CV, conservative binomial ranks for
quantiles and Wilson intervals for sign fractions. Integral values are numerical
evaluations of finite-N formulas, not asymptotic replacements.
Empty theory CSV cells preserve unavailable estimates; they do not denote zero.

`reported_*` endpoint fields and JSON statistics are saved comparison values,
not substitutes for recomputation from labels. The manifest lists every file
under this silicon directory except itself. Its scope does not extend to any
parent source-data collection or combined ZIP.
