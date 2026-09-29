# Force-error energetics — numerical reproducibility package

Version **2.0.0** reconstructs force-error work, directional response,
atomic participation, size scaling and successive force-matching predictions
from saved numerical records. It runs with Python 3.10+ and NumPy, without
Pyramid, model inference, DFT, training or trajectory generation.

```sh
python -B check_all.py
```

The checker verifies the complete checksum inventory, reconstructs physical
quantities from arrays, and compares them with the distributed tables. It prints
a JSON report and leaves the package unchanged. All scripts use the package
location, so the command also works from an unrelated working directory.

## Data map

| Directory | Numerical content |
|---|---|
| `interface/` | Four original interface paths, force probes, energies, forces, atomic work integrals and electronic/integration sensitivity records |
| `material_endpoints/` | Four equal-force-budget DFT endpoints |
| `member_ensemble/` | Two interface anchors, 128 directions and 256 paired budget endpoints |
| `derivative_labels/` | Sixteen development directions, each at two displacement scales |
| `atomic_work/` | All 474 signed atomic contributions at each of two interface endpoints |
| `silicon/` | Eight fixed geometries, 256 responses, two thermal configurations, four PBE endpoints, integration controls and finite-size benchmarks |
| `longtime/` | Every segment of two 500-fs MACE-reference and two 80-fs PBE-reference trajectories: 1160 segments |
| `formal_interval/` | Eight 80-fs MACE-reference paths at 0.25, 0.5, 1 and 2 fs: 1200 segments, including 160 reused 1-fs baseline segments |
| `tables/`, `data/` | Tabulated numerical summaries and current derived comparisons, including free-predictor baselines |
| `historical/` | Retained electronic-setting sensitivity summary; its original geometry is unidentified |

`schema.json` records each NPZ array's shape and dtype and each CSV's columns
and row count. `MANIFEST.sha256` covers all package files except itself.
The silicon subdirectory additionally retains its own standalone manifest and
checker. The readmes in `silicon/`, `longtime/` and `formal_interval/` specify
settings and numerical scope. No rendered figures, plotting layouts, manuscript
files, model weights, software binaries or pseudopotentials are distributed.

## Definitions and conventions

For a segment starting at anchor `a`, the force correction is
`c = F_reference(a) - F_base(a)` and
`W = delta(E_reference) - delta(E_base) + c dot (X_end - X_a)`.
The force residual is `F_base + c - F_reference`.
Central probes give `H_delta u = [r(-h)-r(+h)]/(2h)`, where
`r = F_reference - F_base`. The directional curvature is
`kappa = u dot H_delta u`; the prospective segment prediction is
`W_pred = v^2 kappa tau^2 / 2`.

Energies are eV unless a field explicitly says meV, distances angstrom, time fs,
forces eV/angstrom, and drift rates meV/atom/ps. Momenta use ASE internal units;
the numerical conversion is retained in the arrays. Atomic indices are zero
based; displayed configuration numbers are one based.

`E_seg = sum(abs(W_pred-W)) / sum(abs(W))` and
`E_cum = max(abs(cumsum(W_pred-W))) / sum(abs(W))`.
CSV percentage columns multiply these ratios by 100. Aggregate errors and
maximum individual-segment errors are distinct. The empirical numerical scale
is `U=sum(eta)`, with no square-root reduction; it is not a confidence interval.
Both initial configurations, all work signs and all segments are retained.
Successive segments are correlated observations.

The baseline comparison excludes the first segment of each path for every
predictor. `persistence_v2` means previous-segment work multiplied by the ratio
of current to previous squared speed; `constant_rate` uses the running mean
of previous segment work. `compute_baselines.py INPUT.csv OUTPUT.csv` rebuilds
the 16-row comparison from `data/formal_strips.csv`.

## Version and scope

The canonical version is
[force-error-repro-v2.0.0](https://github.com/kpleo/pyraimd/tree/force-error-repro-v2.0.0/reproducibility/force_error).
It consolidates the current interface and silicon records and replaces the
earlier demonstration collection at this directory on the default branch.
The earlier collection remains available at its immutable
[v1.2.0 tag](https://github.com/kpleo/pyraimd/tree/force-error-repro-v1.2.0/reproducibility/force_error).
This data-package version is independent of the Pyramid software version.
No new DOI is assigned by this GitHub update.

The scripts recompute results from retained labels and response arrays. They
do not recreate electronic-structure calculations or the fitted potentials.
Detailed coverage and unavailable raw records are stated in the subdirectory
readmes. The existing MIT software license is retained in `LICENSE`; this
release does not assign a new dataset license.
