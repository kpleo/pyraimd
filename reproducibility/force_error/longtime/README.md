# Successive-anchor residual work in thermal silicon

The base potential is unmodified MACE-MP-0b3-medium; references are unmodified
MACE-MPA-0-medium or Quantum ESPRESSO PBE (no D3), Si.pbe-n-kjpaw_psl.1.0.0,
50/400 Ry, unshifted 2x2x2 k grid, fixed occupations, non-spin-polarized,
nosym/noinv, conv_thr=1e-10 Ry. Si64 has masses 28.085 amu.

Each reference has two preparations, identified as configurations 1 and 2
(original replicas 0 and 1). Formal trajectories contain 500 or 80 one-fs
segments, after a separate 20-fs development prefix. The base integration
step is 0.05 fs and the full configuration probe displacement is 0.02 angstrom.
These are fixed-interval force updates, with continuous momenta and no thermostat.

## Contents

- Four NPZ files retain every formal segment's center, displaced-probe and
  endpoint positions, base/reference energies and forces, initial/final momenta,
  direction, original prediction, reference work, integration defect and
  prediction/motion/label times. Energies are eV; positions are angstrom;
  forces are eV/angstrom; time is fs; momenta use ASE internal units.
- `segments.csv`, `curves.csv`, `dynamics.csv`, `summary.csv`: all plotted
  formal data, full and formal cumulative curves, velocity rotation and metrics.
- `curves.json`: the same curves with the separate development/pilot prefixes
  and empirical sensitivity scales. Formal curves reset at global 20 fs.
- `size_norm_comparison.csv`: The force-norm comparison, normalized separately to the 64-atom
  value for each force norm; original size data remain under `../silicon/`.
- The root manifest records the distributed numerical inputs. The check script uses
  only this archive and NumPy; no models or external calculation are required.

Run `uv run --no-sync python -B check_longtime_source_data.py` in an existing
NumPy environment. It reconstructs force probes, frozen work predictions,
endpoint work, integration defects, both aggregate errors and all curve points.
The full and formal energy zeros, error signs and denominators are distinct.
`E_seg=sum(abs(predicted-measured))/sum(abs(measured))`;
`E_cum=max(abs(cumsum(predicted-measured)))/sum(abs(measured))`.
The empirical scale is a sum of monitored components, not a confidence interval
or a uniform error bound. The two thermal preparations are replicate units;
successive segments are correlated. All formal works are positive.

Original QE text outputs and full integration-step trajectories remain in the
research archive. This compact package reconstructs the formal predictions and
endpoint comparisons without redistributing software, model weights or PAW files.
