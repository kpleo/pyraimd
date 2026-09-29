# Formal update-interval series

Eight complete 80-fs paths use the same two Si64 initial configurations, with
force updates every 0.25, 0.5, 1 or 2 fs. The base is MACE-MP-0b3-medium and
the reference MACE-MPA-0-medium. Six branches are new; the two 1-fs branches
reuse the first 80 fs of the formal trajectories in `../longtime/`.
These 160 reused segments are not additional independent observations.
The comparison contains 1200 segments in total and requires no new DFT labels.

The integration step is 0.05 fs and probe displacement 0.02 angstrom in the
full normalized configuration direction. `initial_states.json` contains
geometry, momenta, cell, masses and model hashes. The additional branches
have displacement-scale and half-step diagnostics at 0, 32 and 64 fs; reused
baselines retain their initial and 35/65-fs diagnostics. Diagnostics never
change the saved predictions or propagation. Shadow endpoints have no separate
reference evaluation. Dense quadrature is unavailable on the new branches;
null values denote missing observations, not zero error.

The leading rate is `tau * mean(v^2*kappa) / 2`, using each branch's own evolving
anchors. Comparing `rate/tau` with `mean(v^2*kappa)` separates interval scaling
from trajectory feedback. Two initial configurations are the replicate units.
All signed work, weak-signal flags and empirical numerical scales are retained.
`U=sum(eta)` has no independent-noise reduction and is not a uniform error bound.

From the package root, run:

```sh
python -B check_interval_export.py --directory formal_interval
```

The checker reconstructs work from endpoint energies and force correction,
integration defects from kinetic/base energy changes, velocity from momenta,
curvature from the directional response, and all segment/aggregate metrics.
It checks initial-state equality, continuity, numerical scales and rate ratios.
