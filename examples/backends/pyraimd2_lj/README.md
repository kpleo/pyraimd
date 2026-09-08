# pyraimd2-lj — periodic Lennard-Jones backend plugin

A minimal third-party backend pair for the periodic CI recipe
(`examples/periodic_lj`): a Lennard-Jones reference engine and a slightly
softer LJ surrogate, both wrapping ASE's `LennardJones` calculator and
registered through the `pyraimd2.backends` entry-point group. No external
programs — NumPy + ASE only.

- `lj_reference`: LJ(epsilon, sigma, rc), full capabilities declared
  (energy, force-consistent, conservative, stress when periodic).
- `lj_surrogate`: same form with `epsilon * softening` (default 0.95), so
  adaptive runs see a small deterministic shadow error; no honest
  uncertainty spread (NaN).

Install: `uv pip install examples/backends/pyraimd2_lj`, then reference the
names in `[reference]` / `[surrogate]` sections as `lj_reference` /
`lj_surrogate`.
