# Periodic Lennard-Jones CI recipe (no external software)

The L0 periodic example: a 32-atom fcc Lennard-Jones solid, exercising the
full Pyramid surface — singlepoint, fixed-model relax, plain and adaptive
NVE, checkpoint/resume and export — with only NumPy + ASE installed. It
exists so CI can check packaging and the periodic run mechanics; it is not a
materials result and does not stand in for the QE+MACE recipes.

Backends: the example plugin `examples/backends/pyraimd2_lj`
(`lj_reference` = LJ ε=1.0, `lj_surrogate` = 5% softer LJ, honestly
NaN uncertainty). Install once with
`uv pip install examples/backends/pyraimd2_lj`, or let the unit test inject
it through the entry-point group (see `tests/unit/test_periodic_lj.py`).

Files:
- `make_structure.py` — writes `structure.extxyz` (32-atom fcc, a = 1.58 A).
- `run-singlepoint.toml` — one surrogate evaluation.
- `run-relax.toml` — fixed-surrogate FIRE relaxation (fmax 0.05 eV/A).
- `run-md-reference.toml` / `run-md-surrogate.toml` — 8 plain NVE steps.
- `run-md-adaptive.toml` — 8 adaptive NVE steps, checks at p = 1.
- `README.md` — this note; commands below.

Commands (from this directory):

```sh
python make_structure.py
pyramid validate run-md-adaptive.toml
pyramid run run-singlepoint.toml
pyramid run run-relax.toml
pyramid run run-md-reference.toml
pyramid run run-md-surrogate.toml
pyramid run run-md-adaptive.toml
pyramid resume runs/lj-adaptive --steps 4
pyramid inspect runs/lj-adaptive
pyramid export runs/lj-adaptive --force-source driving
```
