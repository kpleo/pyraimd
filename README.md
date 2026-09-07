# Pyramid

**Python wrapped Ab initio Molecular Dynamics** — a name drawn from the
PYR-AIMD letters.

Pyramid brings first-principles engines, machine-learned interatomic potentials
(MLIPs), solvers and molecular dynamics into one extensible Python framework.
Reference calculations supply energies and forces; fast potentials propagate
atomic motion; the framework coordinates when to evaluate, check and update them.

Its energetic force-error method connects force accuracy to the motion of the
atoms. A local directional response predicts how force errors grow away from a
reference configuration. Signed residual work measures their energetic effect
along the trajectory. Together, these quantities guide reference calculations
and provide a physical account of the forces used during a run.

## Start here

Python 3.12 or later is required. From a checkout, preferably in a virtual
environment:

```sh
pip install -e '.[dev]'
python examples/energetic_loop.py
```

The harmonic example runs with NumPy and ASE, without model downloads or an
electronic-structure installation. It demonstrates the energetic MD loop with
an analytical reference and an approximate potential.

Optional backends can be installed separately:

```sh
pip install -e '.[pyscf]'
python examples/energetic_pyscf.py

pip install -e '.[mace]'
```

The PySCF example adds molecular reference calculations. MACE models are loaded
when selected; named pretrained models may require a download. The `all` extra
installs both backend dependencies, while `builders` adds optional structure
building tools. Quantum ESPRESSO requires a separately installed executable and
suitable pseudopotentials.

The Python import remains `pyraimd2`. Both `pyramid` and `pyraimd2` print the
installed version and the entry point for the examples.

## The energetic MD loop

1. **Anchor.** Evaluate the reference and fast potential at the same atomic
   configuration. A constant force correction matches the reference there.
2. **Forecast.** Probe the local force response along a configuration-space
   direction. Use the observed displacement to predict the residual force and
   signed work before requesting a reference at the new geometry.
3. **Propagate or request a reference.** Use the corrected fast force while the
   forecast meets the chosen force budget and domain settings. Otherwise, obtain
   a new reference label.
4. **Measure the work.** At a labeled endpoint, compare reference and anchored
   potential energies to evaluate signed residual work over the preceding
   segment. Independently sampled checks also measure errors of accepted forces.
5. **Update and reanchor.** An optional `on_label` callback can use new labels to
   adapt the potential. A model change or a detected violation requires a fresh
   anchor and response before accepting further fast forces.

Directional probes supply an empirical forecast. Independent accepted-force
checks provide a separate statistical assessment of violations under their
sampling protocol. The [method guide](docs/energetic_force_error.md) explains
these quantities, their units and the conditions for interpreting them.

## Backends and extension points

- **Reference engines:** PySCF closed-shell molecular RKS and Quantum ESPRESSO
  fixed-cell PBE+D3 calculations; `AseEngine` wraps other configured ASE
  calculators that provide compatible energies and forces.
- **Fast potentials:** a MACE potential, a MACE committee with optional
  fine-tuning, or another ASE calculator through `AseSurrogate`.
- **Dynamics:** `EnergeticCalculator` connects the energetic policy to ASE;
  `EnergeticRunner` supplies fixed-cell Velocity Verlet integration.
- **Numerical tools:** `pyraimd2.energetics` contains the directional estimator,
  force forecast, signed-work calculations and independent-check accounting.
- **Alternative routing:** existing threshold, scheduled and conformal policies
  remain available through the original switching interface.

The [architecture guide](docs/architecture.md) describes the backend protocols
and how to add a solver or potential. Additional first-principles solvers and
broader dynamical settings are future extension directions.

## Documentation and development

- [Architecture and backend integration](docs/architecture.md)
- [Energetic force errors and runtime decisions](docs/energetic_force_error.md)
- [Force-error reproduction data, code and demonstrations](reproducibility/force_error/README.md)
- [Changes in version 0.3.0](CHANGELOG.md)
- `examples/energetic_*.py` — starting points for the energetic method.
- Other examples and `experiments/` retain scheduled and conformal workflows.
- `src/pyraimd2/` — framework implementation.
- `tests/` — numerical, interface and integration checks.

After installing the `dev` extra, run the unit suite with:

```sh
pytest tests/unit -q
```

Version 0.3.0 was checked on Linux CPUs with 170 software tests, the harmonic
example, and short H2 trajectories using PySCF with both an analytic potential
and a local MACE model. These checks exercise the software interfaces, routing
and work records.

Backend integration tests require the corresponding optional software and model
files. Configure production executables, model locations and compute resources
in your own run setup.

## License

Pyramid is released under the [MIT License](LICENSE).
