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

## Install

Python 3.12 or later. The core install needs only NumPy and ASE — no torch, no
PySCF, no external programs. From a checkout:

```sh
pip install .            # user install
pip install -e '.[dev]'  # development: adds pytest and ruff
```

Optional backends install separately and are only imported when selected:

```sh
pip install '.[pyscf]'   # molecular reference engine
pip install '.[mace]'    # MACE surrogate potentials
```

`all` installs both; `builders` adds optional structure-building tools. Quantum
ESPRESSO additionally needs a `pw.x` executable and pseudopotentials installed
outside Python.

The Python import is `pyraimd2`; the CLI is `pyramid` (alias `pyraimd2`).
`pyramid --version` and `pyramid --help` work offline and download nothing.

## Pick your task

### Run a first adaptive MD in five minutes

`examples/harmonic_adaptive/` is a complete, offline adaptive NVE run: builtin
analytic reference and surrogate models drive a toy H2O molecule. It exercises
anchoring, surrogate acceptance, independent checks, checkpoints, stop and
resume — no external programs, no model downloads, finishes in seconds:

```sh
cd examples/harmonic_adaptive
pyramid validate run.toml
pyramid run run.toml
pyramid inspect runs/harmonic-demo
pyramid resume runs/harmonic-demo --steps 10
pyramid export runs/harmonic-demo --force-source reference --output reference.extxyz
```

`pyramid init --template harmonic --output my_run` writes the same pair of
files as a starting point for your own configuration. The template's policy and
verification numbers are demonstration values, not accuracy recommendations for
any material. The full field reference is in
[docs/configuration.md](docs/configuration.md).

### A serial relax → NVT → NVE workflow, offline

`examples/periodic_lj/` runs a 32-atom fcc Lennard-Jones cell through the three
stages with only NumPy + ASE plus the example plugin. The NVT stage is adaptive
(fixed model) and the NVE stage continues with the NVT boundary's complete
momenta — never rethermalized:

```sh
pip install .                       # the core package
pip install examples/backends/pyraimd2_lj   # the example backend plugin
cd examples/periodic_lj
python run_recipe.py --output results/lj-recipe   # relax + NVT + NVE
python run_recipe.py --output results/lj-recipe   # continue/resume, idempotent
pyramid inspect results/lj-recipe/nvt
pyramid export results/lj-recipe/nve --force-source driving
```

For guarded online model updates under NVT (Python interface),
`examples/guarded_nvt.py` runs offline on analytic potentials, with stop and
new-process resume.

### Periodic materials (Quantum ESPRESSO + MACE)

`examples/qe_mace_skeleton/` shows the configuration shape for a real-material
adaptive run: `pw.x` as the reference engine and a MACE foundation model as the
surrogate over a small periodic silicon cell. It is a skeleton, not a verified
materials recipe. For complete structure generators and task configurations,
see the [bulk Si](examples/si_bulk_qe_mace/README.md) and
[Al(111) slab](examples/al_surface_qe_mace/README.md) examples. They include
input requirements, short NVE runs, resume and export commands.

Before `pyramid run` works you need, on that machine:

- `pw.x` on PATH (or `reference.pw_cmd`) and the pseudopotential files the
  configuration names;
- `pip install '.[mace]'` and a MACE model (a local file, or a foundation-model
  name that MACE itself downloads on first use).

Without them, `pyramid validate run.toml` still parses the configuration and
reports exactly which pieces are missing. With them,
`pyramid validate run.toml --probe-backends` evaluates the structure once with
each backend before committing to a run.

### Stop, resume and recover

- `Ctrl-C` during `pyramid run` stops at the next complete-step boundary and
  writes a checkpoint; nothing is half-written.
- `pyramid resume RUN_DIR --steps N` always adds N steps to wherever the run
  actually is, and prints the current and target step numbers first. Settings
  come from the run's `resolved_config.json` — same physics, model chain and
  check stream. This works for adaptive runs and for plain
  reference/surrogate-only runs.
- Resume replays the event log from the last valid checkpoint generation;
  truncated or tampered checkpoints are skipped, and reference/model identity
  mismatches are refused rather than silently re-anchored.
- If a crashed process left the event-log writer lock behind,
  `pyramid resume RUN_DIR --steps N --force-unlock` reclaims it (only when no
  live writer exists).
- Changing settings means a new run, or a library-level `fork` that inherits
  the physical state and model chain under a new check stream.

### Plug in your own ASE calculator

Two routes, both without modifying the core package:

1. **Python API** — the generic adapters wrap separately configured ASE
   calculator instances:

   ```python
   from pyraimd2.engines import AseEngine
   from pyraimd2.surrogate import AseSurrogate

   engine = AseEngine(reference_calculator, force_consistent=True)
   surrogate = AseSurrogate(potential_calculator)
   ```

2. **Plugin package** — register factories under the `pyraimd2.backends`
   entry-point group and the backend name becomes usable from TOML
   configurations. `examples/backends/pyraimd2_harmonic/` is a minimal,
   installable example (two analytic backends, ~one file of code).
   `pyramid backends` lists everything the registry can see.

Capability declarations (energy kind, force consistency, stress, uncertainty)
are checked when a backend is created — before any expensive calculation.
See [docs/architecture.md](docs/architecture.md) for the protocols.

### Develop a new method

- [docs/architecture.md](docs/architecture.md) — the workflow layers and the
  `Engine`/`Surrogate` protocols, capabilities and identity fingerprints.
- [docs/api.md](docs/api.md) — public entry points of the `engines`,
  `surrogate`, `runtime`, `workflows` and `backends` modules.
- `pyraimd2.energetics` is a pure numerical layer (directional response,
  forecasts, signed work, independent checks) usable outside MD.
- `pyraimd2.loop.online` (`GuardedUpdater`, `UpdatePolicy`) is the guarded
  candidate → validate → publish/rollback protocol for stateful model updates;
  `pyraimd2.runtime` provides the event log, checkpoints and the immutable
  model registry a new updater must plug into to be resumable.

## Current scope

- **Tasks.** Single-point evaluation and fixed-model FIRE/BFGS relaxation
  use either a reference engine or a surrogate. MD supports reference-only,
  surrogate-only and adaptive modes, with checkpoints, resume and export.
  Adaptive mode applies only to MD.
- **Dynamics.** The released version (0.4.2) supports fixed-cell NVE and
  `FixAtoms`. Version 0.5.0 adds fixed-cell NVT (ASE Langevin,
  `fixcm=False`): plain reference/surrogate modes, and adaptive MD in both
  ensembles — with a fixed base model via TOML/CLI, or with guarded online
  updates through the Python `GuardedUpdater` interface
  (`EnergeticRunner(on_label=...)`; re-anchoring supported, each segment
  recorded independently). Online updates are not expressible in
  configuration files: no TOML key installs an update callback.
  Variable-cell dynamics and other constraint types are unsupported.
  The default force budget measures the free coordinates
  (`active_dofs_max_atom`); `all_atoms_max_atom` is an explicit alternative.
  Checkpoints retain two generations.
- **Serial recipes.** `relax → NVT → NVE` chains run through
  `run_serial_recipe` with per-stage run identities and deduped
  stop/continue (`examples/periodic_lj` offline; `examples/si_bulk_qe_mace`
  for QE + MACE). Stage handoff carries the authoritative completed state:
  positions, cell/PBC, masses, `FixAtoms`, initial charges/magmoms, and
  complete momenta — initialized once at the first MD stage, preserved
  afterwards, never rethermalized.
- **Backends.** QE, molecular closed-shell PySCF, MACE and analytic harmonic
  backends are provided. Generic ASE adapters, the ASE-native QE path and
  backend plugins provide extension interfaces; compatibility declarations
  alone do not establish a backend's numerical accuracy for a material.
  Training capability is per backend: the analytic harmonic example
  (`examples/guarded_nvt.py`) demonstrates guarded online updates on a toy
  model; MACE is used as a fixed model in the examples (its online
  fine-tuning interface is not an accepted feature of this release).
- **Energy consistency.** Each side's forces must be the negative gradient
  of its reported scalar for anchored energies and endpoint work to be meaningful.
  A smeared QE `free_energy` reference can be paired with a MACE `energy`
  surrogate when both declare force-energy consistency and conservative
  forces. Cross-kind combinations with an undeclared requirement are
  rejected; `unknown` is never evidence of consistency. Reference identity
  includes smearing type/width, XC and pseudopotential content.
- **Examples and costs.** The Si and Al examples demonstrate software
  workflows. Their short trajectories and illustrative budgets do not
  establish thermodynamic convergence or speed-ups. Use `pyramid inspect`
  to count anchors, probes, checks, retries and cache hits; a surrogate
  acceptance rate is not a reference-calculation saving.

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

## Documentation and development

- [Configuration and CLI reference](docs/configuration.md)
- [Architecture and backend integration](docs/architecture.md)
- [API reference](docs/api.md)
- [Energetic force errors and runtime decisions](docs/energetic_force_error.md)
- [Changelog](CHANGELOG.md)
- [Supplementary Materials: force-error data and code](reproducibility/force_error/README.md)
- `examples/` — the demos above, plus direct Python-API scripts:
  `energetic_loop.py` runs offline on analytic backends, `guarded_nvt.py`
  demonstrates guarded online updates under NVT (fresh run, stop, resume),
  and `periodic_lj/run_recipe.py` drives the serial relax → NVT → NVE
  recipe; the others need the optional extras they name.
- [Slurm example](examples/slurm/README.md) — a generic submission template
  to adapt to your compute environment.
- `src/pyraimd2/` — framework implementation; `tests/` — numerical, interface
  and integration checks.

Run the offline core suite (NumPy + ASE only, no optional backends needed):

```sh
pytest tests/unit tests/test_smoke.py -q
```

Backend integration tests marked `slow` require the corresponding optional
software and model files; they are deselected by default and never count as
passing when their dependency is missing. Configure production executables,
model locations and compute resources in your own run setup.

## Citing

If you use Pyramid in your research, please cite the underlying method paper:

- *The energetics of force errors in machine-learned molecular dynamics*,
  arXiv:2609.09251 (2026). https://arxiv.org/abs/2609.09251

## License

Pyramid is released under the [MIT License](LICENSE).
