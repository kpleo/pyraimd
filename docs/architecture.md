# Architecture

Pyramid separates the calculation of energies and forces from decisions about
their use in molecular dynamics. A reference engine, a fast potential and a
runtime policy can be replaced independently. ASE supplies the atomic structure
and integration interfaces.

## Reference engines and fast potentials

An `Engine` implements `compute(atoms) -> EngineResult`. The result contains an
energy, an `(N, 3)` force array, optional stress and the evaluation wall time.
An engine raises `EngineError` if it cannot produce a valid label.

A `Surrogate` implements `predict(atoms) -> SurrogatePrediction`. Its result
contains energy, forces, optional stress and an `(N,)` uncertainty array. A
single potential can leave uncertainty unavailable; the energetic method obtains
its directional response from paired reference and surrogate forces. A trainable
potential can additionally implement `finetune(labels) -> TrainReport`.

Current reference adapters are:

- `PyscfEngine`: finite, closed-shell molecules using restricted Kohn–Sham DFT.
  The exchange-correlation functional, basis and convergence tolerance are
  configurable.
- `QeEngine`: periodic, fixed-cell PBE+D3 single-point calculations through
  Quantum ESPRESSO `pw.x`. `QeConfig` specifies pseudopotentials, cutoffs,
  k points, electronic settings and the execution command. Its electronic
  input settings use Quantum ESPRESSO's native units; returned labels use ASE
  units.
- `AseEngine`: an externally configured ASE calculator providing energy and
  forces.

For the fast-potential role, `MaceSurrogate` wraps a MACE model,
`CommitteeSurrogate` combines MACE members and exposes fine-tuning, and
`AseSurrogate` accepts another ASE calculator.

The generic adapters have matching construction options:

```python
from pyraimd2.engines import AseEngine
from pyraimd2.surrogate import AseSurrogate

# Supply separately configured ASE calculator instances.
engine = AseEngine(reference_calculator, force_consistent=True)
surrogate = AseSurrogate(potential_calculator, force_consistent=True)
```

Here `reference_calculator` and `potential_calculator` are the calculators chosen
by the application. Set `force_consistent=True` when their forces differentiate
a free energy rather than the default reported energy. Both adapters default to
`force_consistent=False` and `include_stress=False`; stress can be requested when
the backend supplies it. Give each run its own calculator instances and working
directories.

## Energetic prediction and dynamics

`pyraimd2.energetics` holds the array-based calculations: force residuals, local
directional response, empirical forecasts, signed residual work and independent
accepted-force checks. These calculations can be used separately from MD to
analyze paired energies and forces.

`EnergeticCalculator` in `pyraimd2.loop` owns the current reference anchor and
uses the forecast to choose between an anchored fast force and a new reference.
New reference labels provide an endpoint work measurement and support the next
anchor. Its optional `on_label` hook gives the application a place to collect
labels or adapt the fast potential. A model change requires a fresh anchor and
directional response because it changes the residual being predicted.

`EnergeticRunner` connects this calculator to ASE Velocity Verlet integration.
Given an `atoms` configuration, a surrogate and a reference engine:

```python
from pyraimd2.loop import EnergeticRunner
from pyraimd2.store import Store

runner = EnergeticRunner(
    atoms, surrogate, engine, Store("trajectory.db"), run_id="example",
    force_budget=0.1, timestep_fs=0.5,
)
summary = runner.run(20)
```

Choose the budget and timestep for the physical problem. Each new run needs a
fresh `run_id`. Calling `run` again continues the same live instance. Restoring
an energetic run from disk is not yet supported; the legacy restart interface
does not restore its calibration and check state. Existing
momenta are preserved; otherwise `temperature_K` and `velocity_seed` initialize
them. The default probe direction follows velocity. A `direction(atoms)`
callback can return one `(N, 3)` vector or several `(D, N, 3)` vectors, which the
runtime normalizes over the full configuration.

Start with
[`examples/energetic_loop.py`](../examples/energetic_loop.py) for a complete
analytical example, or
[`examples/energetic_pyscf.py`](../examples/energetic_pyscf.py) for a molecular
reference. The examples show the concrete runtime configuration and how to read
its results.

Independent checks are taken after the force has been accepted. The comparison
therefore refers to the force selected for that evaluation; a new label can
inform later forces and subsequent anchors. This ordering keeps checking
distinct from reference-based correction of the force being checked.

The `on_label(observation)` callback runs after the force is selected and its
record is stored. The observation carries the uncorrected base prediction and
the reference label at that geometry; directional probes do not invoke it.
Return exactly `False` when the model is unchanged, for example when only
collecting labels. Any other return requests recalibration before the next
evaluation. Apply model updates through this hook so each segment retains a
well-defined base potential.

For a trainable committee, the existing updater can collect labels and fit at
a chosen interval. Report whether a fit occurred so unchanged models retain
their current response:

```python
from pyraimd2.loop import OnlineUpdater

updater = OnlineUpdater(committee, observe=lambda spread, error: None, n_label=8)

def adapt(observation):
    return updater(observation) is not None

# Pass on_label=adapt when constructing EnergeticRunner.
```

## Run records

`Store` writes ASE database rows containing the configuration, route and reason,
base prediction, available reference label and the energy and force actually
used. Energetic metadata adds the forecast, anchor and probe records, measured
endpoint work, independent-check outcome and cumulative check bound. This keeps
the propagated force distinct from a later check of it.

`EnergeticRunSummary` reports accepted evaluations and successful reference-call
counts split into anchors, probes and checks, together with detected violations
and verification state. Probe calls count toward reference cost even though they
are off the trajectory.

## Units and configurations

The backend boundary uses ASE units:

- Positions and directional displacement scales: angstrom.
- Total energies and signed residual work: eV.
- Forces and force-error budgets: eV/angstrom.
- Optional stress: eV/angstrom cubed, in ASE Voigt order
  `(xx, yy, zz, yz, xz, xy)`.
- MD timestep and elapsed-time settings: femtoseconds at the runner interface.

Reference and surrogate results must describe the same coordinates, atom order,
cell and boundary conditions. Use continuous, unwrapped coordinates when
accumulating displacement and work across periodic boundaries. Keep the
reference settings and the surrogate fixed within each anchored segment.

The energetic loop currently treats unconstrained, fixed-cell motion with fixed
atom identities and masses. An engine's ability to return stress is a backend
capability; variable-cell dynamics also requires a consistent treatment of cell
work in the runtime policy.

## Adding a backend or solver

If a solver already has an ASE calculator, configure it in the application and
wrap it with `AseEngine` or `AseSurrogate`. Otherwise, implement the corresponding
protocol, convert units at that boundary and return energies consistent with the
forces. The runtime policy then operates on the common result types.

A solver can be external software or a Python implementation. Future solvers
developed within Pyramid can use the same interfaces. Electronic-structure
methods, model parameters, pseudopotentials and parallel execution settings
remain explicit choices of the calculation.

## Existing switching workflows

The original `Runner`, `SwitchingCalculator`, routing policies in
`pyraimd2.switch`, `OnlineUpdater` and trajectory `Store` remain available for
existing workflows. Their policy-specific calibration and update interfaces
are separate from the energetic runner. Use the energetic examples when
starting a run based on directional response and residual work.
