# Force-error analysis

Small, independent scripts for directional force-error predictions, signed
residual work, force-accuracy horizons and independent reference checks.

The controlled calculation runs without input files. Atomistic calculations use
externally supplied arrays. No simulation data, trained models or electronic-
structure software are included.

## Installation

Use Python 3.10 or later. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The controlled calculation uses only the Python standard library. The other
scripts require NumPy. Keep the three Python files together.

## 1. Run the controlled calculation

```sh
python controlled_dynamics.py --output results/controlled.json
```

The reference force is `-k*x`; the approximate force is `-x+b`. Each interval
uses the exact harmonic flow. Reference measurements update the constant bias.
The calculation includes calibration, subsequent independent initial conditions,
and conditions outside the supplied stiffness class. Reference cost selects the
fixed-streak comparison before its force errors are measured. The output includes
force-budget violations and signed work split into the initial residual and
curvature contributions at each reference anchor.

All simulation parameters and random seeds are specified in the script. Repeating
the command recomputes the results; it does not load stored answers.

## 2. Supply local directional probes

Create a NumPy archive `inputs/probes.npz` with the following arrays. `N` is the
number of atoms; `D` is the number of probed directions at a **single** origin.
The scale dimension has length two, ordered from smaller to larger displacement.

| Array | Shape | Meaning |
|---|---|---|
| `origin_positions_A` | `(N,3)` | Reference-origin coordinates |
| `origin_base_forces_eV_A` | `(N,3)` | Base-potential forces at the origin |
| `origin_reference_forces_eV_A` | `(N,3)` | Reference forces at the same origin |
| `unit_directions` | `(D,N,3)` | Unit vectors in the full configuration space |
| `probe_steps_A` | `(2,)` | Two positive central-displacement scales |
| `plus_displacements_A` | `(D,2,N,3)` | Actual displacements at the positive probes |
| `minus_displacements_A` | `(D,2,N,3)` | Actual displacements at the negative probes |
| `plus_base_forces_eV_A` | `(D,2,N,3)` | Base forces at positive probes |
| `minus_base_forces_eV_A` | `(D,2,N,3)` | Base forces at negative probes |
| `plus_reference_forces_eV_A` | `(D,2,N,3)` | Reference forces at positive probes |
| `minus_reference_forces_eV_A` | `(D,2,N,3)` | Reference forces at negative probes |

Arrays can be saved with `numpy.savez(filename, **arrays)`, where `arrays` maps
these names to numerical arrays. Do not use object arrays. All force pairs must
refer to identical coordinates, cell, atom order and electronic-reference settings.
Use the actual saved displacements, including coordinate-rounding effects. They
must correspond to `+h*u` and `-h*u` within `1e-7` angstrom per component.
Normalize each direction over **all 3N
components**, rather than separately for each atom. Prepare a separate archive
for each reference origin, including all directions that define its shared
transverse estimate.

```sh
python directional_response.py estimate inputs/probes.npz --output results/response.json
```

For residual `R = F_base + c - F_ref`, with `c = F_ref(0) - F_base(0)`, the central
response is `q_h = (R(+h*u) - R(-h*u))/(2*h)`. The larger scale supplies `q`.
The directional residual-work coefficient is estimated by

```text
C_rw = (u · q) / max_i(|q_i|)^2
```

The units of `C_rw` are squared length per energy. The output also separates the
response participation and signed inverse curvature. It computes empirical
envelope coefficients from the two scales: twice the response difference for
parallel sensitivity, twice the largest response norm across all supplied
directions and scales for transverse response, and four times the largest
probe-vector defect divided by squared displacement for the remainder term.

## 3. Predict before evaluating future reference forces

Generate a trajectory under the fixed base potential with the fixed constant
correction `c`. Save `inputs/motion.npz` containing **only**:

| Array | Shape | Meaning |
|---|---|---|
| `time_fs` | `(T,)` | Times starting at zero, strictly increasing |
| `positions_A` | `(T,N,3)` | Unwrapped coordinates at every integration state |

```sh
python directional_response.py predict inputs/motion.npz --response results/response.json --direction 0 --budget 0.25 --floor 0.00025 --time-cap 1 --transverse-cap 0.1 --output results/predictions.json
```

Here `--direction` uses zero-based indexing. For displacement `d`, the code uses
`alpha = u · d`, `z = d - alpha*u`, and

```text
linear force error = |alpha| max_i(|q_i|)
predicted work     = alpha^2 (u · q) / 2
envelope           = linear error + floor + eta|alpha| + K|z| + M|d|^2/2
```

The first state outside the force budget, time cap or transverse-fraction cap
ends the admitted prefix. Include every integration state so intermediate
failures are counted. The horizon refers to this discrete forecast grid.
The coefficients inferred from finite probes are empirical estimates; their
accuracy is evaluated against subsequent independent reference calculations.
Save the prediction file before obtaining those reference results. Predictions
contain geometry digests to check subsequent matching.

## 4. Recalculate residual work from paired forces and energies

Create `inputs/labels.npz` with the arrays below. Include the origin and every
reference-evaluated state to be used. Each state must retain the same atom order
and the exact coordinates used for prediction.

| Array | Shape | Meaning |
|---|---|---|
| `time_fs` | `(T,)` | Reference-evaluation times, starting at zero |
| `positions_A` | `(T,N,3)` | Unwrapped coordinates, in angstrom |
| `base_forces_eV_A` | `(T,N,3)` | Base-potential forces |
| `reference_forces_eV_A` | `(T,N,3)` | Reference forces at the same coordinates |
| `base_energy_eV` | `(T,)` | Base-potential energies |
| `reference_energy_eV` | `(T,)` | Reference energies consistent with reference forces |
| `kinetic_energy_eV` | `(T,)` | Optional total kinetic energies |
| `group_ids` | `(N,)` | Optional fixed integer group for each atom |

```sh
python residual_work.py analyze inputs/labels.npz --prediction results/predictions.json --output results/work.json
```

The fixed origin correction is reconstructed from the force pair. The code
evaluates maximum-atom force errors and signed endpoint work,

```text
W = [U_ref(X)-U_ref(0)] - [U_base(X)-U_base(0)] + c · [X-X(0)]
Q = sum_j (R_j + R_(j+1))/2 · [X_(j+1)-X_j]
```

`Q` is a trapezoidal force integral along the supplied coordinates. Atomic and
group contributions retain their signs and sum to the same integral. When kinetic
energies are supplied, the output separates reference-Hamiltonian change from
the anchored-Hamiltonian integration drift. Use one fixed reference, base
potential and correction per input. At finite electronic smearing, the reference
energy must be the thermodynamic potential whose derivatives give those forces.

To compare quadrature resolutions at the same endpoint:

```sh
python residual_work.py analyze inputs/labels.npz --end 1 --spacing 0.5 --output results/work_coarse.json
python residual_work.py analyze inputs/labels.npz --end 1 --spacing 0.25 --output results/work_medium.json
python residual_work.py analyze inputs/labels.npz --end 1 --spacing 0.125 --output results/work_fine.json
```

Every requested grid point must have a reference label; missing points are not
interpolated. Without `--spacing`, the integral uses the supplied, possibly
nonuniform grid. For electronic-convergence or timestep comparisons, prepare
separate paired inputs for each setting. Changing an origin force changes `c`;
the reanchored work comparison and a work comparison at fixed `c` are different.

Compare two directions at a common origin, initial total kinetic energy and
evaluation time using `maximum_force_error_eV_A` and `endpoint_work_eV`.
The measured ratio `2W/e^2`, the finite-time work prediction and the leading
small-budget relation are separate diagnostics. The latter requires exact force
matching, locally Lipschitz Hessian and velocity, and a nonzero directional
response; for small budget `epsilon`, `W(t_epsilon) = epsilon^2*C_rw/2 + O(epsilon^3)`.

## 5. Calculate an independent-check bound

Provide an event CSV with columns `accepted,checked,violation`. Flags are zero or
one. `checked` denotes only the independent check of an already accepted force;
refused steps use zero. The violation entry may be empty when unchecked and must
be observed when checked. The accepted force is fixed before drawing the check;
its reference label can inform subsequent forces.

```sh
python residual_work.py verify inputs/checks.csv --probability 0.05 --failure-probability 0.05 --tilt 0.6931471805599453 --output results/verification.json
```

At each prefix the output uses accepted count `N` and detected-violation count
`D` to evaluate

```text
min(1, [lambda*D + log(1/eta)] / [N*(-log(1-p+p*exp(-lambda)))])
```

The bound is undefined before the first acceptance and is written as JSON `null`.
Its simultaneous confidence is `1-eta` when the check probability `p` and positive
tilt `lambda` are chosen in advance and the checks are independent of the current
accepted outcome. This independence is a property of the acquisition protocol;
it cannot be established from a CSV alone. The bound describes accepted-force
violations, while the controlled oscillator measures complete unreferenced
intervals under its separate reference-correction protocol.

## Output and scope

All outputs are newly calculated JSON files. Existing output files are not
overwritten. Paths are command-line arguments and can be relative to the current
directory. No network access is performed by the scripts.

The atomistic workflow reanalyzes supplied calculations; it does not generate
initial structures, train potentials or run electronic-structure calculations.
Reproducing a particular atomistic trajectory additionally requires its initial
state, potential parameters, reference settings and paired force/energy inputs.
