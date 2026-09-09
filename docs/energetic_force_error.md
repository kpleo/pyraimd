# Energetic force errors

A force error changes a trajectory through its direction as well as its
magnitude. Pyramid follows both: the largest atomic residual measures force
accuracy, while the residual projected along atomic motion measures signed
work. A local directional response links these quantities before the next
reference calculation.

## Reference anchors

Let `X0` be a reference configuration, `F_base` the fast-potential force and
`F_ref` the reference force. Define a constant correction at the anchor:

```text
c = F_ref(X0) - F_base(X0)
F_anchor(X) = F_base(X) + c
R(X) = F_anchor(X) - F_ref(X)
e(X) = max_i |R_i(X)|
```

The norm in `e` is the Euclidean norm of each atomic three-vector. The correction
sets `R(X0) = 0` and remains fixed within the segment. When the base forces are
energy derivatives, the corresponding anchored potential is

```text
U_anchor(X) = U_base(X) - c · (X - X0)
```

An additive energy constant can align the two potentials at `X0` without changing
their forces or work differences. Anchoring corrects the local force offset;
the directional response describes how the residual grows as the atoms move.

With `FixAtoms`, the default `active_dofs_max_atom` metric measures `e` on
free coordinates, and probe directions and driving forces are projected onto
those coordinates. `all_atoms_max_atom` instead includes fixed atoms in the
force-error norm. Raw forces remain available for diagnostics; work uses the
actual constrained displacement, so fixed atoms contribute no displacement work.

## Local directional response

A direction `u` has unit Euclidean norm over all `3N` coordinates. Paired
reference and base forces at positive and negative displacements give

```text
q_h = [R(X0 + h*u) - R(X0 - h*u)] / (2*h)
```

The two-scale estimate uses central probes at two displacement magnitudes to
measure the response and its sensitivity to probe scale. Write the selected
response as `q`, and define

```text
g = max_i |q_i|
kappa = u · q
C_rw = kappa / g^2
```

`C_rw` is the **directional residual-work coefficient**, with units of
angstrom squared per eV. It relates leading directional work to squared force
error. For displacement `a*u`,

```text
e_linear = |a|*g
W_predicted = a^2*kappa/2 = C_rw*e_linear^2/2
```

Two directions can therefore have similar maximum force errors and different
energetic effects. The sign of `kappa` retains whether the residual adds or
removes energy along that motion.

The numerical entry point is `pyraimd2.energetics.estimate_responses`. It accepts
unit directions of shape `(D, N, 3)`, two increasing probe scales, and positive
and negative displacement and residual arrays of shape `(D, 2, N, 3)`. It returns
one `DirectionalResponse` per direction. The larger probe scale supplies `q`;
all supplied directions share the transverse estimate. One direction at two
scales requires four displaced reference evaluations in addition to the anchor.
Include these probes and independent checks when reporting reference cost.

## Prospective force and work forecasts

For a displacement `d = X - X0`, decompose

```text
a = u · d
z = d - a*u
```

The forecast combines the linear response with an empirical envelope:

```text
B(d) = |a|*g + floor + eta*|a| + K*|z| + M*|d|^2/2
W_predicted(d) = a^2*kappa/2
```

The numerical floor is in eV/angstrom. The coefficients `eta` and `K` describe
parallel sensitivity and transverse response in eV/angstrom squared; `M`
describes the remainder in eV/angstrom cubed. Probe-scale comparisons and
probe-vector defects supply these local estimates.

Choose the force budget, probe scales, numerical floor and forecast domain
before using the forecast for subsequent motion. The domain limits the elapsed
time and the departure from the probed direction. At each force evaluation,
Pyramid uses the anchored fast force when `B` meets the budget within that
domain, and requests a reference when a new anchor is needed. The admitted
prefix on a sampled path ends at its first failing state; its endpoint defines
the discrete force-accuracy horizon.

The force forecast uses the displacement and the previously measured response.
Reference forces at the new geometry enter the subsequent check, rather than
the prospective prediction.

For a fitted `response`, the array-based API is:

```python
forecast = response.forecast(
    displacement=positions - origin_positions,
    elapsed_fs=elapsed_fs,
    force_budget=0.1,
    numerical_floor=0.001,
    time_cap_fs=0.5,
    transverse_cap=0.1,
)
```

These numerical settings are illustrative. The returned `Forecast` exposes
`linear_error`, `envelope`, `predicted_work`, `transverse_fraction`, `in_domain`
and `admitted`. Each call assesses one point; a caller analyzing a path must
also enforce the uninterrupted admitted prefix.

## Signed work at a reference endpoint

With one fixed base potential and correction, endpoint energies give

```text
W(X) = [U_ref(X) - U_ref(X0)]
     - [U_base(X) - U_base(X0)] + c · (X - X0)
```

This is the line integral of `R` along the segment when the reference and base
forces differentiate their respective energies. Positive and negative values
retain the direction of energy transfer caused by the residual.

With intermediate paired force labels, numerical integration gives an additional
comparison:

```text
Q = sum_j (R_j + R_(j+1))/2 · (X_(j+1) - X_j)
```

Atomic or fixed-group contributions to `Q` add to the same signed total.
Comparing `Q` with `W` tests the resolution of the force quadrature. In a
conservative MD segment, the reference-Hamiltonian change separates as

```text
Delta H_ref = W + Delta H_anchor
```

This separates the residual's energetic effect from the integration drift of
the anchored dynamics. Each reanchor or model update starts a new segment;
evaluate the previous segment's work with its original potential and correction.
The numerical functions are `residual_work` for endpoint energies and
`integrate_residual_work` for paired arrays of shape `(T, N, 3)`. The latter
returns cumulative work per atom with shape `(T, N)`; summing the atom axis
gives the total.

## Independent accepted-force checks

An independent check fixes the accepted force first, draws whether to evaluate
the reference, then compares that reference with the accepted force. Its label
can be used for subsequent adaptation and anchoring. Routine reference requests
and directional probes serve different purposes and are not counted as these
independent checks.

For `N` accepted force evaluations and `D` detected violations under a fixed
check probability `p`, the numerical core evaluates an upper bound on the
accepted-force violation fraction:

```text
min(1, [lambda*D + log(1/delta)]
       / [N*(-log(1 - p + p*exp(-lambda)))])
```

Choose `p`, the positive tilt `lambda` and failure probability `delta` in
advance. With independent checks, the confidence statement holds simultaneously
over prefixes at level `1-delta`. Before any acceptance, the fraction is
undefined. More independent observations make this assessment informative;
the forecast and the measured checks remain separately readable outputs.
Setting `check_probability=0` disables this assessment. At `p=1`, every
acceptance is checked and the implementation reports the exact observed
violation fraction.

## Interpretation and current scope

The local envelope is inferred from finite probes and is evaluated against
later references. Its domain and force budget are operating settings, not a
certification of every force. The independent-check bound concerns accepted
force evaluations under the stated sampling protocol. Signed residual work
measures an energetic contribution and should be interpreted with its sign;
thermostat, constraint and variable-cell work require their own accounting.
The current energetic loop supports fixed-cell NVE with optional `FixAtoms`
and requires energy-consistent forces on each segment. NVT, other constraints
and variable-cell dynamics are unsupported. At finite electronic smearing, use
the thermodynamic potential whose negative gradient gives the reported forces.
The coefficient `C_rw` requires a nonzero directional response.

See the [architecture guide](architecture.md) for backend units and the
[harmonic example](../examples/energetic_loop.py) for a complete run.
The [Supplementary Materials](../reproducibility/force_error/README.md)
provide saved force-error data and analysis code with their own reproduction
instructions and scope.
