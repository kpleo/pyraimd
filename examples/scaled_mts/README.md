# Scaled surrogate + fixed-model MTS — analytic offline demo

**What this shows.** How the public `scaled` correction wrapper combines
with the fixed-model MTS (respa) NVE path: the surrogate section becomes
`scaled` around a base fast model, and the MTS loop then integrates the
slow residual `F_reference - F_used` with `F_used = c · F_base`. Everything
runs offline on the builtin analytic harmonic backends — the two marked
atoms are test particles in a toy well, not a real silicon material.

**Teaching setup (fixed, do not read as a recommendation).** Reference
`k = 1.0`, base fast model `k = 0.8`, same `r0 = 0.9`, `bias = 0`, and
`scale = 1.25`; inner step 1 fs, `outer_ratio = 4`, 32 inner steps (32 fs,
8 complete outer steps). With these numbers the scaled fast force field
equals the reference one exactly (`1.25 × 0.8 = 1.0`), so the scaled arm's
MTS residual vanishes identically while the uncalibrated arm keeps a
nonzero residual. That is a deliberate mechanical demonstration of the
wiring — it is not evidence about any material, and `1.25` is not a
suggested scale or ratio for any real system.

**A real scale is never chosen inside the run.** In actual use the scalar
must be determined beforehand on independent calibration data (a separate
least-squares analysis of reference/model labels) and then frozen for the
whole run; `scaled` applies it to BOTH the energy and the forces (scaling
only one side would break force consistency). There is no fit API here and
nothing is downloaded.

## Steps

```sh
python make_structure.py                 # writes structure.extxyz (with momenta)

# static checks (no evaluation, no execution)
pyramid validate run_base.toml
pyramid validate run_scaled.toml
pyramid validate run_scaled.toml --check-environment

# surrogate-only readiness probe: one real prediction by the configured
# surrogate (here the scaled harmonic chain), reference never constructed
pyramid validate run_scaled.toml --probe-surrogate

# run both arms
pyramid run run_base.toml
pyramid run run_scaled.toml

# inspect: MTS progress stated as outer/inner split
pyramid inspect run-base
pyramid inspect run-scaled

# export reference boundary labels and surrogate (used, already-scaled)
# predictions separately
pyramid export run-base --force-source reference
pyramid export run-base --force-source base
pyramid export run-scaled --force-source reference
pyramid export run-scaled --force-source base
```

## Expected output shape

Each run commits 9 boundary frames (the initial one plus 8 complete outer
steps = 32.0 fs physical time), with 9 reference evaluations and 33 fast
predictions on the cost ledger. `pyramid inspect` prints, e.g. for either
arm:

```
mts progress          : 8 complete outer steps = 32 inner steps (32.0 fs physical time)
```

`--probe-surrogate` prints the selected surrogate's declared contract
(`energy_kind=energy force_consistent=True forces_conservative=True`), its
fingerprint (the scaled chain's fingerprint starts with `scaled:`), the
probe energy in eV and the force array shape — and states explicitly that
the reference backend was not constructed or evaluated. The probe performs
a real backend call; for machine-learning surrogates run it on a
compute-authorized node, and point `model` at an existing local weights
file (the probe refuses bare base-model names rather than downloading).

Comparing the two exports is where the teaching point lands: the scaled
arm's stored fast labels are already the used (scaled) values — never
multiply by the scale again.
