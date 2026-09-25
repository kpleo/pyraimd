# Fixed-model MTS (respa) NVE — analytic offline demo

**Experimental.** The MTS path integrates a fixed slow residual
`F_slow = F_reference - F_fast` around a fixed fast model with symmetric
outer half-kicks, for a fixed cell/composition NVE run with definite
initial momenta. This demo runs entirely offline on the builtin
analytic harmonic backends — the two marked atoms are test particles in
a toy well, not a real silicon material.

## The shortest path: the init template

The installed package ships this demo as the `harmonic-mts` init template —
the same `run.toml` plus the same `structure.extxyz` with the fixed initial
momenta embedded, no source checkout needed:

```sh
pyramid init --template harmonic-mts --output demo
pyramid validate demo/run.toml
pyramid validate demo/run.toml --check-environment
pyramid run demo/run.toml
pyramid inspect demo/runs/harmonic-mts-demo
pyramid export demo/runs/harmonic-mts-demo --force-source reference   # reference boundary labels
pyramid export demo/runs/harmonic-mts-demo --force-source base        # surrogate predictions
pyramid resume demo/runs/harmonic-mts-demo --steps 64                 # extend by 64 inner steps
pyramid inspect demo/runs/harmonic-mts-demo
```

128 inner steps of 1 fs at `outer_ratio = 4` give 32 complete outer steps:
128.0 fs physical time, 33 committed boundary frames (the initial one plus
32), 33 reference evaluations and 129 fast predictions on the ledger. The
64-inner-step resume takes the run to 192.0 fs: 48 complete outer steps and
49 frames. `pyramid inspect` states the outer/inner split explicitly:

```
mts progress          : 32 complete outer steps = 128 inner steps (128.0 fs physical time)
```

## This directory: the same demo from a source checkout

`run.toml` and `make_structure.py` spell out the template's content.
`make_structure.py` is the optional manual route to the identical initial
state (two Si-labelled test particles, mass 28.085 amu, definite momenta via
the ASE unit conversion, never re-randomized):

```sh
python make_structure.py      # writes structure.extxyz (with momenta)
pyramid validate run.toml
pyramid validate run.toml --check-environment
pyramid run run.toml
pyramid inspect run
pyramid export run --force-source reference   # reference boundary labels
pyramid export run --force-source base        # surrogate predictions
pyramid resume run --steps 64                 # extend by 64 inner steps
```

Notes:

- `timestep_fs` is always the inner step; `steps` and
  `resume --steps N` count inner steps and must be multiples of
  `outer_ratio` (only complete outer steps exist — never rounded).
- There is no single per-step "driving force" in MTS, so
  `--force-source driving` refuses for these runs; export the reference
  or base labels explicitly.
- The initial structure must carry momenta; this mode never
  thermalizes. Constraints, NVT/NPT, online model updates and adaptive
  step sizes are outside this experimental path and are refused at
  configuration time.
- Resume continues from the last committed outer boundary and reuses
  its cached boundary labels — a continuous run and a segmented run make
  exactly the same backend calls.

## Swapping in your own backends (QE reference, MACE surrogate)

The analytic harmonic pair exists so the demo runs offline. For a real
material, replace the `[reference]` and `[surrogate]` sections with your own
configured backends and keep this file's `[dynamics]`/`[checkpoint]`
semantics unchanged. The configuration shape — QE `pw_cmd`,
`pseudo_dir`/`pseudos`, MACE `model`, the `mace` install extra and where
input files live — is documented in
[../al_surface_qe_mace/](../al_surface_qe_mace/) and
[../si_bulk_qe_mace/](../si_bulk_qe_mace/); their READMEs cover installing
the extras and providing `pw.x`, pseudopotentials and model files. Preflight
the machine before running:

```sh
pyramid validate run.toml                       # configuration only
pyramid validate run.toml --check-environment   # local prerequisites: pw.x, packages, model files
```

MTS checks both backends' declarations before the first evaluation: each
side must declare force-consistent conservative forces, a known
`energy_kind` and a content fingerprint; a backend that cannot declare them
is refused before any calculation.

## Preparing definite initial momenta

MTS never thermalizes, so the structure file must carry the initial momenta.
In extxyz that is a per-atom momenta column in ASE units, declared on the
`Properties` line — exactly what the template structure and
`make_structure.py` produce:

```
Properties=species:S:1:pos:R:3:masses:R:1:momenta:R:3
```

Any writer that sets ASE momenta works (`atoms.set_momenta(...)`, then
`ase.io.write`); a file without the momenta column is refused before the
first evaluation.
