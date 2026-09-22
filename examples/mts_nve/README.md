# Fixed-model MTS (respa) NVE — analytic offline demo

**Experimental.** The MTS path integrates a fixed slow residual
`F_slow = F_reference - F_fast` around a fixed fast model with symmetric
outer half-kicks, for a fixed cell/composition NVE run with definite
initial momenta. This demo runs entirely offline on the builtin
analytic harmonic backends — the two marked atoms are test particles in
a toy well, not a real silicon material.

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
