# Persistent QE density chain (plain serial reference MD)

A small, runnable tour of the opt-in persistent density chain: every
evaluation's charge density is published into the run-owned registry as
an immutable generation, the next calculation warm-starts from the
published seed, and disk use stays bounded by construction.

Works with plain serial reference MD (`task.kind = "md"`,
`task.mode = "reference"`) on the `qe` and `qe-ase` backends.  Recipe,
adaptive and surrogate combinations are refused up front; the chain
requires `[scratch] retention = "all"`.

## Enable

See `run.toml` in this directory.  The two switches:

```toml
[scratch]
root = "./tmp"          # one managed scratch root
retention = "all"       # required by the density chain

[density]
persist = true          # publish every density into restart/density/
```

With `startpot_file = true` (the QE default in this mode) the next
evaluation stages the published seed as its starting density.  Nothing
changes for runs that leave `[density]` unset.

## Observe the space

- `run/restart/density/g000001/` … one directory per published
  generation (charge density + the QE schema XML, plus the PAW `paw.txt`
  when the run produces one), each with a `manifest.json` (content
  digests, producing attempt identity, compatible settings).
- `run/restart/density/state.json` — `latest`, the `attached` history,
  and `reclaimed`, the durable tombstones of deliberately deleted
  generations (a reclaimed generation is never mistaken for corruption,
  and its number is never reused).
- `run/scratch_records/` — the authoritative lifecycle of every attempt
  (`kept` → `archived` → `cleaned`; failures stay `failed_kept`).  A
  producer attempt carries its release receipt here: `pending` until a
  later calculation has independently read its exact seed and succeeded,
  then `consumed` with the consuming attempt's identity.  "Read" means
  the consuming solver's raw output actually shows it: QE's
  `The initial density is read from file` marker naming the attempt's own
  staged save tree, bound to the launch input (`startingpot = 'file'`)
  and the archived raw output by content digest.  A successful
  calculation that never reports the read — an unknown or silent output
  format — keeps the producer and every later source rather than
  assuming one; staging plus success is never enough.
- Dry-run at any time (nothing is deleted):

```sh
python -c 'from pyraimd2.runtime.restart import plan_density_reclaim; \
import json, sys; \
print(json.dumps(plan_density_reclaim(sys.argv[1]), indent=1))' run
```

## Resume

```sh
python -c 'from pyraimd2.workflows import resume_workflow; \
resume_workflow("run", 2)'   # two more steps, in a fresh process
```

Resume binds exactly the density generation of the one authoritative
restored boundary — the committed evaluation whose row supplied the
restored positions/forces.  A boundary whose referenced generation is
missing or corrupt is refused before anything is written or computed
(never a silent swap to a newer generation); a legacy record without the
field initializes from the configured external source.  The `resumed`
event records the actual boundary evaluation and the bound density
generation and content digest.

## Reclaim old generations

The driver reclaims automatically after each step commit and checkpoint
retention update.  On demand, dry-run first and then execute against the
fresh plan (a stale plan is refused):

```sh
python - <<'EOF'
from pyraimd2.runtime.restart import (plan_density_reclaim,
                                      execute_density_reclaim)
plan = plan_density_reclaim("run")          # per-generation keep/hold/reclaim reasons
receipt = execute_density_reclaim("run", plan=plan)
print(receipt["status"], receipt["reclaimed"])
EOF
```

## What is kept, and why

Only generations the run fully owns — once attached, validated,
unreferenced — are deleted.  Always kept: the `latest` pointer, every
retained checkpoint's referenced generation, the committed recoverable
boundary's generation, in-flight consumer inputs, and every producer
seed not yet independently consumed (the terminal producer of a finished
run keeps its full scratch — a protected resource, not a cleanup
failure).  Publish leftovers, corrupt or never-attached directories,
external `density_source` trees, symlinks and anything outside the
registry are never auto-deleted.  Interrupted deletions resume from
their tombstones only while a readable manifest still confirms ownership
(directory name, durable tombstone and run_root/run_id must agree;
payload already deleted needs no re-validation); a same-number foreign
tree is held forever, a missing manifest is never resumed, and a
re-referenced tombstone stays suspended.  Per-generation failures are
reported in the receipt and never block the rest.

Honest accounting: the seed generations plateau (latest + retained
checkpoint references + boundary + one unconsumed producer), while
lightweight results (events/trajectory/summaries), per-attempt
`pw.in`/`pw.out` archives and the registry state metadata still grow
with the number of steps — no constant-whole-disk claim.

## Verified scope

Verified on QE 7.5 (HDF5 build), PAW, non-spin-polarized SCF restarts
(the seed pack carries the charge density, the schema XML and the PAW
`paw.txt`).  Other QE builds/formats/profiles are not claimed: an
unproven profile simply keeps its producer scratch until a later
calculation proves the seed independently sufficient.

## Try the whole flow without QE

```sh
sh examples/density_persist_qe/demo_fake.sh
```

runs 3 steps + a fresh-process 2-step resume on a fake `pw.x` (a shell
script printing a bundled valid output fixture), then inspects the
registry, prints the consumption receipts and tombstones, and does a
dry-run plus a real reclaim.  **It verifies the program flow only; it
produces no real DFT results.**

With a real `pw.x` and pseudopotentials, point `run.toml`'s `pw_cmd` and
`pseudo_dir` at them and use your own structure file.
