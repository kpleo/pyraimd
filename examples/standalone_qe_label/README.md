# Standalone QE label on the unified scratch lifecycle

A template for one-off energy/force labels from Quantum ESPRESSO on the
**unified managed tmp root**: the attempt runs in its exclusive directory
under one shared `tmp/`, the verified result is archived into the label's
persistent run directory, and the attempt's scratch subtree is reclaimed
immediately — wavefunction scratch never accumulates across labels.

## Lifecycle

allocate (exclusive `tmp/<run-uuid>/reference/<request-id>/attempt-N/`)
→ run → parse-verified (complete finite energy/forces/stress) → archive
(the durable result out of scratch, fsynced) → cleaned. Failures are
kept with their reason; an archived attempt whose cleanup failed is
`cleanup_pending` and safely retryable — a successful DFT is never
rerun by its cleanup.

## Configuration

- Engine: `QeConfig(scratch_root=..., retention=...)`; workflow:
  `[scratch] root = "...", retention = "..."` in `run.toml` (a relative
  root resolves against the configuration file's directory).
- `retention = "all"` (default): keep the attempt's scratch (the unified
  root still applies). `retention = "results"`: archive the verified
  result and reclaim the attempt's scratch — it does not compose with
  `startpot_file` / `density_source` (a chained density lives in the
  scratch it would reclaim) and the combination is refused up front.
- Without `scratch_root` (or the section) everything behaves exactly as
  before: attempts live under the run's own `calculations/`.
- Records (`scratch_records/`, outside scratch) are the authoritative
  lifecycle state; `pw.in`/`pw.out` and the marked density manifest are
  archived; `_last_density_dir` never points at a reclaimed `.save`.

## Usage

```sh
python examples/standalone_qe_label/label.py \
    --structure structure.extxyz --run-root runs/si-label \
    --scratch-root ./tmp --pw-cmd "pw.x" --pseudo-dir ./pseudos \
    --label case-00
```

The script writes `runs/si-label/label-case-00.json` (energy/forces/
provenance and the scratch record state), the archived
`runs/si-label/case-00-000000/attempt-1/{pw.in,pw.out,density_manifest.json}`,
and prints the reclaim summary. Requires a real `pw.x` and
pseudopotentials. Inspect or retry a root with:

```sh
pyramid scratch inspect --root ./tmp
pyramid scratch clean --root ./tmp --dry-run
```

## Try the whole flow without QE

`demo_fake.sh` runs the full lifecycle end-to-end on a fake `pw.x`
(a shell script printing a bundled valid output fixture) — install,
configure the root, run, read the archived result, inspect and clean,
including an idempotent repeat.  **It verifies the program flow only;
it produces no real DFT label.**

```sh
sh examples/standalone_qe_label/demo_fake.sh /tmp/scratch-demo
```
