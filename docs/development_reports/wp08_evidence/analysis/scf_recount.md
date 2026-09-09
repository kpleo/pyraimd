# WP08 SCF recount (2026-09-09, from run events.jsonl + slurm logs)

Recomputed from the per-run `events.jsonl` task records (this directory),
not from the WP08 prose. Convention: *logical reference request* = one
reference task event (or the pilot validate probe); *actual pw.x launch* =
a reference task that reached the subprocess (status success) plus the
pilot. The runs predate the R6 logical/physical ledger schema, so this is
the historical mapping stated explicitly.

## Per-run reference tasks

- si-singlepoint: 0 (surrogate singlepoint; 1 inference task, 37.5 s)
- si-relax: 0 (surrogate FIRE; 11 inference tasks)
- si-reference: 7 (`md/success`) — 6 steps + initial evaluation, 80.3 s wall
- si-surrogate: 0 (7 inference tasks)
- si-adaptive: 21 = anchor 1 + probe 12 + refusal 2 + verification 6
  (fresh 6-step segment 19 = 1+12+2+4; resume +2 segment 2 = verification 2)
- si-adaptive-short: 20 = anchor 1 + probe 12 + refusal 2
  + verification 4 success + verification 1 failed
  (fresh 4-step segment 13 = 1+8+1+3; failed resume segment 1 failed;
  post-fix resume segment 6 = probe 4 + refusal 1 + verification 1)
- al-run-singlepoint: 0 (surrogate singlepoint; 1 inference task, 152.7 s)
- al-relax: 0 (surrogate FIRE; 17 inference tasks)
- al-reference: 9 (`md/success`) — 6 steps (7) + plain resume +2 (2)
- al-surrogate: 0 (7 inference tasks)

## Totals

- Logical reference requests: 57 in run events + 1 pilot = **58**
- Actual pw.x launches: 56 successful tasks + 1 pilot = **57**
- The one failed task (`si-adaptive-short-task-33`, verification, eval 5):
  `FileExistsError(17)` at attempt-directory creation after 0.05 s — pw.x
  never started (an SCF here takes 13–24 s; the successful retry as
  `task-34` took 18.1 s). Counts: logical 1, actual 0.

## Corrected per-purpose breakdown (replaces WP08's 6/24/9/16/1/1)

- pilot `validate --probe-backends` (slurm log 7625052: energy
  −5087.018956 eV, 12.521 s): 1 actual
- anchor: 2 actual (1 + 1)
- refusal: 4 actual (2 + 2)
- probe: 24 actual (12 + 12)
- verification: 10 actual success (6 + 4) + 1 failed (logical only)
- plain-MD: 16 actual (7 Si + 9 Al)
- Sums: actual 1+2+4+24+10+16 = 57; logical 57+1 = 58.

WP08's "anchor 6" used the runner's printed summary convention, which
merges refusals into the anchor counter (3 + 3 per run); "check 9"
double-counted the failed verification (4 + 4 + 1-failed, while also
listing it as the "1 failed attempt"). WP08's total of 58 was correct as
logical requests; as physical SCF launches it is 57.

## Fingerprint / cache

WP08's "QeEngine 无 fingerprint" is inconsistent with the records:
`run_start.reference_id` and `manifest.json` carry
`qe-pbe-d3:d23205a4a5469f4f` (Si settings) and `qe-pbe-d3:b5880bfa70564c2d`
(Al settings, metallic mv smearing) — the WP05 settings fingerprint was
present and correctly distinct per recipe. The label cache was armed; it
recorded 0 hits because every checked geometry was new (MD never revisits
an identical configuration) and the WP02 label cache is per-process
in-memory (cold after each resume). logical == actual within these runs is
therefore expected and not evidence of a missing cache path.
