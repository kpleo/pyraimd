# File-model relocation: move a run and its model file, then resume

A complete, offline demonstration of run relocation in two real processes,
via the Python workflow API or the CLI (one **alternative** route each,
chosen below). An analytic harmonic potential reads its stiffness from a
plain-text model file (the
[pyraimd2_filemodel](../backends/pyraimd2_filemodel/) example plugin); the
file is a declared, content-verified resource of the run. You will:

1. create a short NVE run (process 1) and exit;
2. move the run directory together with the model file — a directory name
   with a space or non-ASCII characters (including emoji) works the same;
3. watch the resume refuse clearly because the recorded model path is gone;
4. resume with the explicit mapping (process 2) and export the committed
   trajectory — via the Python script **or** the CLI, not both.

No external programs, no model downloads, finishes in seconds.

## Setup

```sh
uv pip install .                                        # the core package
uv pip install ./examples/backends/pyraimd2_filemodel   # the example plugin
```

Run the commands from the repository root; the demo directory is yours to
choose. `--output` accepts a path that does not exist yet or a completely
empty directory; anything else is refused with nothing written.

## Walkthrough (shared start, then ONE of the two resume routes)

```sh
# process 1: new run — 3 steps, writes everything under demo/origin
python examples/file_model_relocation/new_run.py --output demo/origin

# shell: relocate the run and the model file together
mv demo/origin "demo/moved run"

# process 2a: resume without a mapping — refused, naming the missing file
# (the refusal changes nothing in the run; either route below can follow)
python examples/file_model_relocation/resume_run.py --run "demo/moved run" --extra-steps 2
```

From here choose exactly ONE resume route: route A (Python, step 2b
below) or route B (the CLI further down). Running both on the same run
would add a second batch of steps and collide on the export file — this
demo needs no `--force` anywhere.

```sh
# process 2b (route A, Python): resume with the explicit mapping —
# continues to 5 steps, then exports demo/moved run/export-driving.extxyz
python examples/file_model_relocation/resume_run.py --run "demo/moved run" --extra-steps 2 \
    --model "$PWD/demo/moved run/inputs/model.dat"
```

The export never overwrites — note the two different scopes. The *script*
above resumes and exports in one invocation, so it checks the export
target (the default `export-driving.extxyz` in the run directory, or your
`--export` path) before resuming and refuses a conflict up front, never
stranding a second batch of steps; pass a different `--export` path or
move the existing file aside. The CLI below instead has two independent
commands: `pyramid export` refuses an existing output file unless you
pass `--force`, and that refusal only affects the export — it does not
undo the resume that already completed.

Expected: `new_run.py` prints `"steps_completed": 3` and the declared
resource `reference.potential` with the model's original path. Step 2a
exits with code 2 and a message like `resource 'reference.potential':
/old/path/inputs/model.dat is missing or not a regular file — restore the
original or map a valid replacement through resource_paths=...`. Route A
prints `"steps_completed": 5` and exports 6 frames (the initial evaluation
plus five complete steps).

## Route B: the same resume on the CLI

The `pyramid resume` command exposes the same mapping (repeatable
`--resource BACKEND.ROLE=PATH`). This is the **alternative** to step 2b
above, not a follow-up: start it from a relocated run that has NOT been
resumed or exported yet (fresh 3 steps, then `mv`). If you already ran
route A on this run, recreate the demo in a new directory instead of
continuing here — rerunning either route on the same run adds a second
batch of steps, and the default export would refuse the existing file.

```sh
# from any working directory; a relative PATH resolves against THAT
# directory (the shell's cwd), never against the run directory
cd demo
pyramid resume "moved run" --steps 2 \
    --resource 'reference.potential=moved run/inputs/model.dat'
pyramid inspect "moved run"
pyramid export "moved run" --force-source driving
```

Expected, like route A: the resume continues to 5 steps (inspect shows
`complete steps        : 5`) and the export writes 6 frames to
`moved run/export-driving.extxyz`. Usage errors are refused before
anything runs: a missing `=` separator, an empty role key, an empty path,
or the same role given twice (even with the same path) all exit with code
2 and never touch the run. A mapped file that is missing or whose bytes
differ from the baseline is refused with the resource named; fix the
mapping and rerun the same command. Without `--resource` the command
keeps its original resume semantics.

## The mapping rules

- Keys are the baseline's declared `<section>.<role>` names — this example
  declares exactly one, `reference.potential` (see the run's
  `file_resources.json`). Unknown keys are refused.
- For the Python `resource_paths` API, values must be absolute paths of
  existing regular files (not symlinks). The CLI `--resource` additionally
  accepts a relative path, resolved against the shell's current working
  directory; the same file and content-identity checks then apply.
- The file's current bytes are re-read and compared against the baseline's
  SHA-256 before any backend is built. A file with different content —
  even one byte changed, with size and mtime preserved — is refused.
- Only the declared option slots are rebound, in memory: the run's config
  copy, manifest, baseline and history are never rewritten. Each verified
  binding appends one receipt under `resource_bindings/`; a receipt is a
  record, not a new trust baseline and not a progress claim.
- The mapping is never remembered: every later restart while the recorded
  path is stale needs the same explicit mapping again.

## Support boundary

Relocation is supported for a **fixed model** (byte-identical file at a new
location), in the **same environment** (same pyraimd2 and backend code —
the physical identity checks still apply), for **runs created with
declared file resources** (the opt-in `file_parameters` declaration of the
backend; older runs without a baseline are never upgraded). It does not
compose with an online updater. The CLI surface is exactly the one
`--resource` option of `pyramid resume` — there is no separate relocation
command; the two scripts' `--run`/`--model` arguments remain example
plumbing around `resume_workflow(..., resource_paths=...)`.

To write your own backend with a declared file resource, see the plugin's
[source](../backends/pyraimd2_filemodel/src/pyraimd2_filemodel/__init__.py)
and the backend conventions in [CONTRIBUTING.md](../../CONTRIBUTING.md).
The `run.toml` fields are documented in
[docs/configuration.md](../../docs/configuration.md).
