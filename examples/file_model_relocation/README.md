# File-model relocation: move a run and its model file, then resume

A complete, offline demonstration of run relocation through the Python
workflow API, in two real processes. An analytic harmonic potential reads
its stiffness from a plain-text model file (the
[pyraimd2_filemodel](../backends/pyraimd2_filemodel/) example plugin); the
file is a declared, content-verified resource of the run. You will:

1. create a short NVE run (process 1) and exit;
2. move the run directory together with the model file — a directory name
   with a space or non-ASCII characters (including emoji) works the same;
3. watch the resume refuse clearly because the recorded model path is gone;
4. resume with an explicit `resource_paths` mapping (process 2) and export
   the committed trajectory.

No external programs, no model downloads, finishes in seconds.

## Setup

```sh
uv pip install .                                        # the core package
uv pip install ./examples/backends/pyraimd2_filemodel   # the example plugin
```

Run the commands from the repository root; the demo directory is yours to
choose. `--output` accepts a path that does not exist yet or a completely
empty directory; anything else is refused with nothing written.

## Walkthrough

```sh
# process 1: new run — 3 steps, writes everything under demo/origin
python examples/file_model_relocation/new_run.py --output demo/origin

# shell: relocate the run and the model file together
mv demo/origin "demo/moved run"

# process 2a: resume without a mapping — refused, naming the missing file
python examples/file_model_relocation/resume_run.py --run "demo/moved run" --extra-steps 2

# process 2b: resume with the explicit mapping — continues to 5 steps,
# then exports demo/moved run/export-driving.extxyz
python examples/file_model_relocation/resume_run.py --run "demo/moved run" --extra-steps 2 \
    --model "$PWD/demo/moved run/inputs/model.dat"
```

The export never overwrites: if the target (the default
`export-driving.extxyz` in the run directory, or your `--export` path)
already exists, the script refuses before resuming and names the conflict,
so a retry never adds a second batch of steps; pass a different `--export`
path or move the existing file aside.

Expected: `new_run.py` prints `"steps_completed": 3` and the declared
resource `reference.potential` with the model's original path. Step 2a
exits with code 2 and a message like `resource 'reference.potential':
/old/path/inputs/model.dat is missing or not a regular file — restore the
original or map a valid replacement through resource_paths=...`. Step 2b
prints `"steps_completed": 5` and exports 6 frames (the initial evaluation
plus five complete steps).

## The mapping rules

- Keys are the baseline's declared `<section>.<role>` names — this example
  declares exactly one, `reference.potential` (see the run's
  `file_resources.json`). Unknown keys are refused.
- Values must be absolute paths of existing regular files (not symlinks).
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
compose with an online updater. There is no CLI for relocation yet: the
two scripts' arguments are example plumbing around
`resume_workflow(..., resource_paths=...)`, not a product command.

To write your own backend with a declared file resource, see the plugin's
[source](../backends/pyraimd2_filemodel/src/pyraimd2_filemodel/__init__.py)
and the backend conventions in [CONTRIBUTING.md](../../CONTRIBUTING.md).
The `run.toml` fields are documented in
[docs/configuration.md](../../docs/configuration.md).
