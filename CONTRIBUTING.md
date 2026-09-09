# Contributing to Pyramid

Use small changes with a runnable example and tests of the affected behavior.
Keep the numerical method, backend integration, and workflow interfaces separate.
State supported energy conventions, constraints, units, and restart behavior for
each new backend or integrator. Optional backends must remain optional imports.

## Local checks

```sh
uv sync --extra dev
git config core.hooksPath .githooks
uv run pytest tests/unit tests/test_smoke.py -q
uv run python scripts/check_public_tree.py --staged
```

The hooks require `uv`. The public-tree check reads staged blobs at commit time
and the proposed revision at push time. CI also checks the submitted tree.
New public documentation or binary assets require updating the reviewed allowlist.
Keep working notes, local machine configuration, credentials, and run outputs
outside the checkout. Supply portable examples and minimal test fixtures instead.

The `reproducibility/` directory is a frozen published supplement. Its file names,
modes, and contents are protected by the publication check; software development
must not modify it. New software tests belong under `tests/` or `examples/`.

Ignore rules and hooks prevent common accidental additions. They do not erase
Git history and can be bypassed; review the staged diff and the target revision
before publishing.

## Reporting a problem

Include the Pyramid, Python, NumPy, ASE, and backend versions, a minimal input,
the expected behavior, and the error. Remove private paths, account names, model
files, and calculation outputs that are not intended for public distribution.
For numerical issues, state the units and a tolerance tied to the calculation.
