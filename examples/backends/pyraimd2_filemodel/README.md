# pyraimd2-filemodel: example file-backed backend plugin

A minimal third-party backend whose potential is read from a plain-text
model file, registered through the `pyraimd2.backends` entry-point group:

- `file_model_reference` — an engine (`compute`) with
  E = 1/2 k |r - r0|^2, where the stiffness k is the first line of the
  model file named by the backend's `model` option.

The factory declares `file_parameters={"model": "potential"}`: the option
of the same name becomes a declared, content-verified file resource of
every run (baseline key `reference.potential`). This is what makes the run
relocatable through `resume_workflow(..., resource_paths=...)`.

Install into the environment that already has pyraimd2:

```sh
uv pip install ./examples/backends/pyraimd2_filemodel
```

Then:

```python
from pyraimd2.backends import create_backend

engine = create_backend("file_model_reference", kind="engine",
                        model="/absolute/path/to/model.dat")
```

The complete new-run → relocate → mapped-resume → export walkthrough is in
[../../file_model_relocation/](../../file_model_relocation/).
