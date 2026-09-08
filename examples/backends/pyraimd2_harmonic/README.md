# pyraimd2-harmonic: example backend plugin

Minimal third-party backend plugin for pyraimd2, used to verify that a new
backend can be added without modifying the core package. It provides two
analytic backends through the `pyraimd2.backends` entry-point group:

- `harmonic_reference` — an engine (`compute`) with E = 1/2 k |r - r0|^2.
- `harmonic_surrogate` — a surrogate (`predict`) for the same well plus a
  deterministic force bias.

Install into the environment that already has pyraimd2:

```sh
uv pip install ./examples/backends/pyraimd2_harmonic
```

Then:

```python
from pyraimd2.backends import available_backends, create_backend

print(available_backends())
engine = create_backend("harmonic_reference", kind="engine", k=1.2)
```

The factories only need keyword arguments; capability requirements are
checked at creation time, e.g. `create_backend("harmonic_reference",
require=EngineCapabilities(stress_available=True))` is rejected because the
harmonic backends declare no stress.
