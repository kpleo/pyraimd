"""Example pyraimd2 backend plugin: an analytic potential stored in a file.

The potential is a harmonic well whose stiffness is read from a plain-text
model file at construction time.  The file is declared to pyraimd2 through
``file_parameters={"model": "potential"}`` — the single declaration that
turns the backend option of the same name (``model``) into a first-class,
content-verified file resource: every run records the file's content digest
in its resource baseline, and a resumed run re-verifies the current file
byte-for-byte before any computation.  The relocation walkthrough that uses
this plugin lives in ``examples/file_model_relocation/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ase.calculators.calculator import Calculator

from pyraimd2.engines.ase_engine import AseEngine


class FileModelCalculator(Calculator):
    """E = 1/2 k |r - r0|^2, F = -k (r - r0); k from the model file.

    The model file's first line is the stiffness k in eV/angstrom^2 — the
    file genuinely feeds the physics, so a run resumed against a different
    file (or none) must not silently compute.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model: str, r0: float = 0.9) -> None:
        super().__init__()
        self.parameters = {"model": str(model), "r0": float(r0)}
        self._k = float(Path(model).read_text().splitlines()[0])
        self._r0 = float(r0)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        dr = self.atoms.positions - self._r0
        self.results = {"energy": 0.5 * self._k * float((dr**2).sum()),
                        "forces": -self._k * dr}


def reference_factory(**kwargs) -> AseEngine:
    """Build the engine; the config's ``model`` option names the file.

    ``file_parameters`` maps the backend option name (``model``) to its
    resource role (``potential``); the run's baseline key becomes
    ``reference.potential``.
    """
    return AseEngine(FileModelCalculator(**kwargs),
                     file_parameters={"model": "potential"})


reference_factory.backend_kind = "engine"
