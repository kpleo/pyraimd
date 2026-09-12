"""Directed checks for the opt-in file-content identity (T1).

A declared calculator file parameter joins the versioned
``ase-file-identity-v1`` branch: content in, path out.  Undeclared
adapters keep the pre-existing fingerprint semantics byte-for-byte.
Analytic test calculators only — zero real backend budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.surrogate.ase_surrogate import AseSurrogate


class FileBacked(Calculator):
    """A test calculator whose top-level parameter names a model file."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model_path, *, epsilon=1.0):
        super().__init__()
        self.parameters = {"model": str(model_path), "epsilon": epsilon}
        self._model_path = str(model_path)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        self.results = {"energy": 0.0,
                        "forces": np.zeros((len(self.atoms), 3))}


class SiblingFileBacked(Calculator):
    """A different calculator class with the same parameters."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model_path, *, epsilon=1.0):
        super().__init__()
        self.parameters = {"model": str(model_path), "epsilon": epsilon}

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=None):
        super().calculate(atoms, properties, system_changes or [])
        self.results = {"energy": 0.0,
                        "forces": np.zeros((len(self.atoms), 3))}


def _model(directory, name="model.dat", content="weights-v1"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content)
    return path


def test_same_content_different_path_same_identity(tmp_path):
    first = AseEngine(FileBacked(_model(tmp_path / "a")),
                      file_parameters={"model": "potential"})
    second = AseEngine(FileBacked(_model(tmp_path / "b")),
                       file_parameters={"model": "potential"})
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith("ase:")


def test_one_byte_content_change_changes_identity(tmp_path):
    path = _model(tmp_path)
    first = AseEngine(FileBacked(path), file_parameters={"model": "potential"})
    path.write_text("weights-v2")  # same length, different bytes
    second = AseEngine(FileBacked(path), file_parameters={"model": "potential"})
    assert first.fingerprint != second.fingerprint


def test_role_parameter_class_and_flags_change_identity(tmp_path):
    path = _model(tmp_path)
    base = AseEngine(FileBacked(path), file_parameters={"model": "potential"})
    role = AseEngine(FileBacked(path), file_parameters={"model": "weights"})
    assert base.fingerprint != role.fingerprint
    other_class = AseEngine(SiblingFileBacked(path),
                            file_parameters={"model": "potential"})
    assert base.fingerprint != other_class.fingerprint
    flags = AseEngine(FileBacked(path), file_parameters={"model": "potential"},
                      force_consistent=True)
    assert base.fingerprint != flags.fingerprint
    stress = AseEngine(FileBacked(path), file_parameters={"model": "potential"},
                       include_stress=True)
    assert base.fingerprint != stress.fingerprint


def test_physical_parameter_and_explicit_identity_change_identity(tmp_path):
    path = _model(tmp_path)
    base = AseEngine(FileBacked(path), file_parameters={"model": "potential"})
    epsilon = AseEngine(FileBacked(path, epsilon=2.0),
                        file_parameters={"model": "potential"})
    assert base.fingerprint != epsilon.fingerprint
    explicit = AseEngine(FileBacked(path),
                         file_parameters={"model": "potential"},
                         identity="my-model-v1")
    assert base.fingerprint != explicit.fingerprint


def test_undeclared_fingerprint_matches_the_pre_existing_baseline(tmp_path):
    """Without the declaration the fingerprint is the old algorithm's,
    including the path-keyed embedded-file hashing."""
    path = _model(tmp_path)
    undeclared = AseEngine(FileBacked(path))
    assert undeclared.fingerprint == AseEngine(FileBacked(path)).fingerprint
    # the old branch is path-keyed: same content elsewhere still differs
    elsewhere = AseEngine(FileBacked(_model(tmp_path / "elsewhere")))
    assert undeclared.fingerprint != elsewhere.fingerprint
    # and differs from the declared branch for the same calculator
    declared = AseEngine(FileBacked(path),
                         file_parameters={"model": "potential"})
    assert undeclared.fingerprint != declared.fingerprint


def test_declaration_refusals(tmp_path):
    path = _model(tmp_path)

    # no such parameter
    with pytest.raises(ValueError, match="no such top-level parameter"):
        AseEngine(FileBacked(path),
                  file_parameters={"nonexistent": "potential"})
    # empty role
    with pytest.raises(ValueError, match="nonempty string"):
        AseEngine(FileBacked(path), file_parameters={"model": ""})
    # duplicate roles
    class TwoFiles(FileBacked):
        def __init__(self, first, second):
            super().__init__(first)
            self.parameters = {"model": str(first), "other": str(second)}
    second = _model(tmp_path, name="other.dat")
    with pytest.raises(ValueError, match="declared twice"):
        AseEngine(TwoFiles(path, second),
                  file_parameters={"model": "potential",
                                   "other": "potential"})
    # a directory is not a regular file
    with pytest.raises(ValueError, match="not an existing regular file"):
        AseEngine(FileBacked(tmp_path),  # parameters name a directory
                  file_parameters={"model": "potential"})
    # unreadable file
    unreadable = _model(tmp_path, name="unreadable.dat")
    unreadable.chmod(0)
    try:
        with pytest.raises(ValueError, match="not readable"):
            AseEngine(FileBacked(unreadable),
                      file_parameters={"model": "potential"})
    finally:
        unreadable.chmod(0o644)
    # a non-path parameter value
    with pytest.raises(TypeError, match="must be a file path"):
        AseEngine(FileBacked(path), file_parameters={"epsilon": "potential"})
    # an explicit identity never substitutes for the content check
    missing = tmp_path / "missing.dat"
    with pytest.raises(ValueError, match="not an existing regular file"):
        AseEngine(FileBacked(missing), identity="trusted",
                  file_parameters={"model": "potential"})
    # a calculator whose state cannot be identified
    class Opaque(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.parameters = {"blob": object()}  # not serializable

    with pytest.raises(ValueError, match="cannot be identified"):
        AseEngine(Opaque(), file_parameters={"blob": "potential"})


def test_wrapper_subtree_refused(tmp_path):
    path = _model(tmp_path)
    left = FileBacked(path)
    right = FileBacked(path)
    wrapper = SumCalculator([left, right])
    with pytest.raises(ValueError, match="wrapper"):
        AseEngine(wrapper, file_parameters={"model": "potential"})


def test_surrogate_forwards_the_declaration(tmp_path):
    path = _model(tmp_path)
    surrogate = AseSurrogate(FileBacked(path),
                             file_parameters={"model": "potential"})
    engine = AseEngine(FileBacked(path),
                       file_parameters={"model": "potential"})
    assert surrogate.fingerprint == f"ase-surrogate:{engine.fingerprint}"
    assert surrogate.file_resources == engine.file_resources
    assert surrogate.file_resources[0].role == "potential"
    assert len(surrogate.file_resources[0].sha256) == 64
