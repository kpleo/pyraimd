"""The qe_mace_mts example skeleton: three configs that parse against the
real schema and keep the compare chain's shared-inputs contract (one
structure with momenta, one reference recipe, one physical-time span).
Everything is checked statically — no backend is ever constructed.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read as ase_read

from pyraimd2.config import load_config

EXAMPLE = Path(__file__).parents[2] / "examples" / "qe_mace_mts"
CONFIGS = ("run_reference.toml", "run_mts.toml", "run_mts_scaled.toml")


def test_skeleton_configs_parse_and_share_inputs() -> None:
    configs = [load_config(EXAMPLE / name) for name in CONFIGS]
    assert [c.task.mode for c in configs] == ["reference", "mts", "mts"]
    ids = [c.run.id for c in configs]
    assert len(set(ids)) == 3
    reference = configs[0]
    for candidate in configs[1:]:
        # one structure, one reference recipe, one timestep and span
        assert Path(candidate.structure.file) == Path(reference.structure.file)
        assert candidate.reference.options == reference.reference.options
        assert candidate.dynamics.timestep_fs == reference.dynamics.timestep_fs
        assert candidate.dynamics.steps == reference.dynamics.steps
        # candidate boundaries (every outer_ratio inner steps) land on the
        # reference's 1 fs grid
        assert candidate.dynamics.outer_ratio >= 1
        assert candidate.dynamics.steps % candidate.dynamics.outer_ratio == 0
    # the fast side is explicit and local: a placeholder model path and
    # the scaled wrapper defaulting to 1.0 with a provenance note
    assert configs[1].surrogate.name == "mace"
    assert Path(configs[1].surrogate.options["model"]).name == "user.model"
    scaled = configs[2].surrogate
    assert scaled.name == "scaled"
    assert scaled.options["scale"] == 1.0
    assert scaled.options["calibration_note"]
    assert scaled.options["base"]["name"] == "mace"


def test_skeleton_structure_carries_momenta() -> None:
    atoms = ase_read(EXAMPLE / "si_diamond_2atom.extxyz")
    assert "momenta" in atoms.arrays  # MTS never thermalizes
    assert len(atoms) == 2 and atoms.pbc.all()
