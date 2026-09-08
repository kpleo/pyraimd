"""WP07 singlepoint/relax tasks and explicit rejections (CLI included)."""

from __future__ import annotations

import numpy as np
import pytest

from pyraimd2.cli import main as cli_main
from pyraimd2.config import ConfigError, load_config
from pyraimd2.store import Store
from pyraimd2.workflows import WorkflowError, resume_workflow, run_workflow

CONFIG = """\
schema_version = 1

[run]
id = "task-demo"
directory = "runs/task-demo"
seed = 42

[task]
kind = "{kind}"
mode = "{mode}"

[structure]
file = "structure.extxyz"

[dynamics]
timestep_fs = 0.5
steps = 5

[{backend}]
backend = "harmonic-{backend}"
k = 1.0
r0 = 0.9
{bias}

[relax]
optimizer = "{optimizer}"
fmax_eV_A = {fmax}
steps = 100
"""

STRUCTURE = """\
2
H2 away from the harmonic minimum (r0 = 0.9); no velocities
H       1.300000    0.000000    0.000000
H       1.600000    0.000000    0.000000
"""


def make_config(tmp_path, *, kind="relax", mode="surrogate", optimizer="fire",
                fmax=0.05):
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = "reference" if mode == "reference" else "surrogate"
    bias = "bias = 0.05" if backend == "surrogate" else ""
    text = CONFIG.format(kind=kind, mode=mode, backend=backend, bias=bias,
                         optimizer=optimizer, fmax=fmax)
    (tmp_path / "structure.extxyz").write_text(STRUCTURE)
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def rows(run_dir, run_id="task-demo"):
    return sorted(Store(run_dir / "trajectory.db")._db.select(run_id=run_id),
                  key=lambda row: int(row.key_value_pairs["step"]))


def test_singlepoint_reference_and_surrogate(tmp_path):
    from pyraimd2.backends import backend_factory

    for mode in ("reference", "surrogate"):
        path = make_config(tmp_path / mode, kind="singlepoint", mode=mode)
        result = run_workflow(load_config(path), verbose=False)
        assert result.steps_completed == 0
        run_rows = rows(result.run_dir)
        assert len(run_rows) == 1
        row = run_rows[0]
        assert row.data["reason"] == "singlepoint"
        atoms = row.toatoms()
        # the stored label equals a direct evaluation of the same backend
        factory = backend_factory(f"harmonic-{mode}")
        direct = (factory(k=1.0, r0=0.9).compute(atoms) if mode == "reference"
                  else factory(k=1.0, r0=0.9, bias=0.05).predict(atoms))
        assert row.data["driving"]["energy"] == pytest.approx(
            float(direct.energy))
        np.testing.assert_allclose(
            np.asarray(row.data["driving"]["forces"], dtype=float),
            np.asarray(direct.forces, dtype=float))
        assert row.data["metadata"]["context"]["phase"] == "single_point"


def test_relax_converges_to_user_fmax(tmp_path):
    path = make_config(tmp_path, kind="relax", mode="surrogate", fmax=0.05)
    result = run_workflow(load_config(path), verbose=False)
    final = rows(result.run_dir)[-1]
    forces = np.asarray(final.data["driving"]["forces"], dtype=float)
    assert np.linalg.norm(forces, axis=1).max() < 0.05
    assert result.stopped_early is False


def test_relax_reference_and_bfgs(tmp_path):
    path = make_config(tmp_path, kind="relax", mode="reference",
                       optimizer="bfgs", fmax=0.02)
    result = run_workflow(load_config(path), verbose=False)
    final = rows(result.run_dir)[-1]
    forces = np.asarray(final.data["driving"]["forces"], dtype=float)
    assert np.linalg.norm(forces, axis=1).max() < 0.02


def test_relax_not_converged_reports_stopped(tmp_path):
    path = make_config(tmp_path, kind="relax", mode="surrogate", fmax=1e-9)
    result = run_workflow(load_config(path), verbose=False)
    assert result.stopped_early is True  # 100 steps are not enough for 1e-9


def test_cli_runs_all_modes_end_to_end(tmp_path, capsys):
    for kind, mode in (("singlepoint", "reference"),
                       ("singlepoint", "surrogate"),
                       ("relax", "surrogate"),
                       ("relax", "reference")):
        directory = tmp_path / f"{kind}-{mode}"
        path = make_config(directory, kind=kind, mode=mode)
        assert cli_main(["run", str(path)]) == 0
        out = capsys.readouterr().out
        assert "run directory" in out
        assert (directory / "runs" / "task-demo" / "trajectory.db").exists()


def test_adaptive_relax_rejected_at_config(tmp_path):
    path = make_config(tmp_path, kind="relax", mode="surrogate")
    text = path.read_text().replace('mode = "surrogate"', 'mode = "adaptive"')
    path.write_text(text)
    with pytest.raises(ConfigError, match="requires task.kind = 'md'"):
        load_config(path)


def test_npt_ensemble_rejected_at_config(tmp_path):
    path = make_config(tmp_path, kind="relax", mode="surrogate")
    text = path.read_text().replace("[dynamics]\n",
                                    '[dynamics]\nensemble = "npt"\n', 1)
    path.write_text(text)
    with pytest.raises(ConfigError, match="ensemble"):
        load_config(path)


def test_resume_rejected_for_singlepoint_and_relax(tmp_path):
    for kind in ("singlepoint", "relax"):
        path = make_config(tmp_path / kind, kind=kind, mode="surrogate")
        result = run_workflow(load_config(path), verbose=False)
        with pytest.raises(WorkflowError, match="task.kind"):
            resume_workflow(result.run_dir, 2, verbose=False)


POSCAR_SELECTIVE = """\
H4 with the first two atoms fixed
1.0
 5.0 0.0 0.0
 0.0 5.0 0.0
 0.0 0.0 5.0
H
4
Selective dynamics
Cartesian
 0.20 0.00 0.00  F F F
 0.24 0.00 0.00  F F F
 0.21 0.02 0.00  T T T
 0.19 -0.02 0.00  T T T
"""


def test_structure_carried_fixatoms_merges_with_config(tmp_path):
    directory = tmp_path / "poscar"
    directory.mkdir()
    (directory / "structure.extxyz").write_text(STRUCTURE)
    (directory / "POSCAR").write_text(POSCAR_SELECTIVE)
    text = CONFIG.format(kind="md", mode="surrogate", backend="surrogate",
                         bias="bias = 0.05", optimizer="fire", fmax=0.05)
    text = text.replace('file = "structure.extxyz"', 'file = "POSCAR"')
    text += '\n[constraints]\nfix_atoms_indices = [2]\n'
    path = directory / "run.toml"
    path.write_text(text)
    config = load_config(path)
    from pyraimd2.workflows.setup import load_structure

    atoms = load_structure(config)
    from ase.constraints import FixAtoms

    assert len(atoms.constraints) == 1
    constraint = atoms.constraints[0]
    assert isinstance(constraint, FixAtoms)
    assert sorted(int(i) for i in np.atleast_1d(constraint.index)) == [0, 1, 2]
