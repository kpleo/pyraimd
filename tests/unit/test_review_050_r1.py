"""R1 regression: default random streams are role-derived and independent,
and bad seeds fail at validation (0.5-batch1-fixes §R1)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import write as ase_write
from test_nvt import NVT_CONFIG

from pyraimd2.config import ConfigError, load_config
from pyraimd2.workflows import md as md_module
from pyraimd2.workflows import run_workflow


def _write(tmp_path, *, seed=42, velocity_seed=None, thermostat_seed=None,
           n_atoms=512, steps=1):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = NVT_CONFIG.format(mode="surrogate", backend="surrogate",
                             bias="bias = 0.05", dt=0.5, steps=steps,
                             temperature=300.0, friction=0.04,
                             thermostat_seed=123, k=1.0,
                             checkpoint_interval=10,
                             trajectory_interval=10, summary_interval=10)
    text = text.replace("seed = 42", f"seed = {seed}", 1)
    if velocity_seed is None:
        text = text.replace("velocity_seed = 7\n", "")
    else:
        text = text.replace("velocity_seed = 7", f"velocity_seed = {velocity_seed}")
    if thermostat_seed is None:
        text = text.replace("thermostat_seed = 123\n", "")
    else:
        text = text.replace("thermostat_seed = 123",
                            f"thermostat_seed = {thermostat_seed}")
    atoms = Atoms("H" * n_atoms,
                  positions=[[0.9, 0.9, 0.9]] * n_atoms)
    ase_write(tmp_path / "structure.extxyz", atoms, format="extxyz")
    path = tmp_path / "run.toml"
    path.write_text(text)
    return load_config(path)


def test_default_streams_are_independent(tmp_path):
    config = _write(tmp_path)  # both seeds default to run.seed=42
    from pyraimd2.workflows.md import _run_plain
    from pyraimd2.workflows.setup import load_structure

    atoms = load_structure(config)
    _run_plain(config, atoms, config.run.directory, verbose=False,
               handle_sigint=False)
    log = [json.loads(line) for line in
           (config.run.directory / "events.jsonl").read_text().splitlines()]
    start = next(e for e in log if e["type"] == "run_start")
    streams = start["streams"]
    assert streams["scheme"] == "role-derive-v1"
    assert streams["velocity_seed"] != streams["thermostat_seed"]

    # The initial velocity draws and the bath's first xi must not correlate:
    # normalized initial momenta p/sqrt(m kB T) vs the thermostat's first
    # standard_normal array.  512 atoms give 1536 iid samples; the fixed-seed
    # value is tiny, the 0.4.2 default-seed path gives correlation ~1.
    from pyraimd2.store import Store

    row0 = next(iter(Store(config.run.directory / "trajectory.db")
                     ._db.select(run_id="nvt-demo")))
    momenta = row0.toatoms().get_momenta()
    masses = row0.toatoms().get_masses()
    normalized = (momenta
                  / np.sqrt(masses[:, None] * units.kB * 300.0)).ravel()
    xi0 = np.random.default_rng(
        streams["thermostat_seed"]).standard_normal((len(momenta), 3)).ravel()
    corr = float(np.dot(normalized, xi0)
                 / (np.linalg.norm(normalized) * np.linalg.norm(xi0)))
    assert abs(corr) < 0.05


def test_explicit_same_seed_still_derives_independent_streams(tmp_path):
    config = _write(tmp_path / "same", velocity_seed=7, thermostat_seed=7)
    from pyraimd2.workflows.md import _run_plain
    from pyraimd2.workflows.setup import load_structure

    atoms = load_structure(config)
    _run_plain(config, atoms, config.run.directory, verbose=False,
               handle_sigint=False)
    log = [json.loads(line) for line in
           (config.run.directory / "events.jsonl").read_text().splitlines()]
    streams = next(e for e in log if e["type"] == "run_start")["streams"]
    assert streams["velocity_seed"] != streams["thermostat_seed"]


def test_zero_force_first_step_keeps_the_target_temperature(tmp_path):
    from pyraimd2.engines.base import EngineResult

    class ZeroReference:
        name = "zero-reference"
        fingerprint = "zero-reference:1"

        def compute(self, atoms):
            return EngineResult(0.0, np.zeros((len(atoms), 3)), None, 0.0)

    tmp_path.mkdir(parents=True, exist_ok=True)
    text = NVT_CONFIG.format(mode="reference", backend="reference", bias="",
                             dt=0.5, steps=1, temperature=300.0,
                             friction=0.04, thermostat_seed=123, k=1.0,
                             checkpoint_interval=10,
                             trajectory_interval=10, summary_interval=10)
    # The default path: both seed lines dropped, so both streams default to
    # run.seed — the 0.4.2 defect (correlated draws, +39% first-step
    # heating) lives exactly here.
    text = text.replace("thermostat_seed = 123\n", "")
    text = text.replace("velocity_seed = 7\n", "")
    atoms = Atoms("H" * 512, positions=[[0.9, 0.9, 0.9]] * 512)
    ase_write(tmp_path / "structure.extxyz", atoms, format="extxyz")
    path = tmp_path / "run.toml"
    path.write_text(text)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(md_module, "_plain_backend",
                   lambda config, run_dir, **kwargs: ZeroReference())
    try:
        config = load_config(path)
        run_workflow(config, verbose=False, handle_sigint=False)
    finally:
        monkey.undo()
    from pyraimd2.store import Store

    run_rows = sorted(Store(config.run.directory / "trajectory.db")
                      ._db.select(run_id="nvt-demo"),
                      key=lambda r: int(r.key_value_pairs["step"]))
    from pyraimd2.runtime.inspect import _temperature_K

    initial = _temperature_K(run_rows[0].toatoms())
    after = _temperature_K(run_rows[-1].toatoms())
    assert initial == pytest.approx(300.0, rel=0.05)
    assert after == pytest.approx(initial, rel=0.05)  # 0.4.2 default: ~1.39x


def test_negative_and_bad_seeds_fail_at_validation(tmp_path):
    with pytest.raises(ConfigError, match="thermostat_seed"):
        _write(tmp_path / "neg-thermostat", thermostat_seed=-1)
    with pytest.raises(ConfigError, match="velocity_seed"):
        _write(tmp_path / "neg-velocity", velocity_seed=-1)
    # A negative default comes from run.seed itself.
    with pytest.raises(ConfigError, match="seed"):
        _write(tmp_path / "neg-run", seed=-1)
