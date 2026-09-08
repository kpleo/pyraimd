"""Configuration parsing, validation and path resolution (WP04).

Errors must name the dotted field and a remedy; unknown fields are rejected,
never silently absorbed; relative paths resolve against the configuration
file's directory so results do not depend on the caller's cwd.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyraimd2.config import (
    ConfigError,
    load_config,
    load_resolved_config,
    parse_config,
)
from pyraimd2.workflows.templates import HARMONIC_CONFIG, HARMONIC_STRUCTURE


def write(tmp_path: Path, text: str, name: str = "run.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def template_config(tmp_path: Path) -> Path:
    (tmp_path / "structure.extxyz").write_text(HARMONIC_STRUCTURE)
    return write(tmp_path, HARMONIC_CONFIG)


def test_template_config_parses_with_defaults(tmp_path) -> None:
    config = load_config(template_config(tmp_path))
    assert config.schema_version == 1
    assert config.run.id == "harmonic-demo"
    assert config.run.directory == (tmp_path / "runs" / "harmonic-demo").resolve()
    assert config.task.kind == "md" and config.task.mode == "adaptive"
    assert config.dynamics.ensemble == "nve"
    assert config.dynamics.timestep_fs == 0.5
    assert config.dynamics.steps == 20
    assert config.dynamics.temperature_K == 300.0
    assert config.dynamics.velocity_seed == 7
    assert config.policy.force_budget_eV_A == 0.1
    assert config.policy.probe_steps_A == (0.02, 0.04)
    assert config.verification.probability == 0.1
    assert config.verification.tilt == math.log(2.0)
    assert config.checkpoint.interval_steps == 5
    assert config.output.trajectory_interval_steps == 1
    assert config.reference.name == "harmonic-reference"
    assert config.reference.options["k"] == 1.0
    assert config.surrogate.name == "harmonic-surrogate"
    assert config.surrogate.options["bias"] == 0.05


def test_seed_defaults_derive_from_run_seed(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("velocity_seed = 7\n", "")
    text = text.replace("seed = 19\n", "")
    config = load_config(write(tmp_path, text))
    assert config.dynamics.velocity_seed == 42
    assert config.verification.seed == 42


def test_relative_paths_resolve_against_config_directory(tmp_path, monkeypatch) -> None:
    nested = tmp_path / "project" / "cfg"
    nested.mkdir(parents=True)
    config_path = template_config(nested)
    monkeypatch.chdir("/")  # a different cwd must not change resolution
    config = load_config(config_path)
    assert config.structure.file == (nested / "structure.extxyz").resolve()
    assert config.run.directory == (nested / "runs" / "harmonic-demo").resolve()


def test_unknown_field_is_rejected_with_suggestion(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("timestep_fs", "timestep")
    with pytest.raises(ConfigError, match=r"unknown field 'dynamics\.timestep'.*timestep_fs"):
        load_config(write(tmp_path, text))


def test_unknown_section_is_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG + "\n[optimizer]\nsteps = 5\n"
    with pytest.raises(ConfigError, match="unknown section 'optimizer'"):
        load_config(write(tmp_path, text))


def test_unknown_policy_field_is_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("force_budget_eV_A", "force_budget")
    with pytest.raises(ConfigError, match=r"unknown field 'policy\.force_budget'"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("version", ["2", "0", "\"one\""])
def test_unsupported_schema_version_is_rejected(tmp_path, version) -> None:
    text = HARMONIC_CONFIG.replace("schema_version = 1", f"schema_version = {version}")
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(write(tmp_path, text))


def test_negative_timestep_is_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("timestep_fs = 0.5", "timestep_fs = -0.5")
    with pytest.raises(ConfigError, match=r"dynamics\.timestep_fs must be > 0"):
        load_config(write(tmp_path, text))


def test_zero_steps_are_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("steps = 20", "steps = 0")
    with pytest.raises(ConfigError, match=r"dynamics\.steps must be >= 1"):
        load_config(write(tmp_path, text))


def test_probe_steps_must_increase(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("probe_steps_A = [0.02, 0.04]",
                                   "probe_steps_A = [0.04, 0.02]")
    with pytest.raises(ConfigError, match=r"policy\.probe_steps_A"):
        load_config(write(tmp_path, text))


def test_transverse_cap_range(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("transverse_cap = 0.1", "transverse_cap = 1.5")
    with pytest.raises(ConfigError, match=r"policy\.transverse_cap must be <= 1"):
        load_config(write(tmp_path, text))


def test_zero_check_probability_with_parameters_is_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("probability = 0.1", "probability = 0.0")
    with pytest.raises(ConfigError, match=r"verification\.(failure_probability|tilt)"
                                         r".*probability is 0"):
        load_config(write(tmp_path, text))


def test_zero_check_probability_alone_disables_checks(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("probability = 0.1", "probability = 0.0")
    text = text.replace("failure_probability = 0.05\n", "")
    text = text.replace("tilt = 0.6931471805599453         # ln 2\n", "")
    config = load_config(write(tmp_path, text))
    assert config.verification.probability == 0.0


def test_failure_probability_range(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("failure_probability = 0.05",
                                   "failure_probability = 1.5")
    with pytest.raises(ConfigError, match=r"verification\.failure_probability"):
        load_config(write(tmp_path, text))


def test_checkpoint_interval_must_be_positive(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("interval_steps = 5", "interval_steps = 0")
    with pytest.raises(ConfigError, match=r"checkpoint\.interval_steps"):
        load_config(write(tmp_path, text))


def test_keep_generations_other_than_two_is_explicit(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace("keep_generations = 2", "keep_generations = 5")
    with pytest.raises(ConfigError, match=r"checkpoint\.keep_generations.*exactly 2"):
        load_config(write(tmp_path, text))


def test_reference_mode_rejects_surrogate_section(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace('mode = "adaptive"', 'mode = "reference"')
    with pytest.raises(ConfigError, match=r"task\.mode 'reference' does not use \[surrogate\]"):
        load_config(write(tmp_path, text))


def test_adaptive_requires_both_backends(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace('[surrogate]\nbackend = "harmonic-surrogate"',
                                   '[surrogate]\n# backend removed')
    with pytest.raises(ConfigError, match=r"surrogate\.backend|requires \[surrogate\]"):
        load_config(write(tmp_path, text))


def test_policy_rejected_for_plain_modes(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace('mode = "adaptive"', 'mode = "surrogate"')
    # drop the whole [reference] block so the surrogate-only config is
    # otherwise valid; the [policy] section must then be the problem
    start = text.index("[reference]")
    end = text.index("[surrogate]")
    text = text[:start] + text[end:]
    with pytest.raises(ConfigError, match=r"task\.mode 'surrogate' does not use \[policy\]"):
        load_config(write(tmp_path, text))


def test_unknown_ensemble_is_rejected(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace('ensemble = "nve"', 'ensemble = "nvt"')
    with pytest.raises(ConfigError, match=r"dynamics\.ensemble must be one of"):
        load_config(write(tmp_path, text))


def test_run_id_rejects_path_separators(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace('id = "harmonic-demo"', 'id = "a/b"')
    with pytest.raises(ConfigError, match=r"run\.id"):
        load_config(write(tmp_path, text))


def test_invalid_toml_reports_the_file(tmp_path) -> None:
    path = write(tmp_path, "this is = = not toml\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_missing_config_file_reports_remedy(tmp_path) -> None:
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_config(tmp_path / "nope.toml")


def test_backend_option_paths_resolve(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace(
        '[reference]\nbackend = "harmonic-reference"',
        '[reference]\nbackend = "qe"\npseudo_dir = "pseudos"\n'
        'pseudos = { H = "H.upf" }')
    config = load_config(write(tmp_path, text))
    assert config.reference.options["pseudo_dir"] == str((tmp_path / "pseudos").resolve())
    assert config.reference.options["pseudos"]["H"] == str(
        (tmp_path / "pseudos" / "H.upf").resolve())


def test_mace_model_name_is_not_resolved_as_path(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace(
        '[surrogate]\nbackend = "harmonic-surrogate"',
        '[surrogate]\nbackend = "mace"\nmodel = "small"')
    config = load_config(write(tmp_path, text))
    assert config.surrogate.options["model"] == "small"  # a name, not a path


def test_mace_model_path_is_resolved(tmp_path) -> None:
    text = HARMONIC_CONFIG.replace(
        '[surrogate]\nbackend = "harmonic-surrogate"',
        '[surrogate]\nbackend = "mace"\nmodel = "models/mace.model"')
    config = load_config(write(tmp_path, text))
    assert config.surrogate.options["model"] == str(
        (tmp_path / "models" / "mace.model").resolve())


def test_resolved_config_roundtrip(tmp_path) -> None:
    config = load_config(template_config(tmp_path))
    run_dir = config.run.directory
    run_dir.mkdir(parents=True)
    import json

    (run_dir / "resolved_config.json").write_text(
        json.dumps(config.resolved_dict(), indent=2))
    restored = load_resolved_config(run_dir)
    assert restored.run.id == config.run.id
    assert restored.run.directory == config.run.directory
    assert restored.dynamics == config.dynamics
    assert restored.policy == config.policy
    assert restored.verification == config.verification
    assert restored.checkpoint == config.checkpoint
    assert restored.output == config.output
    assert restored.reference == config.reference
    assert restored.surrogate == config.surrogate
    assert restored.resolved_dict() == config.resolved_dict()


def test_load_resolved_config_requires_a_run_directory(tmp_path) -> None:
    with pytest.raises(ConfigError, match="resolved_config.json"):
        load_resolved_config(tmp_path)


def test_parse_config_rejects_non_table() -> None:
    with pytest.raises(ConfigError):
        parse_config([1, 2, 3], base_dir=Path("/tmp"))
