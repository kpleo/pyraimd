"""CommitteeSurrogate state-restore validation (hermetic — no torch needed:
recipe and length checks must run before any model is loaded)."""

from __future__ import annotations

import pytest

from pyraimd2.surrogate import CommitteeSurrogate


def _valid_state(committee: CommitteeSurrogate) -> dict:
    return {
        "model_specs": list(committee._model_specs),
        "n_members": committee.n_members,
        "seed": committee.seed,
        "perturbation": committee.perturbation,
        "epochs": committee.epochs,
        "lr": committee.lr,
        "force_weight": committee.force_weight,
        "trainable_filters": list(committee.trainable_filters),
        "member_state_dicts": [{} for _ in range(committee.n_members)],
        "energy_shifts": [0.0 for _ in range(committee.n_members)],
    }


def test_load_state_dict_rejects_recipe_mismatch() -> None:
    committee = CommitteeSurrogate(model="small", n_members=2, seed=1, epochs=3)
    state = _valid_state(committee)
    state["seed"] = 999
    with pytest.raises(ValueError, match="seed"):
        committee.load_state_dict(state)


def test_load_state_dict_rejects_wrong_member_count() -> None:
    committee = CommitteeSurrogate(model="small", n_members=2, seed=1, epochs=3)
    state = _valid_state(committee)
    state["member_state_dicts"] = [{}]
    with pytest.raises(ValueError, match="member_state_dicts"):
        committee.load_state_dict(state)


def test_load_state_dict_rejects_wrong_shift_length() -> None:
    committee = CommitteeSurrogate(model="small", n_members=2, seed=1, epochs=3)
    state = _valid_state(committee)
    state["energy_shifts"] = [0.0]
    with pytest.raises(ValueError, match="energy_shifts"):
        committee.load_state_dict(state)


def test_load_state_dict_rejects_filter_mismatch() -> None:
    committee = CommitteeSurrogate(model="small", n_members=2, seed=1, epochs=3)
    state = _valid_state(committee)
    state["trainable_filters"] = ["other"]
    with pytest.raises(ValueError, match="trainable_filters"):
        committee.load_state_dict(state)


def test_committee_declares_cpu_only() -> None:
    """0.4 supports the CPU committee data path only; a device field must not
    silently imply GPU availability. Single-model GPU is MaceSurrogate's job."""
    with pytest.raises(ValueError, match="cpu"):
        CommitteeSurrogate(device="cuda")
