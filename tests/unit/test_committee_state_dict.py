"""CommitteeSurrogate recipe-validation contracts (RED).

One accepted defect pinned down here:

* ``load_state_dict`` must validate the full recipe echo saved by
  ``state_dict`` (seed/perturbation/epochs/lr/force_weight/trainable_filters,
  committee.py:355-374), not just ``model_specs`` and ``n_members``
  (committee.py:375-390).  A resumed run must never silently continue with
  a different surrogate than the one that produced the stored labels.

Environment: torch/mace/pyscf are NOT installed here.  The
``lazy_committee`` fixture stubs ``_ensure_loaded`` -- the only torch-touching
step on the ``load_state_dict`` path (committee.py:376-377) -- so the pure
dict-validation logic is reachable without loading real MACE models.  The
stub never satisfies an assertion itself: every recipe-mismatch test fails
today because the validation is absent.
"""

from __future__ import annotations

import pytest

from pyraimd2.surrogate.committee import CommitteeSurrogate


@pytest.fixture
def lazy_committee(monkeypatch) -> CommitteeSurrogate:
    """Fresh surrogate with the model-loading boundary stubbed.

    ``load_state_dict`` calls ``_ensure_loaded()`` first
    (committee.py:376-377), which imports mace and cannot run in this
    torch-free environment; stubbing that boundary makes the recipe
    validation reachable without loading real models.
    """
    surrogate = CommitteeSurrogate()
    monkeypatch.setattr(surrogate, "_ensure_loaded", lambda: None)
    return surrogate


def _fake_state(**overrides) -> dict:
    """A state dict with exactly the keys state_dict() emits
    (committee.py:355-374); per-member tensors are empty because no real
    models are loaded in this environment."""
    state = {
        "model_specs": ["small"] * 4,
        "n_members": 4,
        "seed": 0,
        "perturbation": 0.01,
        "epochs": 50,
        "lr": 1e-3,
        "force_weight": 10.0,
        "trainable_filters": ("readout",),
        "member_state_dicts": [],
        "energy_shifts": [0.25, 0.5],
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    "field, wrong_value",
    [
        ("seed", 1),
        ("perturbation", 0.05),
        ("epochs", 60),
        ("lr", 1e-2),
        ("force_weight", 5.0),
        ("trainable_filters", ("readout", "interactions")),
    ],
)
def test_load_state_dict_raises_on_recipe_field_mismatch(
    lazy_committee, field, wrong_value
) -> None:
    # The full recipe echo must be validated on restore: a resumed run must
    # never silently continue with a different surrogate than the one that
    # produced the stored labels, and the message must name the field.
    with pytest.raises(ValueError, match=field):
        lazy_committee.load_state_dict(_fake_state(**{field: wrong_value}))


def test_load_state_dict_matching_recipe_restores_without_error(
    lazy_committee,
) -> None:
    # Guard against over-fixing: an identical recipe must not be rejected by
    # the new validation, and the stored payload must still be restored.
    lazy_committee.load_state_dict(_fake_state())
    assert lazy_committee._energy_shifts == [0.25, 0.5]

