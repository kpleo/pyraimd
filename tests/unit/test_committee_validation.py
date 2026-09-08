"""CommitteeSurrogate construction contracts (RED).

Two accepted defects pinned down here:

* ``n_members=1`` (or a single-element ``model`` list) must be rejected --
  with K=1 ``sigma`` is identically zero, violating the single-model-must-
  report-NaN convention in surrogate/base.py and making the switching score
  ``s <= threshold`` constant-true (committee.py:56-57).
* ``perturbation=0.0`` must be rejected -- readout noise scale becomes zero
  (committee.py:247-249), members never differentiate, sigma == 0 again
  (committee.py:62-63).

Environment: torch/mace/pyscf are NOT installed here.  ``__init__`` is lazy
(committee.py:70-72), so construction tests run on bare numpy.
"""

from __future__ import annotations

import pytest

from pyraimd2.surrogate.committee import CommitteeSurrogate


def test_n_members_one_raises_value_error() -> None:
    # K=1 makes sigma identically zero (committee.py:223-224), faking
    # perfect agreement against the single-model-must-be-NaN convention.
    with pytest.raises(ValueError, match=r"n_members"):
        CommitteeSurrogate(n_members=1)


def test_single_element_model_list_raises_value_error() -> None:
    # An explicit one-element model list also defines K=1
    # (committee.py:65-67) — same degenerate committee, same rejection.
    with pytest.raises(ValueError, match=r"n_members"):
        CommitteeSurrogate(model=["a.model"])


def test_zero_perturbation_raises_value_error() -> None:
    # scale = perturbation * base.std() == 0 (committee.py:247-249): every
    # member readout equals the foundation, sigma == 0 in all members.
    with pytest.raises(ValueError, match=r"perturbation"):
        CommitteeSurrogate(perturbation=0.0)


def test_defaults_construct_without_error() -> None:
    # Guard against over-fixing: the default recipe (K=4, perturbation=0.01)
    # is valid and must keep constructing.
    surrogate = CommitteeSurrogate(perturbation=0.01, n_members=4)
    assert surrogate.n_members == 4
    assert surrogate.perturbation == 0.01


