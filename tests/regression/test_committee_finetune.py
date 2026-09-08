"""Real CommitteeSurrogate fine-tune on PySCF labels.

Marked ``slow`` — deselected by default, run with ``pytest --runslow``.
Loads MACE-MP-0 and runs 40 PySCF single points (~1 min on CPU).

Labels: 40 frames of a short MACE-driven NVE trajectory (the regime the
committee fine-tunes on in production), split by interleaving — every 4th
frame is held out — so the split is insensitive to drift along the
trajectory.  Gate: >= 30% held-out force-MAE reduction (spec §5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import molecule
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from pyraimd2.engines import PyscfEngine
from pyraimd2.surrogate import CommitteeSurrogate

pytestmark = pytest.mark.slow

N_TRAIN = 30
N_HELDOUT = 10
IMPROVEMENT_GATE = 0.30
SEED = 20250819


def _trajectory_frames() -> list:
    """40 frames of a 300 K NVE H2O trajectory driven by frozen MACE-MP-0."""
    from mace.calculators import mace_mp  # local import: torch is heavy

    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10  # same distorted start as the examples
    thermalize_momenta(atoms, 300.0, rng=np.random.default_rng(SEED))
    atoms.calc = mace_mp(model="small", device="cpu", default_dtype="float64")
    frames = []
    dyn = VelocityVerlet(atoms, 0.5 * units.fs)
    dyn.attach(lambda: frames.append(atoms.copy()), interval=2)
    dyn.run(2 * (N_TRAIN + N_HELDOUT))
    return frames[: N_TRAIN + N_HELDOUT]


def _force_mae(committee: CommitteeSurrogate, pairs) -> float:
    """Mean per-atom |F_pred − F_true| (eV/Å) over the given frames."""
    errors = []
    for atoms, label in pairs:
        prediction = committee.predict(atoms)
        errors.append(np.linalg.norm(prediction.forces - label.forces, axis=1).mean())
    return float(np.mean(errors))


def test_committee_finetune_improves_held_out_forces() -> None:
    frames = _trajectory_frames()
    engine = PyscfEngine()
    pairs = [(atoms, engine.compute(atoms)) for atoms in frames]
    train = [p for i, p in enumerate(pairs) if i % 4 != 0][:N_TRAIN]
    heldout = [p for i, p in enumerate(pairs) if i % 4 == 0][:N_HELDOUT]
    assert len(train) == N_TRAIN and len(heldout) == N_HELDOUT

    committee = CommitteeSurrogate(n_members=4, seed=0)

    # Cold start: members differ by the seeded load-time perturbation, so
    # the spread is honestly nonzero (spec §1 as revised after smoke
    # 7615191: identical members gave sigma == 0, exploding the conformal
    # ratio r = e/(s+delta) and pinning qhat for a whole window).
    cold = committee.predict(heldout[0][0])
    assert np.isfinite(cold.uncertainty).all()
    assert np.all(cold.uncertainty > 0.0)
    # The load-time perturbation is seeded: an identical committee reproduces
    # the exact same spread (restart/determinism guard).
    cold_twin = CommitteeSurrogate(n_members=4, seed=0).predict(heldout[0][0])
    np.testing.assert_array_equal(cold.uncertainty, cold_twin.uncertainty)

    mae_before = _force_mae(committee, heldout)
    report = committee.finetune(train)

    assert report.n_labels == N_TRAIN
    assert report.n_epochs == 50
    assert len(report.member_losses) == 4
    assert report.final_loss < report.initial_loss  # training descended

    mae_after = _force_mae(committee, heldout)
    # Members now differ: the spread is honestly nonzero.
    warm = committee.predict(heldout[0][0])
    assert np.all(warm.uncertainty > 0.0)

    assert mae_after <= (1.0 - IMPROVEMENT_GATE) * mae_before, (
        f"held-out force MAE {mae_before:.4f} -> {mae_after:.4f} eV/A: "
        f"improvement {1 - mae_after / mae_before:.1%} < {IMPROVEMENT_GATE:.0%}"
    )


# -- mixed-backbone committee + RMS spread (2026-08-21, post-7615281 fixes) ----

_MACE_CACHE = Path.home() / ".cache" / "mace"
_B3 = _MACE_CACHE / "macemp0b3mediummodel"
_MPA = _MACE_CACHE / "macempa0mediummodel"


@pytest.mark.skipif(
    not (_B3.exists() and _MPA.exists()),
    reason="cached 0b3/MPA-0 foundation files not present",
)
def test_mixed_backbone_spread_is_exact_rms() -> None:
    """Two-member mixed committee: population RMS spread == |F0 - F1| / 2.

    Guards the spread estimator (std of deviation *norms* is identically 0 at
    K=2 — the bug behind the smoke-7615281 overconfidence) and the
    mixed-backbone loading path end to end.
    """
    import torch  # local import: heavy
    from mace.tools import torch_geometric

    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10
    mixed = CommitteeSurrogate(model=[str(_B3), str(_MPA)], seed=0)
    pred = mixed.predict(atoms)
    assert np.isfinite(pred.uncertainty).all()
    # Cross-backbone disagreement on a distorted H2O is real, not noise-scale.
    assert pred.uncertainty.max() > 0.01

    batch = torch_geometric.Batch.from_data_list([mixed._to_atomic_data(atoms)])
    member_forces = [
        mixed._forward(m, batch, training=False)[0]["forces"].detach().numpy()
        for m in mixed._models
    ]
    ref = np.linalg.norm(member_forces[0] - member_forces[1], axis=1) / 2.0
    f_conv = mixed._calc.energy_units_to_eV / mixed._calc.length_units_to_A
    np.testing.assert_allclose(pred.uncertainty, ref * f_conv, rtol=1e-10)


@pytest.mark.skipif(not _B3.exists(), reason="cached 0b3 foundation not present")
def test_committee_state_dict_roundtrip() -> None:
    """state_dict -> fresh committee -> load_state_dict reproduces predictions
    to machine precision (resume safety for preempted campaigns), and a
    member-count mismatch raises instead of silently continuing."""
    atoms = molecule("H2O")
    atoms.positions[1, 0] += 0.10
    src = CommitteeSurrogate(model=str(_B3), n_members=2, seed=3)
    before = src.predict(atoms)  # member 1 carries the seeded perturbation
    state = src.state_dict()

    dst = CommitteeSurrogate(model=str(_B3), n_members=2, seed=3)
    dst.load_state_dict(state)
    after = dst.predict(atoms)
    # weights restore exactly; the forward differs only by float reduction
    # ordering (~1e-17 on forces here).
    np.testing.assert_allclose(before.forces, after.forces, rtol=0, atol=1e-12)
    np.testing.assert_allclose(before.uncertainty, after.uncertainty, rtol=0, atol=1e-12)
    assert before.energy == pytest.approx(after.energy, rel=0, abs=1e-10)

    mismatched = CommitteeSurrogate(model=str(_B3), n_members=3, seed=3)
    with pytest.raises(ValueError, match="checkpoint"):
        mismatched.load_state_dict(state)
