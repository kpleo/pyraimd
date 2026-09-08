"""Store roundtrip: positions, momenta, labels, and routes (hermetic)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from conftest import CLUSTER_MOMENTA, CLUSTER_POSITIONS

from pyraimd2.engines.base import EngineResult
from pyraimd2.store import Store
from pyraimd2.surrogate.base import SurrogatePrediction


def _surrogate_prediction(n: int, offset: float = 0.0) -> SurrogatePrediction:
    return SurrogatePrediction(
        energy=-10.0 + offset,
        forces=np.full((n, 3), 0.111) + offset,
        stress=None,
        uncertainty=np.full(n, np.nan),
    )


def _engine_result(n: int, offset: float = 0.0) -> EngineResult:
    return EngineResult(
        energy=-20.0 + offset,
        forces=np.full((n, 3), 0.222) + offset,
        stress=None,
        wall_time_s=1.5 + offset,
    )


def _shifted(cluster: Atoms, dx: float) -> Atoms:
    atoms = cluster.copy()
    atoms.positions += dx
    atoms.set_momenta(cluster.get_momenta() + 0.001 * dx)
    return atoms


def test_roundtrip_positions_and_momenta(tmp_path, cluster) -> None:
    store = Store(tmp_path / "run.db")
    atoms_v1 = _shifted(cluster, 0.1)
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(len(cluster)))
    store.append("r1", 0, atoms_v1, "dft", surrogate=_surrogate_prediction(len(cluster), 0.01),
                 engine=_engine_result(len(cluster)), reason="scheduled dft")

    restored, step = store.latest_state("r1")
    assert step == 0
    np.testing.assert_array_equal(restored.get_positions(), atoms_v1.get_positions())
    np.testing.assert_array_equal(restored.get_momenta(), atoms_v1.get_momenta())


def test_latest_state_picks_last_step_and_isolates_runs(tmp_path, cluster) -> None:
    store = Store(tmp_path / "run.db")
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(len(cluster)))
    store.append("r1", 0, _shifted(cluster, 0.1), "ml", surrogate=_surrogate_prediction(len(cluster)))
    store.append("r1", 1, _shifted(cluster, 0.2), "ml", surrogate=_surrogate_prediction(len(cluster)))
    store.append("r2", -1, _shifted(cluster, 9.9), "ml", surrogate=_surrogate_prediction(len(cluster)))

    restored, step = store.latest_state("r1")
    assert step == 1
    np.testing.assert_array_equal(restored.get_positions(), _shifted(cluster, 0.2).get_positions())

    with pytest.raises(RuntimeError, match="r3"):
        store.latest_state("r3")


def test_latest_state_refuses_pre_md_only_run(tmp_path, cluster) -> None:
    store = Store(tmp_path / "run.db")
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(len(cluster)))
    with pytest.raises(RuntimeError, match="step -1"):
        store.latest_state("r1")


def test_iter_labels_yields_only_dft_rows_with_engine_results(tmp_path, cluster) -> None:
    store = Store(tmp_path / "run.db")
    n = len(cluster)
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(n))
    store.append("r1", 0, _shifted(cluster, 0.1), "dft",
                 surrogate=_surrogate_prediction(n, 0.01), engine=_engine_result(n, 0.0),
                 reason="step 0")
    store.append("r1", 1, _shifted(cluster, 0.2), "ml", surrogate=_surrogate_prediction(n, 0.2))
    store.append("r1", 2, _shifted(cluster, 0.3), "dft",
                 surrogate=_surrogate_prediction(n, 0.31), engine=_engine_result(n, 2.0),
                 reason="step 2")

    labels = list(store.iter_labels("r1"))
    assert len(labels) == 2
    atoms0, result0 = labels[0]
    np.testing.assert_array_equal(atoms0.get_positions(), _shifted(cluster, 0.1).get_positions())
    np.testing.assert_array_equal(result0.forces, np.full((n, 3), 0.222))
    assert result0.energy == -20.0
    assert result0.wall_time_s == 1.5
    assert result0.stress is None
    _, result2 = labels[1]
    assert result2.energy == -18.0  # insertion order == step order


def test_driving_label_returns_the_forces_that_drove_the_md(tmp_path, cluster) -> None:
    store = Store(tmp_path / "run.db")
    n = len(cluster)
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(n))
    store.append("r1", 0, _shifted(cluster, 0.1), "dft",
                 surrogate=_surrogate_prediction(n, 0.01), engine=_engine_result(n))

    energy_ml, forces_ml = store.driving_label("r1", -1)
    assert energy_ml == -10.0
    np.testing.assert_array_equal(forces_ml, np.full((n, 3), 0.111))

    energy_dft, forces_dft = store.driving_label("r1", 0)
    assert energy_dft == -20.0
    np.testing.assert_array_equal(forces_dft, np.full((n, 3), 0.222))

    with pytest.raises(RuntimeError, match="step 7"):
        store.driving_label("r1", 7)


def test_uncertainty_nan_survives_roundtrip(tmp_path, cluster) -> None:
    import ase.db

    store = Store(tmp_path / "run.db")
    n = len(cluster)
    store.append("r1", -1, cluster, "ml", surrogate=_surrogate_prediction(n))
    # Read back through a fresh public ase.db connection (tests may inspect
    # raw rows; the Store API itself stays minimal).
    row = next(iter(ase.db.connect(str(tmp_path / "run.db")).select(run_id="r1")))
    assert np.isnan(row.data["surrogate"]["uncertainty"]).all()


def test_stored_positions_match_fixture_exactly(tmp_path, cluster) -> None:
    """The db roundtrip must be bit-exact (restart equality depends on it)."""
    store = Store(tmp_path / "run.db")
    store.append("r1", 0, cluster, "ml", surrogate=_surrogate_prediction(len(cluster)))
    restored, step = store.latest_state("r1")
    assert step == 0
    np.testing.assert_array_equal(restored.get_positions(), CLUSTER_POSITIONS)
    np.testing.assert_array_equal(restored.get_momenta(), CLUSTER_MOMENTA)


def test_iter_observations_replays_switch_stream(tmp_path, cluster) -> None:
    """(step, s, e) rebuilt from stored payloads must equal what the switch
    observed live: s = max per-atom spread, e = max per-atom |ΔF|."""
    store = Store(tmp_path / "run.db")
    n = len(cluster)
    cases = [(0.05, 0.10, 0.12), (0.07, 0.20, 0.26)]
    for step, (spread, f_sur, f_eng) in enumerate(cases):
        sur = SurrogatePrediction(
            energy=1.0,
            forces=np.full((n, 3), f_sur),
            stress=None,
            uncertainty=np.full(n, spread),
        )
        eng = EngineResult(
            energy=0.0,
            forces=np.full((n, 3), f_eng),
            stress=None,
            wall_time_s=0.1,
        )
        store.append("r1", step, cluster, "dft", surrogate=sur, engine=eng)

    obs = list(store.iter_observations("r1"))
    assert [o[0] for o in obs] == [0, 1]
    for (_, s, e), (spread, f_sur, f_eng) in zip(obs, cases):
        assert s == pytest.approx(spread)
        assert e == pytest.approx(np.sqrt(3.0) * abs(f_eng - f_sur))
