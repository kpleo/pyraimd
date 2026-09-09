"""WP01 contracts: evaluation context, capability metadata, model generation.

Hermetic: analytic harmonic fakes only — no DFT, no torch, no RNG beyond the
seeded check stream.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from pyraimd2.engines import (
    AseEngine,
    CapabilityMismatchError,
    EnergyKind,
    EngineCapabilities,
    EngineResult,
    PyscfEngine,
    QeConfig,
    QeEngine,
    engine_capabilities,
)
from pyraimd2.loop import EnergeticCalculator, EnergeticRunner
from pyraimd2.runtime import EvaluationContext, EvaluationPhase, fingerprint_of
from pyraimd2.store import Store
from pyraimd2.surrogate import (
    AseSurrogate,
    CommitteeSurrogate,
    MaceSurrogate,
    SurrogateCapabilities,
    SurrogatePrediction,
    assert_compatible_energy_contract,
    surrogate_capabilities,
)


class Harmonic:
    def __init__(self, k=0.8):
        self.k = k
        self.calls = 0

    def predict(self, atoms):
        self.calls += 1
        return SurrogatePrediction(0.5 * self.k * float(np.sum(atoms.positions**2)),
                                   -self.k * atoms.positions, None,
                                   np.full(len(atoms), np.nan))


class Reference:
    name = "analytic-reference"

    def __init__(self, k=1.2):
        self.k = k
        self.attempts = 0

    def compute(self, atoms):
        self.attempts += 1
        x = atoms.positions
        return EngineResult(float(np.sum(0.5 * self.k * x**2)), -self.k * x, None, 0.0)


def setup(tmp_path, *, base=0.8, reference=None, position=0.2, **kwargs):
    atoms = Atoms("H", positions=[[position, 0, 0]])
    atoms.set_velocities([[0.1, 0, 0]])
    store = Store(tmp_path / "run.db")
    model = Harmonic(base)
    engine = Reference() if reference is None else reference
    options = {"force_budget": 0.1, "timestep_fs": 0.1, "check_probability": 1,
               "time_cap_fs": 1.0}
    options.update(kwargs)
    calc = EnergeticCalculator(model, engine, store, "run", **options)
    atoms.calc = calc
    return atoms, calc, store, model, engine


def rows(store):
    return list(store._db.select(run_id="run"))


# -- result/capability dataclass contracts ---------------------------------


def test_positional_result_construction_stays_compatible_and_unknown():
    forces = np.zeros((2, 3))
    result = EngineResult(1.0, forces, None, 0.0)  # legacy positional form
    assert result.energy_kind == EnergyKind.UNKNOWN == "unknown"
    assert result.force_consistent is None
    prediction = SurrogatePrediction(1.0, forces, None, np.full(2, np.nan))
    assert prediction.energy_kind == EnergyKind.UNKNOWN
    assert prediction.force_consistent is None
    declared = EngineResult(1.0, forces, None, 0.0, energy_kind="free_energy",
                            force_consistent=True)
    assert declared.energy_kind == EnergyKind.FREE_ENERGY == "free_energy"
    assert declared.force_consistent is True
    with pytest.raises(ValueError, match="energy_kind"):
        EngineResult(1.0, forces, None, 0.0, energy_kind="enthalpy")
    with pytest.raises(TypeError, match="force_consistent"):
        SurrogatePrediction(1.0, forces, None, np.full(2, np.nan), force_consistent=1)


def test_capabilities_default_unknown_and_validate_values():
    caps = EngineCapabilities()
    assert caps.energy_kind == EnergyKind.UNKNOWN
    assert caps.force_consistent is None and caps.forces_conservative is None
    assert caps.stress_available is False
    surrogate_caps = SurrogateCapabilities()
    assert surrogate_caps.uncertainty_available is False
    with pytest.raises(ValueError, match="energy_kind"):
        EngineCapabilities(energy_kind="bogus")
    with pytest.raises(TypeError, match="forces_conservative"):
        EngineCapabilities(forces_conservative="yes")


def test_undeclared_backends_read_as_unknown_never_supported():
    assert engine_capabilities(object()) == EngineCapabilities()
    assert surrogate_capabilities(object()) == SurrogateCapabilities()
    assert fingerprint_of(object()) is None

    class Sloppy:
        capabilities: ClassVar[dict] = {"energy_kind": "energy"}  # wrong container type

    with pytest.raises(TypeError, match="EngineCapabilities"):
        engine_capabilities(Sloppy())
    with pytest.raises(TypeError, match="SurrogateCapabilities"):
        surrogate_capabilities(Sloppy())


def test_evaluation_context_validation_and_probe_alias():
    context = EvaluationContext("run", -1, 0, "initial", 0.0, "Harmonic#g0")
    assert context.phase == EvaluationPhase.INITIAL
    probe = context.for_probe()
    assert probe.phase == EvaluationPhase.PROBE == "probe"
    assert (probe.step_id, probe.evaluation_id, probe.physical_time_fs,
            probe.model_id) == (-1, 0, 0.0, "Harmonic#g0")
    assert context.as_dict() == {
        "run_id": "run", "step_id": -1, "evaluation_id": 0,
        "phase": "initial", "physical_time_fs": 0.0, "model_id": "Harmonic#g0",
    }
    for overrides in ({"run_id": ""}, {"step_id": -2}, {"evaluation_id": -1},
                      {"phase": "bogus"},
                      {"physical_time_fs": -0.1}, {"physical_time_fs": np.nan},
                      {"model_id": ""}):
        with pytest.raises(ValueError):
            EvaluationContext(**({"run_id": "r", "step_id": 0, "evaluation_id": 0,
                                   "phase": "md_step", "physical_time_fs": 0.5,
                                   "model_id": "m"} | overrides))
    with pytest.raises(TypeError, match="step_id"):
        EvaluationContext("r", True, 0, "md_step", 0.5, "m")


# -- acceptance: one logical evaluation, one decision -----------------------


def test_energy_then_forces_is_a_single_decision(tmp_path):
    atoms, calc, store, model, engine = setup(tmp_path)  # check_probability=1
    atoms.get_forces()  # eval 0: initial reference route + deferred calibration
    atoms.positions[0, 0] = 0.21
    energy = atoms.get_potential_energy()  # eval 1: accepted + checked decision
    assert calc.n_evaluations == 2 and calc.reference_calls["check"] == 1
    snapshot = (calc.n_evaluations, calc.reference_calls.copy(),
                calc.verification.accepted_count, calc.verification.detected_count,
                engine.attempts, model.calls, len(rows(store)))
    forces = atoms.get_forces()  # same committed evaluation: replay, no new decision
    assert (calc.n_evaluations, calc.reference_calls.copy(),
            calc.verification.accepted_count, calc.verification.detected_count,
            engine.attempts, model.calls, len(rows(store))) == snapshot
    metadata = rows(store)[1].data["metadata"]
    assert metadata["checked"] is True and metadata["check_draw"] is not None
    assert metadata["verification"]["accepted_count"] == 1  # counted once
    assert metadata["context"]["evaluation_id"] == 1
    assert np.isfinite(energy) and np.isfinite(forces).all()


# -- acceptance: unchanged coordinates still advance physical time ----------


def test_unchanged_coordinates_new_step_advances_physical_time(tmp_path):
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.set_velocities([[0, 0, 0]])
    store = Store(tmp_path / "run.db")
    runner = EnergeticRunner(atoms, Harmonic(), Reference(), store, "run",
                             force_budget=0.1, timestep_fs=0.1, check_probability=0)
    summary = runner.run(3)
    assert summary.n_evaluations == 4
    contexts = [row.data["metadata"]["context"] for row in rows(store)]
    assert [c["step_id"] for c in contexts] == [-1, 0, 1, 2]
    assert [c["evaluation_id"] for c in contexts] == [0, 1, 2, 3]
    np.testing.assert_allclose([c["physical_time_fs"] for c in contexts],
                               [0.0, 0.1, 0.2, 0.3])
    assert [c["phase"] for c in contexts] == ["initial", "md_step", "md_step", "md_step"]
    # Legacy clock fields keep their exact old meaning under the fixed step.
    legacy = [row.data["metadata"]["time_fs"] for row in rows(store)]
    np.testing.assert_allclose(legacy, [c["physical_time_fs"] for c in contexts])


# -- acceptance: probes never advance physical time -------------------------


def test_probes_do_not_advance_physical_time(tmp_path):
    atoms, calc, store, _, engine = setup(tmp_path)
    atoms.get_forces()  # eval 0: 1 anchor label + 4 calibration probes
    assert engine.attempts == 5 and calc.n_evaluations == 1
    metadata = rows(store)[0].data["metadata"]
    assert metadata["context"]["physical_time_fs"] == 0.0
    calibration = metadata["new_anchor"]["calibration"]
    assert calibration["force_call_count"] == 5
    assert calibration["model_id"].endswith("#g0")
    probes = calibration["probes"]
    assert len(probes) == 4
    assert {p["phase"] for p in probes} == {"probe"}
    assert {p["evaluation_id"] for p in probes} == {0}  # parent's identity
    atoms.positions[0, 0] = 0.21
    atoms.get_forces()
    # Exactly one timestep has passed despite the five reference calls.
    assert rows(store)[1].data["metadata"]["context"]["physical_time_fs"] == pytest.approx(0.1)
    assert calc.n_evaluations == 2


# -- acceptance: a model change invalidates the same-geometry cache ---------


def test_model_change_invalidates_same_geometry_decision_cache(tmp_path):
    atoms, calc, store, model, engine = setup(tmp_path, check_probability=0)

    def update(observation):
        model.k = 0.9

    calc.on_label = update
    atoms.get_forces()  # eval 0 under generation 0; callback announces the change
    assert calc.model_generation == 1 and calc.n_evaluations == 1
    gen0_id = rows(store)[0].data["metadata"]["context"]["model_id"]
    assert gen0_id.endswith("#g0")
    atoms.get_forces()  # same geometry, new generation: a NEW decision
    assert calc.n_evaluations == 2 and len(rows(store)) == 2
    assert rows(store)[1].route == "ml"  # recalibrated, then forecast-accepted
    metadata = rows(store)[1].data["metadata"]
    assert metadata["context"]["evaluation_id"] == 1
    assert metadata["context"]["model_id"].endswith("#g1")
    assert metadata["context"]["physical_time_fs"] == pytest.approx(0.1)
    assert metadata["calibration_after_previous_label"]["anchor"]["calibration"][
        "model_id"].endswith("#g1")
    # The new proposal came from the updated model (k=0.9), not the old cache.
    assert rows(store)[1].data["surrogate"]["forces"][0][0] == pytest.approx(-0.9 * 0.2)
    assert rows(store)[0].data["surrogate"]["forces"][0][0] == pytest.approx(-0.8 * 0.2)
    # Without a further model change, re-reading the committed state replays.
    assert engine.attempts == 5
    atoms.get_forces()
    assert calc.n_evaluations == 2 and len(rows(store)) == 2
    assert engine.attempts == 5


# -- acceptance: unit/energy-convention mismatch fails before compute --------


class FreeEnergyReference(Reference):
    capabilities: ClassVar[EngineCapabilities] = EngineCapabilities(
        energy_kind="free_energy", force_consistent=True, forces_conservative=True)


class EnergyHarmonic(Harmonic):
    capabilities: ClassVar[SurrogateCapabilities] = SurrogateCapabilities(
        energy_kind="energy", force_consistent=True, forces_conservative=True)


class InconsistentHarmonic(Harmonic):
    capabilities: ClassVar[SurrogateCapabilities] = SurrogateCapabilities(
        energy_kind="energy", force_consistent=False)


class UnverifiedHarmonic(Harmonic):
    # Cross-kind partner with undeclared consistency: not a verified side.
    capabilities: ClassVar[SurrogateCapabilities] = SurrogateCapabilities(
        energy_kind="energy", force_consistent=None, forces_conservative=True)


class UnknownKindHarmonic(Harmonic):
    capabilities: ClassVar[SurrogateCapabilities] = SurrogateCapabilities(
        energy_kind="energy", force_consistent=True, forces_conservative=True)


def test_cross_kind_combination_allowed_when_both_sides_verified(tmp_path):
    # QE-metallic free_energy x MACE energy (INDEPENDENT_REVIEW §5): the
    # contract needs each side's scalar consistent with its own forces, not
    # equal kind strings. Both sides strictly consistent: allowed, and no
    # expensive call happens at construction either way.
    engine, model = FreeEnergyReference(), EnergyHarmonic()
    EnergeticCalculator(model, engine, Store(tmp_path / "a.db"), "run",
                        force_budget=0.1)
    assert engine.attempts == 0 and model.calls == 0  # no SCF, no inference
    assert issubclass(CapabilityMismatchError, ValueError)


def test_cross_kind_combination_rejected_when_a_side_is_unverified(tmp_path):
    engine, model = FreeEnergyReference(), UnverifiedHarmonic()
    with pytest.raises(CapabilityMismatchError, match="both sides verified"):
        EnergeticCalculator(model, engine, Store(tmp_path / "b.db"), "run",
                            force_budget=0.1)
    assert engine.attempts == 0 and model.calls == 0


def test_declared_force_inconsistency_fails_before_expensive_compute(tmp_path):
    engine, model = Reference(), InconsistentHarmonic()
    with pytest.raises(CapabilityMismatchError, match="force_consistent"):
        EnergeticCalculator(model, engine, Store(tmp_path / "c.db"), "run",
                            force_budget=0.1)
    assert engine.attempts == 0 and model.calls == 0


def test_matching_or_undeclared_conventions_construct_fine(tmp_path):
    class FreeEnergyHarmonic(Harmonic):
        capabilities: ClassVar[SurrogateCapabilities] = SurrogateCapabilities(
            energy_kind="free_energy", force_consistent=True,
            forces_conservative=True)

    EnergeticCalculator(FreeEnergyHarmonic(), FreeEnergyReference(),
                        Store(tmp_path / "d.db"), "run", force_budget=0.1)
    # Undeclared (unknown) cannot prove a mismatch: legacy fakes still work.
    EnergeticCalculator(Harmonic(), Reference(), Store(tmp_path / "e.db"), "run",
                        force_budget=0.1)


def test_contract_combination_modes():
    same = assert_compatible_energy_contract(
        EngineCapabilities(energy_kind="energy", force_consistent=True,
                           forces_conservative=True),
        SurrogateCapabilities(energy_kind="energy", force_consistent=True,
                              forces_conservative=True))
    cross = assert_compatible_energy_contract(
        EngineCapabilities(energy_kind="free_energy", force_consistent=True,
                           forces_conservative=True),
        SurrogateCapabilities(energy_kind="energy", force_consistent=True,
                              forces_conservative=True))
    undeclared = assert_compatible_energy_contract(
        EngineCapabilities(), SurrogateCapabilities())
    assert (same, cross, undeclared) == ("same_kind", "cross_kind", "unknown")


class MutableReference(Reference):
    def __init__(self, k=1.2):
        super().__init__(k)
        self.variant = "a"

    @property
    def fingerprint(self):
        return f"mutable-reference:{self.variant}"


def test_reference_identity_change_fails_before_expensive_compute(tmp_path):
    engine = MutableReference()
    atoms, calc, store, _, _ = setup(tmp_path, reference=engine, force_budget=0.05)
    atoms.get_forces()  # eval 0: anchor + 4 probes under identity "a"
    assert engine.attempts == 5
    assert rows(store)[0].data["metadata"]["reference_id"] == "mutable-reference:a"
    engine.variant = "b"  # settings swap mid-run
    atoms.positions[0, 0] = 0.6  # failed forecast -> reference route
    with pytest.raises(ValueError, match="reference settings identity"):
        atoms.get_forces()
    assert engine.attempts == 5  # rejected before the expensive call
    assert calc.results == {} and len(rows(store)) == 1


# -- concrete backend declarations -------------------------------------------


def test_ase_adapters_declare_capabilities_and_fingerprints():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.1, 0, 0]])
    engine = AseEngine(LennardJones())
    caps = engine.capabilities
    assert caps.energy_kind == EnergyKind.ENERGY
    assert caps.force_consistent is None  # calculator-dependent: not claimed
    assert caps.forces_conservative is None
    assert caps.stress_available is False
    result = engine.compute(atoms)
    assert result.energy_kind == EnergyKind.ENERGY and result.force_consistent is None
    consistent = AseEngine(LennardJones(), force_consistent=True, include_stress=True)
    assert consistent.capabilities.energy_kind == EnergyKind.FREE_ENERGY
    assert consistent.capabilities.force_consistent is True
    assert consistent.capabilities.stress_available is True
    assert consistent.fingerprint != engine.fingerprint
    assert engine.fingerprint.startswith("ase:lennardjones")
    surrogate = AseSurrogate(LennardJones())
    assert surrogate.capabilities.uncertainty_available is False
    assert surrogate.capabilities.energy_kind == EnergyKind.ENERGY
    prediction = surrogate.predict(atoms)
    assert prediction.energy_kind == EnergyKind.ENERGY
    assert np.isnan(prediction.uncertainty).all()


def test_pyscf_qe_identities_without_importing_backends(tmp_path):
    pbe = PyscfEngine(functional="pbe")
    assert pbe.capabilities.stress_available is False
    assert pbe.capabilities.force_consistent is True
    assert pbe.fingerprint != PyscfEngine(functional="blyp").fingerprint
    qe = QeEngine(QeConfig(pseudo_dir="/nonexistent"), run_root=tmp_path / "qe")
    assert qe.fingerprint == qe.fingerprint  # stable
    assert qe.capabilities.stress_available is True
    other = QeEngine(QeConfig(pseudo_dir="/nonexistent", ecutwfc=99.0),
                     run_root=tmp_path / "qe2")
    assert other.fingerprint != qe.fingerprint


def test_mace_and_committee_capabilities_without_torch():
    mace = MaceSurrogate(model="small")
    assert mace.capabilities.uncertainty_available is False  # honest NaN spread
    assert mace.capabilities.stress_available is True
    assert mace.capabilities.force_consistent is True
    assert "mace-mp:small" in mace.fingerprint
    committee = CommitteeSurrogate(n_members=2)
    assert committee.capabilities.stress_available is False  # not implemented
    assert committee.capabilities.uncertainty_available is True
    assert committee.capabilities.force_consistent is True
    assert committee.fingerprint.startswith("committee:2x")
