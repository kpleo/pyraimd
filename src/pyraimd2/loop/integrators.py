"""Integrator adapters: algorithm identity and resumable step state.

The dynamics drivers do not construct ASE integrators directly; they go
through one small adapter per algorithm so that a completed step's state —
not a formula tied to one integrator — is what checkpoints, resume and
exports consume (M1).  Two adapters exist: :class:`VelocityVerletAdapter`
(NVE) and :class:`LangevinAdapter` (NVT).  Their shared contract:

- :class:`IntegratorSpec` — algorithm id/version, ensemble, timestep,
  temperature, friction, thermostat seed and the center-of-mass
  convention.  The spec is part of the run identity: changing it means a
  new run, never a resume.
- :class:`CommittedStepState` — a finished step's step number, physical
  time, complete positions/momenta, driving source and model identity,
  plus the integrator/thermostat-RNG state needed to resume exactly.
- :class:`PendingStepState` — the mid-step crash record: the RNG state
  before and after the step's draws (never a fresh draw on resume), the
  configuration to evaluate, and the evaluation/update identities.

Each adapter creates/advances one step, exports the committed-step state,
restores it, and hands the integrator the configuration to evaluate.
Velocity Verlet completes mid-step momenta with its exact second
half-kick; Langevin momenta are already complete at the step boundary and
are never touched by that formula.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import ase
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

__all__ = [
    "CommittedStepState",
    "IntegratorSpec",
    "LangevinAdapter",
    "PendingStepState",
    "VelocityVerletAdapter",
    "derive_stream_seed",
    "state_digest",
]

# Stable per-role seed derivation ("role-derive-v1"): SeedSequence child
# seeds are deterministic across processes (unlike Python's salted hash),
# and two roles never share a generator even at the same user seed.
STREAM_SCHEME = "role-derive-v1"
_ROLE_CODES = {"velocity": 1, "thermostat": 2}

# Persisted step-summary format emitted by the current code.  0.4.x runs
# recorded no step summary at all; the previous batch persisted only the
# CommittedStepState digest (JSON); this batch persists the array digest
# plus the full boundary digest including the thermostat stream state.
DIGEST_FORMAT = "boundary-v2"


def derive_stream_seed(seed: int, role: str) -> int:
    """The effective seed for one named stream (fixed role convention).

    Different roles yield independent streams at every seed, including a
    user explicitly giving two streams the same seed.
    """
    if role not in _ROLE_CODES:
        raise ValueError(f"unknown stream role {role!r}")
    return int(np.random.SeedSequence(
        [int(seed), _ROLE_CODES[role]]).generate_state(1)[0])


@dataclass(frozen=True)
class IntegratorSpec:
    """What integrates the dynamics, exactly — the run's identity fields."""

    algorithm: str  # "velocity_verlet" | "langevin"
    ensemble: str  # "nve" | "nvt"
    timestep_fs: float
    temperature_K: float | None = None
    friction_per_fs: float | None = None
    thermostat_seed: int | None = None
    com_convention: str = "free"  # fixcm=False everywhere; FixCom unsupported
    algorithm_version: str = field(
        default_factory=lambda: f"ase-{ase.__version__}")

    def __post_init__(self) -> None:
        if self.algorithm not in ("velocity_verlet", "langevin"):
            raise ValueError(f"unsupported integrator {self.algorithm!r}")
        if self.ensemble not in ("nve", "nvt"):
            raise ValueError(f"unsupported ensemble {self.ensemble!r}")
        if self.timestep_fs <= 0:
            raise ValueError("timestep_fs must be positive")
        if self.algorithm == "langevin":
            if self.temperature_K is None or self.temperature_K < 0:
                raise ValueError("langevin needs temperature_K >= 0")
            if self.friction_per_fs is None or self.friction_per_fs <= 0:
                raise ValueError("langevin needs a positive friction_per_fs")
        elif (self.friction_per_fs is not None
              or self.thermostat_seed is not None):
            raise ValueError(
                "thermostat fields do not belong to a velocity-verlet run")

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "ensemble": self.ensemble,
            "timestep_fs": float(self.timestep_fs),
            "temperature_K": self.temperature_K,
            "friction_per_fs": self.friction_per_fs,
            "thermostat_seed": self.thermostat_seed,
            "com_convention": self.com_convention,
        }

    def identity(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True)
        return f"{self.algorithm}:{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


def state_digest(*arrays: np.ndarray) -> str:
    """Content digest binding a completed boundary's arrays (sha256[:24])."""
    hasher = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        hasher.update(array.dtype.str.encode())
        hasher.update(json.dumps(list(array.shape)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()[:24]


@dataclass
class CommittedStepState:
    """A completed step boundary: the resumable physical state, verbatim."""

    step: int
    physical_time_fs: float
    positions: np.ndarray
    momenta: np.ndarray
    driving_source: str  # "reference" | "surrogate"
    model_id: str
    spec: IntegratorSpec
    nsteps: int = 0
    thermostat_rng: dict | None = None

    def as_dict(self) -> dict:
        return {
            "step": int(self.step),
            "physical_time_fs": float(self.physical_time_fs),
            "positions": np.asarray(self.positions, dtype=float).tolist(),
            "momenta": np.asarray(self.momenta, dtype=float).tolist(),
            "driving_source": self.driving_source,
            "model_id": self.model_id,
            "nsteps": int(self.nsteps),
            "thermostat_rng": self.thermostat_rng,
            "integrator": self.spec.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommittedStepState:
        spec_data = dict(data["integrator"])
        return cls(
            step=int(data["step"]),
            physical_time_fs=float(data["physical_time_fs"]),
            positions=np.asarray(data["positions"], dtype=float),
            momenta=np.asarray(data["momenta"], dtype=float),
            driving_source=str(data["driving_source"]),
            model_id=str(data["model_id"]),
            spec=IntegratorSpec(**spec_data),
            nsteps=int(data.get("nsteps", 0)),
            thermostat_rng=data.get("thermostat_rng"),
        )

    def digest(self) -> str:
        """Full boundary identity: state, model, integrator AND the bath
        stream (a boundary that cannot name its RNG is not a complete
        stochastic state)."""
        canonical = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def legacy_digest(self) -> str:
        """The previous batch's identity (state/model/integrator WITHOUT
        the bath stream) — only for verifying records written in that
        format; never written any more."""
        canonical = json.dumps(
            {key: value for key, value in self.as_dict().items()
             if key != "thermostat_rng"},
            sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]


@dataclass
class PendingStepState:
    """Mid-step crash record: the random state around the step's draws.

    ``rng_before``/``rng_after`` pin the thermostat draws that produced the
    configuration being evaluated — a resume never draws them again.
    """

    evaluation_id: int
    positions: np.ndarray
    rng_before: dict | None
    rng_after: dict | None
    model_generation: int = 0
    label_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "evaluation_id": int(self.evaluation_id),
            "positions": np.asarray(self.positions, dtype=float).tolist(),
            "rng_before": self.rng_before,
            "rng_after": self.rng_after,
            "model_generation": int(self.model_generation),
            "label_id": self.label_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PendingStepState:
        return cls(
            evaluation_id=int(data["evaluation_id"]),
            positions=np.asarray(data["positions"], dtype=float),
            rng_before=data.get("rng_before"),
            rng_after=data.get("rng_after"),
            model_generation=int(data.get("model_generation", 0)),
            label_id=data.get("label_id"),
        )


class VelocityVerletAdapter:
    """ASE VelocityVerlet behind the integrator-state contract (NVE)."""

    def __init__(self, atoms: Atoms, spec: IntegratorSpec) -> None:
        self.spec = spec
        self.dyn = VelocityVerlet(atoms, spec.timestep_fs * units.fs)

    def step(self, forces=None):
        return self.dyn.step(forces)

    @property
    def nsteps(self) -> int:
        return self.dyn.nsteps

    @nsteps.setter
    def nsteps(self, value: int) -> None:
        self.dyn.nsteps = int(value)

    def complete_momenta(self, atoms: Atoms,
                         driving_forces: np.ndarray) -> np.ndarray:
        """The complete-step momenta of a mid-step record: velocity Verlet's
        exact second half-kick (momenta, not velocities — no mass division)."""
        return (atoms.get_momenta()
                + 0.5 * self.spec.timestep_fs * units.fs
                * np.asarray(driving_forces, dtype=float))

    def committed_state(self, step: int, atoms: Atoms, *,
                        driving_source: str, model_id: str,
                        timestep_fs: float) -> CommittedStepState:
        return CommittedStepState(
            step=step, physical_time_fs=(step + 1) * timestep_fs,
            positions=atoms.positions.copy(),
            momenta=atoms.get_momenta().copy(),
            driving_source=driving_source, model_id=model_id,
            spec=self.spec, nsteps=self.dyn.nsteps)

    def thermostat_state(self) -> dict | None:
        """Velocity Verlet has no thermostat stream."""
        return None

    def load_thermostat_state(self, state: dict | None) -> None:
        if state is not None:
            raise ValueError("a thermostat state does not belong to a "
                             "velocity-verlet run")


class LangevinAdapter:
    """ASE Langevin (NVT) with an explicit NumPy generator and fixcm=False.

    Random numbers come from one named ``numpy.random.Generator`` per run —
    never the global ``np.random`` — and its bit-generator state is the
    thermostat stream that checkpoints persist and resume restores (no
    re-seeding, no re-thermalizing).
    """

    def __init__(self, atoms: Atoms, spec: IntegratorSpec, *,
                 thermostat_rng_state: dict | None = None) -> None:
        self.spec = spec
        if thermostat_rng_state is None and spec.thermostat_seed is None:
            raise ValueError(
                "a new Langevin run needs thermostat_seed; a resume restores "
                "the persisted thermostat state instead")
        self.rng = (np.random.default_rng(spec.thermostat_seed)
                    if thermostat_rng_state is None
                    else np.random.default_rng())
        if thermostat_rng_state is not None:
            self.rng.bit_generator.state = thermostat_rng_state
        from ase.md.langevin import Langevin

        self.dyn = Langevin(
            atoms, spec.timestep_fs * units.fs,
            temperature_K=spec.temperature_K,
            friction=spec.friction_per_fs / units.fs,
            fixcm=False, rng=self.rng)

    def step(self, forces=None):
        return self.dyn.step(forces)

    @property
    def nsteps(self) -> int:
        return self.dyn.nsteps

    @nsteps.setter
    def nsteps(self, value: int) -> None:
        self.dyn.nsteps = int(value)

    def committed_state(self, step: int, atoms: Atoms, *,
                        driving_source: str, model_id: str,
                        timestep_fs: float) -> CommittedStepState:
        # Langevin momenta at the boundary are already complete — no
        # half-kick reconstruction applies to a stochastic integrator.
        return CommittedStepState(
            step=step, physical_time_fs=(step + 1) * timestep_fs,
            positions=atoms.positions.copy(),
            momenta=atoms.get_momenta().copy(),
            driving_source=driving_source, model_id=model_id,
            spec=self.spec, nsteps=self.dyn.nsteps,
            thermostat_rng=dict(self.rng.bit_generator.state))

    def thermostat_state(self) -> dict:
        return dict(self.rng.bit_generator.state)

    def load_thermostat_state(self, state: dict | None) -> None:
        if state is None:
            raise ValueError("a Langevin resume needs the persisted "
                             "thermostat state")
        self.rng.bit_generator.state = state
