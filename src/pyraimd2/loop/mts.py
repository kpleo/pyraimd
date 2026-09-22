"""Experimental fixed-model, fixed-step symmetric MTS (rRESPA) NVE kernel.

One narrow entry point: :func:`run_mts`.  It integrates

    F_slow(x) = F_ref(x) - F_fast(x)

as the outer force and the fast force through the inner velocity-Verlet
blocks, with the symmetric outer half-kicks

    1. p <- p + H F_s(x)/2                                  (H = m h)
    2. m inner VV steps: p <- p + h F_fast/2; x <- x + h M^-1 p;
       evaluate F_fast(x); p <- p + h F_fast/2
    3. evaluate F_ref at the outer endpoint; F_s = F_ref - F_fast
    4. p <- p + H F_s(x)/2  — only now the boundary is committed

The outer-endpoint fast force carries into the next outer step's residual
(the model is fixed, so it is exactly the same configuration's force);
the reference endpoint label likewise becomes the next step's start.
Only complete outer steps exist: L = outer_ratio * n_outer_steps inner
steps, and a successful run makes exactly ``n_outer_steps + 1`` reference
computations and ``L + 1`` surrogate predictions.

Scope and honesty boundaries (experimental, fixed-model NVE):

- fixed cell and composition; both backends must declare
  ``force_consistent=True``, ``forces_conservative=True`` and a known
  ``energy_kind`` BEFORE the first evaluation (the existing
  cross-kind contract applies), and every returned energy/force is
  shape- and finiteness-checked against the structure;
- no thermostat, no barostat, no constraints (Atoms with constraints are
  refused rather than silently partially constrained), no adaptive step
  size, no online model updates, no resume/checkpoint support in this
  version — the run record says so, and documentation must not imply
  otherwise;
- boundary momenta are the post-final-half-kick (synchronized) ones;
  inner-block momenta are intermediate integrator state and are never
  reported as physical boundary momenta;
- failures propagate with no automatic retry and no silent fallback: a
  half-finished outer step is never committed, the caller's Atoms is
  never mutated, already-spent backend calls keep their cost records,
  and ``boundary_callback`` receives every successfully completed
  boundary (so a caller may keep the finished prefix);
- backend calls are recorded with the shared runtime.events
  task/physical_attempt semantics (operation "reference"/"inference",
  source "mts") so runtime.costs aggregates them like any other run.

ASE units throughout: positions Å, momenta ASE units, forces eV/Å,
energies eV, times in fs on the interface (internally
``inner_timestep_fs * ase.units.fs``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms, units

from pyraimd2.engines.base import (
    EnergyKind,
    EngineCapabilities,
    EngineResult,
    engine_capabilities,
)
from pyraimd2.runtime import events as _events
from pyraimd2.surrogate.base import (
    SurrogatePrediction,
    assert_compatible_energy_contract,
    surrogate_capabilities,
)

__all__ = ["MtsBoundary", "MtsError", "MtsResult", "run_mts"]

MTS_ALGORITHM_ID = "mts-respa-symmetric-v1"


class MtsError(RuntimeError):
    """A fixed-model MTS run cannot proceed (input, capability or
    backend-result contract violation); nothing is retried silently."""


@dataclass(frozen=True)
class MtsBoundary:
    """One committed outer boundary; every array is an independent copy."""

    outer_index: int
    time_fs: float
    positions_A: np.ndarray
    momenta_ase: np.ndarray   # synchronized (post final slow half-kick)
    U_ref_eV: float
    U_fast_eV: float
    K_eV: float
    H_ref_eV: float           # U_ref + K


@dataclass(frozen=True)
class MtsResult:
    """What one run_mts call returns (boundaries include the initial one)."""

    boundaries: tuple[MtsBoundary, ...]
    final_positions_A: np.ndarray
    final_momenta_ase: np.ndarray
    n_outer_steps: int
    outer_ratio: int
    inner_timestep_fs: float
    reference_calls: int
    surrogate_calls: int
    reference_time_s: float      # sum of backend-reported wall times
    inference_time_s: float
    wall_time_s: float           # kernel wall clock
    algorithm: str = MTS_ALGORITHM_ID


def _validate_inputs(atoms: Atoms, inner_timestep_fs: float,
                     outer_ratio: int, n_outer_steps: int) -> None:
    if len(atoms) == 0:
        raise MtsError("run_mts needs a non-empty structure")
    masses = np.asarray(atoms.get_masses(), dtype=float)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise MtsError("run_mts needs finite, positive masses")
    if "momenta" not in atoms.arrays:
        raise MtsError("run_mts needs existing momenta on the structure "
                       "(fixed-model NVE continues a state; it never "
                       "thermalizes one)")
    momenta = np.asarray(atoms.get_momenta(), dtype=float)
    if not np.all(np.isfinite(momenta)):
        raise MtsError("run_mts needs finite momenta")
    if not np.isfinite(inner_timestep_fs) or inner_timestep_fs <= 0:
        raise MtsError(f"inner_timestep_fs must be finite and positive, got "
                       f"{inner_timestep_fs!r}")
    if isinstance(outer_ratio, bool) or not isinstance(
            outer_ratio, (int, np.integer)) or outer_ratio < 1:
        raise MtsError(f"outer_ratio must be a positive integer, got "
                       f"{outer_ratio!r}")
    if isinstance(n_outer_steps, bool) or not isinstance(
            n_outer_steps, (int, np.integer)) or n_outer_steps < 0:
        raise MtsError(f"n_outer_steps must be a non-negative integer, got "
                       f"{n_outer_steps!r}")
    if len(atoms.constraints):
        raise MtsError("run_mts refuses Atoms with constraints in this "
                       "version: no constraint algorithm is applied and "
                       "silently ignoring them would integrate the wrong "
                       "dynamics")


def _validate_capabilities(reference: object, surrogate: object) -> None:
    if reference is surrogate:
        raise MtsError("reference and surrogate must be two distinct "
                       "backends (shared mutable compute state is not "
                       "accepted)")
    ref_caps = engine_capabilities(reference)
    sur_caps = surrogate_capabilities(surrogate)
    for side, caps in (("reference", ref_caps), ("surrogate", sur_caps)):
        if caps.force_consistent is not True:
            raise MtsError(f"run_mts needs the {side} to declare "
                           "force_consistent=True explicitly")
        if caps.forces_conservative is not True:
            raise MtsError(f"run_mts needs the {side} to declare "
                           "forces_conservative=True explicitly")
        if caps.energy_kind == EnergyKind.UNKNOWN:
            raise MtsError(f"run_mts needs the {side} to declare a known "
                           "energy_kind")
    assert_compatible_energy_contract(ref_caps, sur_caps)


def _check_result(side: str, result, n_atoms: int,
                  declared: EngineCapabilities) -> None:
    energy = getattr(result, "energy", None)
    forces = getattr(result, "forces", None)
    if energy is None or not np.isfinite(float(energy)):
        raise MtsError(f"{side} returned a non-finite energy")
    forces = np.asarray(forces, dtype=float) if forces is not None else None
    if forces is None or forces.shape != (n_atoms, 3):
        raise MtsError(f"{side} returned forces with shape "
                       f"{None if forces is None else forces.shape}, "
                       f"expected ({n_atoms}, 3)")
    if not np.all(np.isfinite(forces)):
        raise MtsError(f"{side} returned non-finite forces")
    result_kind = getattr(result, "energy_kind", EnergyKind.UNKNOWN)
    if result_kind != EnergyKind.UNKNOWN \
            and declared.energy_kind != EnergyKind.UNKNOWN \
            and result_kind != declared.energy_kind:
        raise MtsError(f"{side} result energy_kind {result_kind!r} "
                       f"conflicts with the declared "
                       f"{declared.energy_kind!r}")
    result_fc = getattr(result, "force_consistent", None)
    if result_fc is not None and result_fc != declared.force_consistent:
        raise MtsError(f"{side} result force_consistent={result_fc} "
                       f"conflicts with the declared "
                       f"{declared.force_consistent}")


class _CallLedger:
    """Task/attempt recording and per-side call accounting for one run."""

    def __init__(self, event_log, run_id: str):
        self.event_log = event_log
        self.run_id = run_id
        self.counter = 0
        self.reference_calls = 0
        self.surrogate_calls = 0
        self.reference_time_s = 0.0
        self.inference_time_s = 0.0

    def call(self, *, side: str, backend: object, method: str,
             purpose: str, invoke) -> object:
        self.counter += 1
        task_id = f"{self.run_id}-task-{self.counter}"
        operation = "reference" if side == "reference" else "inference"
        started = time.time()
        start = time.perf_counter()
        status, error = "success", None
        result = None
        try:
            with _events.physical_attempt(
                    backend, self.event_log, operation=operation,
                    request_id=task_id, purpose=purpose, source="mts",
                    method=method):
                result = invoke()
        except Exception as exc:
            status, error = "failed", repr(exc)
            raise
        finally:
            elapsed = time.perf_counter() - start
            if side == "reference":
                self.reference_calls += 1
                wall = getattr(result if status == "success" else None,
                               "wall_time_s", None)
                self.reference_time_s += float(wall) if wall is not None \
                    else elapsed
            else:
                self.surrogate_calls += 1
                wall = getattr(result if status == "success" else None,
                               "wall_time_s", None)
                self.inference_time_s += float(wall) if wall is not None \
                    else elapsed
            if self.event_log is not None:
                self.event_log.append(_events.TASK, {
                    "task_id": task_id, "attempt": 1,
                    "operation": operation, "purpose": purpose,
                    "status": status,
                    "evaluation_id": None,
                    "started_unix": started, "elapsed_s": elapsed,
                    "cpu_cores": None, "gpu": None, "queue_s": None,
                    "source": "mts", "label_id": None,
                    "cache_hit": False, "error": error})
        return result


def run_mts(atoms: Atoms, reference: object, surrogate: object, *,
            inner_timestep_fs: float, outer_ratio: int,
            n_outer_steps: int, boundary_callback=None,
            event_log=None, run_id: str | None = None) -> MtsResult:
    """One fixed-model symmetric-MTS NVE run; see the module docstring.

    ``atoms`` is copied — the caller's positions, momenta and calculator
    are never touched.  ``boundary_callback(boundary)`` receives each
    committed boundary (including the initial one) in order.
    """
    t_start = time.perf_counter()
    _validate_inputs(atoms, inner_timestep_fs, outer_ratio, n_outer_steps)
    _validate_capabilities(reference, surrogate)
    run_id = run_id or "mts"
    ledger = _CallLedger(event_log, run_id)
    ref_caps = engine_capabilities(reference)
    sur_caps = surrogate_capabilities(surrogate)

    work = atoms.copy()
    work.calc = None
    x = np.array(work.positions, dtype=float)
    p = np.array(work.get_momenta(), dtype=float)
    masses = np.asarray(work.get_masses(), dtype=float)
    inv_m = 1.0 / masses[:, None]
    n = len(work)
    h = float(inner_timestep_fs) * units.fs
    hh = int(outer_ratio) * h
    zero = int(n_outer_steps) == 0

    boundaries: list[MtsBoundary] = []

    def fresh_view(x_now: np.ndarray) -> Atoms:
        view = atoms.copy()
        view.calc = None
        view.positions = np.array(x_now, dtype=float)
        return view

    def reference_at(x_now: np.ndarray, purpose: str) -> EngineResult:
        res = ledger.call(side="reference", backend=reference,
                          method="compute", purpose=purpose,
                          invoke=lambda: reference.compute(
                              fresh_view(x_now)))
        _check_result("reference", res, n, ref_caps)
        return res

    def fast_at(x_now: np.ndarray, purpose: str) -> SurrogatePrediction:
        res = ledger.call(side="surrogate", backend=surrogate,
                          method="predict", purpose=purpose,
                          invoke=lambda: surrogate.predict(
                              fresh_view(x_now)))
        _check_result("surrogate", res, n, sur_caps)
        return res

    def commit(index: int, x_now: np.ndarray, p_now: np.ndarray,
               u_ref: float, u_fast: float) -> None:
        k_eV = float(np.sum(p_now ** 2 / (2 * masses[:, None])))
        boundary = MtsBoundary(
            outer_index=index, time_fs=index * float(outer_ratio)
            * float(inner_timestep_fs),
            positions_A=np.array(x_now, dtype=float),
            momenta_ase=np.array(p_now, dtype=float),
            U_ref_eV=float(u_ref), U_fast_eV=float(u_fast),
            K_eV=k_eV, H_ref_eV=float(u_ref) + k_eV)
        boundaries.append(boundary)
        if boundary_callback is not None:
            boundary_callback(boundary)

    if zero:
        # no-op: zero backend calls, no initial energy required
        return MtsResult(
            boundaries=(), final_positions_A=np.array(x, dtype=float),
            final_momenta_ase=np.array(p, dtype=float),
            n_outer_steps=0, outer_ratio=int(outer_ratio),
            inner_timestep_fs=float(inner_timestep_fs),
            reference_calls=0, surrogate_calls=0, reference_time_s=0.0,
            inference_time_s=0.0,
            wall_time_s=time.perf_counter() - t_start)

    ref = reference_at(x, "mts_initial")
    fast = fast_at(x, "mts_initial")
    f_slow = np.asarray(ref.forces, float) - np.asarray(fast.forces, float)
    commit(0, x, p, ref.energy, fast.energy)

    for k in range(int(n_outer_steps)):
        p = p + 0.5 * hh * f_slow                          # outer half-kick
        for _ in range(int(outer_ratio)):
            p = p + 0.5 * h * np.asarray(fast.forces, float)
            x = x + h * p * inv_m
            fast = fast_at(x, "mts_inner")
            p = p + 0.5 * h * np.asarray(fast.forces, float)
        ref = reference_at(x, "mts_outer_endpoint")
        f_slow = np.asarray(ref.forces, float) \
            - np.asarray(fast.forces, float)
        p = p + 0.5 * hh * f_slow                          # final half-kick
        commit(k + 1, x, p, ref.energy, fast.energy)

    return MtsResult(
        boundaries=tuple(boundaries),
        final_positions_A=np.array(x, dtype=float),
        final_momenta_ase=np.array(p, dtype=float),
        n_outer_steps=int(n_outer_steps), outer_ratio=int(outer_ratio),
        inner_timestep_fs=float(inner_timestep_fs),
        reference_calls=ledger.reference_calls,
        surrogate_calls=ledger.surrogate_calls,
        reference_time_s=ledger.reference_time_s,
        inference_time_s=ledger.inference_time_s,
        wall_time_s=time.perf_counter() - t_start)
