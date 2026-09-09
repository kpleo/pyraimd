"""Quantum ESPRESSO through ASE's Espresso calculator: the ASE-native QE path.

Programs ASE can drive should be reached through a small factory rather than
another handwritten subprocess layer. This module builds an
``ase.calculators.espresso.Espresso`` calculator from the same :class:`QeConfig`
the handwritten :class:`~pyraimd2.engines.qe_engine.QeEngine` uses, so the two
QE paths share one scientific-settings source (XC, dispersion, cutoffs,
k-points, smearing, pseudopotentials) and — as far as QE can express them —
the same electronic state: collinear magnetic moments are mapped by ASE's own
writer (``nspin = 2``, one species per (element, magmom)), a nonzero net
charge is injected as ``tot_charge`` per evaluation, and noncollinear moments
are rejected before launch on both paths.

Both adapters also clear the same success bar before a label exists
(:func:`~pyraimd2.engines.qe_engine.check_qe_run_text` on the run's
``espresso.pwo``): completion marker, no non-convergence or QE error banner,
and a complete finite numeric block with an ordered 1..nat force block and a
full stress block. A truncated or non-converged run is rejected here exactly
as on the handwritten path — parsing "some energy and forces" is never
enough.

Result conventions are identical to the handwritten path by construction —
energy in eV, forces in eV/Å, stress as a 6-component ASE Voigt vector with
the ASE (compression-negative) sign — and are verified against each other on
a real pw.x fixture in tests/unit/test_ase_qe.py. ASE's espresso reader
converts Rydbergs with CODATA-2006 constants while the handwritten parser
uses the current ``ase.units`` defaults, so energies can differ at the 1e-7
relative level; that scale difference is documented, not silently absorbed.

Directories follow the same safe allocation as the handwritten path
(:func:`~pyraimd2.engines.qe_engine.allocate_run_dir`): a fresh instance or
process continues the on-disk numbering instead of colliding with earlier
runs. Retries are bounded by ``max_retries`` with the same classification
(deterministic non-convergence/input errors are never retried); a failed
chained-density start falls back to one atomic start. ``timeout_s`` cannot
be enforced through ASE's FileIO layer, so a non-default value is rejected
at construction instead of being silently ignored.

Interface note: ``EspressoProfile(command, pseudo_dir)`` and
``Espresso(profile=..., directory=..., input_data=..., pseudopotentials=...,
kpts=...)`` follow the ASE version locked in uv.lock (3.29), verified by
tests — not copied from older blog examples. ASE takes the launch command as
a single string for its own FileIO machinery; our authoritative runner
(:class:`QeEngine`) keeps the command as an argv array and never goes through
a shell.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.base import (
    EnergyKind,
    EngineCapabilities,
    EngineError,
    EngineResult,
)
from pyraimd2.engines.qe_engine import (
    QE_PREFIX,
    DensitySource,
    QeConfig,
    QeEngineError,
    _initial_magmoms,
    _settings_digest,
    _stage_density_into,
    _total_charge,
    allocate_run_dir,
    check_qe_run_text,
    classify_qe_failure_text,
    load_density_source,
    normalize_config_paths,
    recipe_name,
    write_density_manifest,
)

_TIMEOUT_DEFAULT = QeConfig.timeout_s  # dataclass field default


def espresso_input_data(config: QeConfig) -> dict:
    """Map a QeConfig onto ASE Espresso ``input_data`` namelists.

    This is the same physics mapping as
    :func:`~pyraimd2.engines.qe_engine.write_qe_input`: both paths must
    produce equivalent pw.x inputs for the same config.
    """
    control = {
        "calculation": "scf",
        "prefix": QE_PREFIX,
        "pseudo_dir": config.pseudo_dir,
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
    }
    system: dict = {
        "ibrav": 0,
        "ecutwfc": config.ecutwfc,
        "ecutrho": config.ecutrho,
        "input_dft": config.xc,
    }
    if config.dispersion is not None:
        system["vdw_corr"] = config.dispersion
    if config.nbnd is not None:
        system["nbnd"] = config.nbnd
    if config.metallic:
        system.update(
            occupations="smearing",
            smearing=config.smearing,
            degauss=config.degauss,
        )
    electrons: dict = {
        "conv_thr": config.conv_thr,
        "mixing_beta": config.mixing_beta,
        "mixing_mode": config.mixing_mode,
        "mixing_ndim": config.mixing_ndim,
        "diago_david_ndim": config.diago_david_ndim,
        "electron_maxstep": config.electron_maxstep,
    }
    if config.diago_full_acc:
        electrons["diago_full_acc"] = True
    if config.startpot_file:
        electrons["startingpot"] = "file"
    return {"control": control, "system": system, "electrons": electrons}


def make_espresso_calculator(config: QeConfig, *, directory: str | Path,
                             command: str | None = None) -> Espresso:
    """Construct ASE's Espresso calculator for ``config``.

    ``command`` overrides the launch string (ASE's FileIO layer takes one
    string); the default joins the config's argv. The calculator writes its
    input as ``<directory>/espresso.pwi`` when ASE calculates.
    """
    profile = EspressoProfile(
        command=command if command is not None else shlex.join(config.pw_cmd),
        pseudo_dir=config.pseudo_dir,
    )
    return Espresso(
        profile=profile,
        directory=str(directory),
        input_data=espresso_input_data(config),
        pseudopotentials=dict(config.pseudos),
        kpts=config.kpts,
    )


class AseQeEngine(AseEngine):
    """Engine-protocol QE reference backed by ASE's Espresso calculator.

    Same Engine contract as :class:`QeEngine`: each compute runs in its own
    safely allocated ``run_root/<label>-NNNNNN`` directory, capabilities
    state the energy convention honestly (smearing → free energy) and the
    per-label metadata matches them (``force_consistent=True`` — ASE's
    espresso reader reports the same scalar as energy and free energy), and
    the fingerprint covers the full reference settings including
    pseudopotential content hashes, under the ``ase-qe-...`` name so labels
    record which path produced them.
    """

    def __init__(self, config: QeConfig, run_root: str | Path, *,
                 command: str | None = None,
                 event_log: object | None = None) -> None:
        config = normalize_config_paths(config)
        if config.timeout_s != _TIMEOUT_DEFAULT:
            raise EngineError(
                "AseQeEngine cannot enforce timeout_s through ASE's FileIO "
                "layer; use the handwritten QeEngine for bounded runs "
                f"(got timeout_s={config.timeout_s})"
            )
        self.config = config
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._command = command
        self._event_log = event_log
        self._io_counter = 0
        self._request_counter = 0
        self._last_density_dir: Path | None = None
        self.last_attempt_records: list[dict] = []
        self.last_density_decision: dict | None = None
        calculator = make_espresso_calculator(
            config, directory=self.run_root / "pending", command=command
        )
        super().__init__(calculator, force_consistent=True, include_stress=True)

    @property
    def name(self) -> str:
        return f"ase-{recipe_name(self.config)}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            energy_kind=EnergyKind.FREE_ENERGY if self.config.metallic else EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
        )

    @property
    def fingerprint(self) -> str:
        digest = _settings_digest(self.config, path_kind="ase-espresso")
        return f"{self.name}:{digest}"

    def fresh_directory(self, label: str | None = None) -> Path:
        """The next unique per-call directory, claimed atomically (shared
        allocation with the handwritten path: new instances continue the
        on-disk numbering instead of colliding)."""
        base = "eval" if label is None else str(label).replace("/", "_")
        return allocate_run_dir(self.run_root, base)

    # -- per-evaluation input state ----------------------------------------

    def _apply_electronic_state(self, atoms: Atoms, *, startpot: bool) -> None:
        """Set the per-evaluation input keys on the owned calculator.

        Collinear moments are mapped by ASE's writer itself; the net charge
        and the warm-start toggle are per-evaluation state injected here
        (ASE 3.29 keeps ``parameters['input_data']`` across writes).
        """
        _initial_magmoms(atoms)  # rejects noncollinear states before launch
        input_data = self.calculator.parameters["input_data"]
        system = input_data.setdefault("system", {})
        charge = _total_charge(atoms)
        if charge != 0.0:
            system["tot_charge"] = charge
        else:
            system.pop("tot_charge", None)
        electrons = input_data.setdefault("electrons", {})
        if startpot:
            electrons["startingpot"] = "file"
        else:
            electrons.pop("startingpot", None)

    # -- execution ----------------------------------------------------------

    def _emit_attempt(self, record: dict, *, request_id: str) -> None:
        """One event per real process launch; the record's status is
        terminal (success/failed/killed/post_processing_failed) by then.
        Input-write failures and a missing executable never launched a
        process, so they emit nothing."""
        if self._event_log is None:
            return
        self._event_log.append(
            "attempt",
            {
                "record": "physical_attempt",
                "operation": "reference",
                "purpose": "scf",
                "request_id": request_id,
                "attempt": record["attempt"],
                "status": record["status"],
                "failure_kind": record.get("failure_kind"),
                "started_unix": record["started_unix"],
                "elapsed_s": record.get("wall_time_s"),
                "process_elapsed_s": record.get("process_s"),
                "returncode": record.get("returncode"),
                "directory": record["directory"],
                "start": record["start"],
                "source": "qe-engine",
                "error": record.get("error"),
            },
        )

    def compute(self, atoms: Atoms, label: str | None = None, *,
                request_id: str | None = None) -> EngineResult:
        # Same sink rule as the handwritten path: a caller naming the parent
        # logical request expects per-launch attempt events to reach its
        # ledger; without a sink they would be silently lost.
        if request_id is not None and self._event_log is None:
            raise EngineError(
                f"compute received request_id {request_id!r} but this engine has "
                "no attempt sink: per-launch attempt events would be lost. "
                "Attach the run's event log (event_log=...) or call without "
                "request_id (execution records then stay in "
                "last_attempt_records only)"
            )
        _initial_magmoms(atoms)  # reject noncollinear states before any launch
        base = "eval" if label is None else str(label).replace("/", "_")
        if request_id is None:
            self._request_counter += 1
            request_id = f"ase-qe-request-{self._request_counter}"
        self.last_attempt_records = []

        density: DensitySource | None = None
        if self.config.startpot_file:
            density = self._resolve_density(atoms)

        retries_done = 0
        attempt = 1
        while True:
            use_density = density is not None and attempt == 1
            try:
                result = self._attempt(atoms, base, density if use_density else None,
                                       attempt=attempt, request_id=request_id)
            except QeEngineError as failure:
                record = self.last_attempt_records[-1]
                if record["status"] == "running":
                    # Pre-launch failure (staging, input write, missing
                    # executable): terminated here; no process ever started,
                    # so no attempt event.
                    record["status"] = "failed"
                record.update(error=str(failure), retryable=failure.retryable)
                # A failed density start gets one atomic-start retry even when
                # the failure itself is deterministic (stale density): the
                # retry runs a different input. Anything else retries only
                # when the failure is classified retryable.
                changes_input = use_density
                if retries_done >= self.config.max_retries:
                    raise
                if not (failure.retryable or changes_input):
                    raise
                retries_done += 1
                attempt += 1
                continue
            self._last_density_dir = Path(self.last_attempt_records[-1]["directory"])
            total_wall = sum(r.get("wall_time_s", 0.0)
                             for r in self.last_attempt_records)
            return EngineResult(
                energy=result.energy,
                forces=result.forces,
                stress=result.stress,
                wall_time_s=total_wall,
                energy_kind=result.energy_kind,
                force_consistent=result.force_consistent,
            )

    def _attempt(self, atoms: Atoms, base: str, density: DensitySource | None,
                 *, attempt: int, request_id: str) -> EngineResult:
        """One attempt: stage (optional), write input, exactly one explicit
        ASE execution, read the whole result, validate the shared contract.

        The phases are driven directly so the launch boundary is exact:
        ``write_inputfiles``/``read_results`` are not launches; ``execute``
        is. A FileNotFoundError out of ``execute`` means the process never
        started (zero launches, no attempt event). Property getters are
        never used for reading — they let ASE silently re-execute when a
        property is missing.
        """
        directory = allocate_run_dir(self.run_root, base)
        record: dict = {
            "attempt": attempt,
            "directory": str(directory),
            "start": "density" if density is not None else "atomic",
            "status": "running",
            "failure_kind": None,
            "error": None,
            "retryable": None,
            "returncode": None,
        }
        self.last_attempt_records.append(record)
        # The attempt span covers staging + write + process + validation, so
        # the density-copy I/O event is genuinely nested inside it.
        record["started_unix"] = time.time()
        t0 = time.perf_counter()
        try:
            if density is not None:
                self._io_counter += 1
                copy_s, copy_bytes = _stage_density_into(
                    density, directory, event_log=self._event_log,
                    request_id=request_id, io_counter=self._io_counter,
                )
                record["density_from"] = str(density.origin_dir)
                record["density_copy_s"] = copy_s
                record["density_copy_bytes"] = copy_bytes
            self.calculator.directory = directory
            self._apply_electronic_state(atoms, startpot=density is not None)
            # Every compute is a real execution: an external SCF must never be
            # served from ASE's geometry cache (reference executions are
            # accounted per launch; legitimate reuse is Pyramid's own label
            # cache, not an invisible calculator cache).
            self.calculator.atoms = None
            self.calculator.results.clear()
            self.calculator.write_inputfiles(atoms, ["energy", "free_energy",
                                                     "forces", "stress"])
        except Exception as error:
            record.update(status="failed", failure_kind="input_write",
                          error=repr(error))
            raise QeEngineError(
                f"espresso input could not be written in {directory}: {error}"
            ) from error

        process_t0 = time.perf_counter()
        try:
            self.calculator.template.execute(directory, self.calculator.profile)
        except FileNotFoundError as error:
            # The executable never started: zero launches, no attempt event.
            record.update(status="failed", failure_kind="executable_missing",
                          error=str(error))
            raise QeEngineError(
                f"pw.x executable not found ({self.calculator.profile.command!r}): "
                f"{error}"
            ) from error
        except subprocess.CalledProcessError as error:
            record["process_s"] = time.perf_counter() - process_t0
            record["wall_time_s"] = time.perf_counter() - t0
            record["returncode"] = error.returncode
            failure = self._classified(
                EngineError(f"pw.x exited with code {error.returncode}"),
                directory)
            record.update(status="failed", failure_kind="process",
                          error=str(failure))
            self._emit_attempt(record, request_id=request_id)
            raise failure from error
        record["process_s"] = time.perf_counter() - process_t0
        try:
            results = dict(self.calculator.template.read_results(directory))
            self.calculator.results = results
            self.calculator.atoms = atoms.copy()
            result = self._validated_result(results, len(atoms))
            check_qe_run_text(self._read_output(directory), len(atoms))
        except QeEngineError as error:
            record["wall_time_s"] = time.perf_counter() - t0
            record.update(status="failed", failure_kind="parse", error=str(error))
            self._emit_attempt(record, request_id=request_id)
            raise
        except Exception as error:
            record["wall_time_s"] = time.perf_counter() - t0
            record.update(status="failed", failure_kind="parse", error=repr(error))
            self._emit_attempt(record, request_id=request_id)
            raise QeEngineError(
                f"espresso output processing failed in {directory}: {error}",
                retryable=True,
            ) from error
        record["wall_time_s"] = time.perf_counter() - t0
        try:
            write_density_manifest(
                directory, engine=self, atoms=atoms,
                source=(
                    {"kind": "atomic"} if density is None else {
                        "kind": "copied",
                        "from": str(density.origin_dir),
                        "from_fingerprint": density.manifest.get(
                            "reference_fingerprint"),
                        "copy_s": record["density_copy_s"],
                        "copy_bytes": record["density_copy_bytes"],
                    }
                ),
            )
        except Exception as error:
            # SCF and parse succeeded; the provenance sidecar did not.
            record.update(status="post_processing_failed",
                          failure_kind="post_processing", error=repr(error))
            self._emit_attempt(record, request_id=request_id)
            raise QeEngineError(
                f"density manifest could not be written in {directory}: {error}"
            ) from error
        record.update(status="success", error=None)
        self._emit_attempt(record, request_id=request_id)
        return result

    def _validated_result(self, results: dict, nat: int) -> EngineResult:
        """Numeric completeness of one executed run's results (units and
        signs are ASE's espresso reader's; the text contract is checked
        separately by check_qe_run_text)."""
        energy_key = "free_energy" if self.force_consistent else "energy"
        energy = results.get(energy_key)
        forces = np.asarray(results.get("forces", []), dtype=float)
        stress = results.get("stress")
        if energy is None or not np.isfinite(float(energy)):
            raise QeEngineError("espresso results have no finite energy")
        if forces.shape != (nat, 3) or not np.isfinite(forces).all():
            raise QeEngineError(
                f"espresso force block shape {forces.shape} != ({nat}, 3)")
        if stress is None:
            raise QeEngineError(
                "espresso results have no stress although tstress was requested")
        stress = np.asarray(stress, dtype=float)
        if stress.shape != (6,) or not np.isfinite(stress).all():
            raise QeEngineError("espresso stress is not a finite (6,) vector")
        return EngineResult(
            energy=float(energy),
            forces=np.array(forces, dtype=float, copy=True),
            stress=np.array(stress, dtype=float, copy=True),
            wall_time_s=float("nan"),  # replaced by the caller's span timing
            energy_kind=self.capabilities.energy_kind,
            force_consistent=True,
        )

    @staticmethod
    def _read_output(directory: Path) -> str:
        # ASE 3.29 EspressoTemplate writes stdout to espresso.pwo; read it
        # defensively by name so a missing file classifies as transient.
        path = directory / "espresso.pwo"
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""

    def _classified(self, error: EngineError, directory: Path) -> QeEngineError:
        kind = classify_qe_failure_text(self._read_output(directory))
        retryable = kind is None
        note = {"nonconverged": "SCF did not converge",
                "input_error": "deterministic input error"}.get(kind)
        message = f"ASE espresso evaluation failed: {error}"
        if note:
            message = f"{message} ({note})"
        return QeEngineError(message, retryable=retryable)

    def _resolve_density(self, atoms: Atoms) -> DensitySource | None:
        """Pick a compatible known-origin density, or decide atomic up front."""
        candidates: list[tuple[str, Path]] = []
        if self.config.density_source is not None:
            candidates.append(("config.density_source", Path(self.config.density_source)))
        if self._last_density_dir is not None:
            candidates.append(("previous attempt", self._last_density_dir))
        reasons: list[str] = []
        for via, origin in candidates:
            source, reason = load_density_source(origin, engine=self, atoms=atoms)
            if source is not None:
                self.last_density_decision = {
                    "start": "density", "origin": str(source.origin_dir), "via": via,
                }
                return source
            reasons.append(f"{via} ({origin}): {reason}")
        self.last_density_decision = {
            "start": "atomic",
            "reason": "; ".join(reasons) if reasons else "no density source configured",
        }
        return None


def create_ase_qe_engine(*, run_root: str | Path, command: str | None = None,
                         event_log: object | None = None,
                         **config_kwargs) -> AseQeEngine:
    """Registry factory: build an AseQeEngine from plain keyword settings."""
    return AseQeEngine(QeConfig(**config_kwargs), run_root, command=command,
                       event_log=event_log)
