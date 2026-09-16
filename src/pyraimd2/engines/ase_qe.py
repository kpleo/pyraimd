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
import uuid
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from pyraimd2.engines import density_publish
from pyraimd2.engines.ase_engine import AseEngine
from pyraimd2.engines.base import (
    EnergyKind,
    EngineCapabilities,
    EngineError,
    EngineResult,
)
from pyraimd2.engines.qe_engine import (
    _DISK_IO_WITHOUT_DENSITY,
    QE_PREFIX,
    DensitySource,
    QeConfig,
    QeEngineError,
    _density_read_evidence_for,
    _initial_magmoms,
    _mark_manifest_scratch_removed,
    _settings_digest,
    _species,
    _stage_density_into,
    _total_charge,
    allocate_run_dir,
    check_execution_options,
    check_qe_run_text,
    check_scratch_options,
    classify_qe_failure_text,
    density_output_file,
    load_density_source,
    normalize_config_paths,
    order_density_candidates,
    recipe_name,
    write_density_manifest,
)
from pyraimd2.runtime import scratch as scratch_mod

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
    if config.disk_io is not None:
        control["disk_io"] = config.disk_io
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

    ``density_registry_run_dir`` (optional) switches on the same persistent
    density chain as the handwritten path — one shared bridge, one
    contract (publication plus the delayed producer release); see
    :mod:`pyraimd2.engines.density_publish`.  Absent the option every
    behavior is exactly as before.
    """

    def __init__(self, config: QeConfig, run_root: str | Path, *,
                 command: str | None = None,
                 event_log: object | None = None,
                 density_registry_run_dir: str | Path | None = None) -> None:
        config = normalize_config_paths(config)
        check_scratch_options(config, engine_name=recipe_name(config))
        check_execution_options(config, engine_name=recipe_name(config))
        if config.timeout_s != _TIMEOUT_DEFAULT:
            raise EngineError(
                "AseQeEngine cannot enforce timeout_s through ASE's FileIO "
                "layer; use the handwritten QeEngine for bounded runs "
                f"(got timeout_s={config.timeout_s})"
            )
        self.config = config
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._density_registry_dir = density_publish.resolve_registry_owner(
            self.run_root, density_registry_run_dir,
            scratch_root=self.config.scratch_root,
            retention=self.config.retention, engine_name=self.name)
        self._command = command
        self._event_log = event_log
        self._scratch_run_uuid = uuid.uuid4().hex[:12]
        self._last_scratch_handle = None
        self._io_counter = 0
        self._request_counter = 0
        self._last_density_dir: Path | None = None
        # persistent-chain state (inert unless the registry is enabled):
        # the resume-pinned generation (cleared after the first successful
        # compute consumes it) and the current persistent head — the
        # generation this run's latest committed state actually depends on
        self._pinned_density_generation: int | None = None
        self._density_head: int | None = None
        # the independent-consumption proof of the latest successful compute
        # (which registry generation it actually read, with the raw-output
        # evidence); consumed by the next release_consumed_scratch call — a
        # cache hit carries no proof.  A successful attempt that staged a
        # registry seed but shows no actual-read evidence lands in
        # _consumption_unproven instead: the producer is preserved.
        self._pending_consumption: dict | None = None
        self._consumption_unproven: dict | None = None
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

    @property
    def density_registry_enabled(self) -> bool:
        """Whether this engine is wired to a run-owned density registry."""
        return self._density_registry_dir is not None

    def current_density_generation(self) -> int | None:
        """The persistent density head, same contract as the handwritten
        path: the generation the last successful evaluation published, or
        the registry generation it consumed when nothing new was
        published; ``None`` when the current state has no registry
        dependency (atomic/legacy/external start, or the ``fixed``
        external-density policy).  Right after a resume the pin stands in
        for the head: the restored boundary's dependency is exactly the
        pinned generation, even if the first evaluation is served from
        the calculator cache and no compute has run yet."""
        if self._density_head is not None:
            return self._density_head
        return self._pinned_density_generation

    def pin_density_generation(self, generation: int | None) -> None:
        """Bind the first new evaluation after a resume to exactly this
        published generation (workflow-resume only) — the same contract
        as the handwritten path: the pin is consumed and cleared by the
        first successful compute, and a pinned generation that fails
        verification refuses the evaluation instead of silently falling
        back to the latest pointer.  The decision trail records the
        pin."""
        self._pinned_density_generation = generation
        if generation is not None:
            self.last_density_decision = {
                "start": "density", "via": "density_registry",
                "generation": int(generation), "pinned": True,
                "note": ("resume binding pinned by the workflow; the first "
                         "new evaluation resolves exactly this generation")}

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

    def _emit_receipt(self, phase: str, record: dict, *,
                      request_id: str, **extra) -> None:
        """The same durable launch-receipt protocol as the handwritten QE
        path (C2), around ASE's ``execute`` boundary: ``prepared`` once the
        input files exist; ``not_launched`` when ASE reports the executable
        missing.  Process creation happens inside ``execute`` and is not
        observable from here, so this adapter never writes ``started`` — a
        crash inside ``execute`` leaves a prepared-only receipt, which the
        ledger honestly reports as launch-unknown."""
        if self._event_log is None:
            return
        self._event_log.append(
            "attempt_receipt",
            {
                "record": "physical_attempt_receipt",
                "phase": phase,
                "operation": "reference",
                "purpose": "scf",
                "request_id": request_id,
                "attempt": record["attempt"],
                "directory": record["directory"],
                "started_unix": record["started_unix"],
                "start": record["start"],
                "source": "qe-engine",
                **extra,
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
        first_archive: Path | None = None
        if request_id is None:
            if self._density_registry_dir is not None:
                # Same persistent directory-derived request identity as the
                # handwritten path: anchored at the first attempt's
                # actually allocated directory, unique under the run owner
                # across engine rebuilds and repeated computes.
                first_archive = allocate_run_dir(self.run_root, base)
                request_id = density_publish.directory_request_id(
                    self._density_registry_dir, first_archive)
            else:
                self._request_counter += 1
                request_id = f"ase-qe-request-{self._request_counter}"
        self.last_attempt_records = []
        self._pending_consumption = None
        self._consumption_unproven = None

        density: DensitySource | None = None
        density_input_generation: int | None = None
        if self.config.startpot_file:
            density = self._resolve_density(atoms)
            decision = self.last_density_decision or {}
            if decision.get("via") == "density_registry":
                density_input_generation = decision.get("generation")

        retries_done = 0
        attempt = 1
        while True:
            use_density = density is not None and attempt == 1
            # this attempt's persistent input binding (None for an atomic
            # or legacy-chain start): recorded on the attempt record and,
            # with a managed scratch, on the durable record BEFORE the
            # launch, so compute_references protects the consumed
            # generation for the attempt's whole in-flight lifetime
            attempt_input_generation = (density_input_generation
                                        if use_density else None)
            try:
                result = self._attempt(atoms, base, density if use_density else None,
                                       attempt=attempt, request_id=request_id,
                                       archive_dir=(first_archive
                                                    if attempt == 1 else None),
                                       density_input_generation=
                                       attempt_input_generation)
            except QeEngineError as failure:
                record = self.last_attempt_records[-1]
                if record["status"] == "running":
                    # Pre-launch failure (staging, input write, missing
                    # executable): terminated here; no process ever started,
                    # so no attempt event.
                    record["status"] = "failed"
                record.update(error=str(failure), retryable=failure.retryable)
                if self._density_registry_dir is not None:
                    record["density_input_generation"] = attempt_input_generation
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
            record = self.last_attempt_records[-1]
            handle = self._last_scratch_handle
            archive_dir = (handle.archive_dir if handle is not None
                           else Path(record["directory"]))
            # only an attempt that actually left a charge density behind may
            # become the chained-start source of a later evaluation
            produced_density = (archive_dir
                                if record.get("density_available", True) else None)
            if self._density_registry_dir is not None:
                record["density_input_generation"] = attempt_input_generation
                if produced_density is not None:
                    # save-only publication through the same shared bridge as
                    # the handwritten path; the outcome is recorded, never raised
                    record["density_publish"] = self._publish_density(
                        record=record, handle=handle, atoms=atoms,
                        request_id=request_id, attempt_id=f"attempt-{attempt}")
                self._update_density_head(
                    record.get("density_publish"), attempt_input_generation)
                if attempt_input_generation is not None:
                    # the independent-consumption proof for the delayed
                    # release — same contract as the handwritten path: this
                    # attempt actually staged the registry generation as its
                    # density start AND its raw output shows the solver
                    # really read it (never an atomic retry, a cache hit, a
                    # borrow from the producer's own tree, or a successful
                    # run that silently ignored the staged seed)
                    staged_from = record.get("density_from")
                    registry = Path(self._density_registry_dir) / "restart" \
                        / "density"
                    if staged_from is not None and Path(staged_from) \
                            .resolve().is_relative_to(registry):
                        read_evidence = record.get("density_read_evidence")
                        if read_evidence is not None:
                            self._pending_consumption = {
                                "generation": int(attempt_input_generation),
                                "consumer_attempt": {
                                    "request_id": request_id,
                                    "attempt_id": f"attempt-{attempt}"},
                                "staged_from": staged_from,
                                "staged_bytes": record.get(
                                    "density_copy_bytes"),
                                "staged_copy_s": record.get(
                                    "density_copy_s"),
                                # the consumed seed's identity pinned at
                                # selection/staging time — the release gate
                                # cross-checks it against the producer's
                                # pending receipt and the live manifest
                                "seed_content_digest":
                                    density.manifest.get("content_digest"),
                                "reference_fingerprint":
                                    density.manifest.get(
                                        "reference_fingerprint"),
                                "read_evidence": read_evidence}
                        else:
                            self._consumption_unproven = {
                                "generation": int(attempt_input_generation),
                                "reason": record.get(
                                    "density_read_evidence_missing")
                                    or "no actual-read evidence was parsed"}
                # the resume pin binds exactly one evaluation: the first
                # successful compute consumed it
                self._pinned_density_generation = None
            if handle is not None and self.config.retention == "results":
                receipt = scratch_mod.cleanup(handle)
                record["scratch_cleanup"] = receipt  # the complete receipt
                if receipt["status"] == "cleaned":
                    _mark_manifest_scratch_removed(handle.archive_dir)
                # the reclaimed density no longer exists — never claim it
                # as a source for a later chained start
                self._last_density_dir = None
            elif handle is not None:
                record["scratch_cleanup"] = scratch_mod.mark_kept(handle)
                self._last_density_dir = produced_density
            else:
                self._last_density_dir = produced_density
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
                 *, attempt: int, request_id: str,
                 archive_dir: Path | None = None,
                 density_input_generation: int | None = None) -> EngineResult:
        """One attempt: stage (optional), write input, exactly one explicit
        ASE execution, read the whole result, validate the shared contract.

        The phases are driven directly so the launch boundary is exact:
        ``write_inputfiles``/``read_results`` are not launches; ``execute``
        is. A FileNotFoundError out of ``execute`` means the process never
        started (zero launches, no attempt event). Property getters are
        never used for reading — they let ASE silently re-execute when a
        property is missing.

        ``archive_dir`` is the caller's pre-allocated directory when the
        persistent request identity was anchored at it (density-registry
        mode); otherwise the attempt claims the next free one as before.
        ``density_input_generation`` is the persistent input binding of
        THIS attempt: with a managed scratch it lands on the durable
        record before the launch (in-flight protection; same contract as
        the handwritten path).
        """
        if archive_dir is None:
            archive_dir = allocate_run_dir(self.run_root, base)
        handle = None
        if self.config.scratch_root is not None:
            handle = scratch_mod.allocate(
                run_root=(self._density_registry_dir
                          if self._density_registry_dir is not None
                          else self.run_root),
                scratch_root=self.config.scratch_root,
                run_uuid=self._scratch_run_uuid, backend_role="reference",
                request_id=request_id, attempt_id=f"attempt-{attempt}",
                archive_dir=archive_dir, retention=self.config.retention)
            if self._density_registry_dir is not None:
                handle.update_record(
                    density_generation=density_input_generation)
        self._last_scratch_handle = handle
        work_dir = handle.scratch_dir if handle is not None else archive_dir
        record: dict = {
            "attempt": attempt,
            "directory": str(work_dir),
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
                    density, work_dir, event_log=self._event_log,
                    request_id=request_id, io_counter=self._io_counter,
                )
                record["density_from"] = str(density.origin_dir)
                record["density_copy_s"] = copy_s
                record["density_copy_bytes"] = copy_bytes
            self.calculator.directory = work_dir
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
            self._mark_scratch_failed(handle, repr(error))
            raise QeEngineError(
                f"espresso input could not be written in {work_dir}: {error}"
            ) from error

        # Durable prepared receipt before ASE's execute boundary (C2).
        self._emit_receipt("prepared", record, request_id=request_id)
        process_t0 = time.perf_counter()
        try:
            self.calculator.template.execute(work_dir, self.calculator.profile)
        except FileNotFoundError as error:
            # The executable never started: zero launches, no attempt event.
            self._emit_receipt("not_launched", record,
                               request_id=request_id, error=str(error))
            record.update(status="failed", failure_kind="executable_missing",
                          error=str(error))
            self._mark_scratch_failed(handle, str(error))
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
                work_dir)
            record.update(status="failed", failure_kind="process",
                          error=str(failure))
            self._emit_attempt(record, request_id=request_id)
            self._mark_scratch_failed(handle, str(failure))
            raise failure from error
        record["process_s"] = time.perf_counter() - process_t0
        try:
            results = dict(self.calculator.template.read_results(work_dir))
            self.calculator.results = results
            self.calculator.atoms = atoms.copy()
            result = self._validated_result(results, len(atoms))
            check_qe_run_text(self._read_output(work_dir), len(atoms))
        except QeEngineError as error:
            record["wall_time_s"] = time.perf_counter() - t0
            record.update(status="failed", failure_kind="parse", error=str(error))
            self._emit_attempt(record, request_id=request_id)
            self._mark_scratch_failed(handle, str(error))
            raise
        except Exception as error:
            record["wall_time_s"] = time.perf_counter() - t0
            record.update(status="failed", failure_kind="parse", error=repr(error))
            self._emit_attempt(record, request_id=request_id)
            self._mark_scratch_failed(handle, repr(error))
            raise QeEngineError(
                f"espresso output processing failed in {work_dir}: {error}",
                retryable=True,
            ) from error
        # The actual-read observation of a density start, parsed from THIS
        # attempt's own raw output and launch input (same parser and
        # contract as the handwritten path); missing/ambiguous evidence is
        # recorded with its reason, never assumed.
        _density_read_evidence_for(
            record, density,
            output_path=work_dir / "espresso.pwo",
            input_path=work_dir / "espresso.pwi")
        save_tree = work_dir / "tmp" / f"{QE_PREFIX}.save"
        # Same product rule as the handwritten path: disk_io none/minimal
        # never write a density from this SCF (any density file present is
        # the staged input copy); producing modes are verified by the actual
        # charge-density file (.dat or .hdf5).
        density_available = (
            self.config.disk_io not in _DISK_IO_WITHOUT_DENSITY
            and density_output_file(save_tree) is not None)
        record["density_available"] = density_available
        try:
            write_density_manifest(
                archive_dir, engine=self, atoms=atoms,
                save_dir=(None if handle is None else str(save_tree)),
                density_available=density_available,
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
            # SCF and parse succeeded; the provenance sidecar did not. The
            # span is settled at this terminal exit, so the attempt covers
            # staging + process + validation including this write.
            record["wall_time_s"] = time.perf_counter() - t0
            record.update(status="post_processing_failed",
                          failure_kind="post_processing", error=repr(error))
            self._emit_attempt(record, request_id=request_id)
            raise QeEngineError(
                "density manifest could not be written in "
                f"{archive_dir}: {error}"
            ) from error
        if handle is not None:
            # the durable result must be archived OUT of the scratch root
            # before any reclaim
            try:
                scratch_mod.archive(handle, ["espresso.pwi", "espresso.pwo"])
            except Exception as error:
                record["wall_time_s"] = time.perf_counter() - t0
                record.update(status="post_processing_failed",
                              failure_kind="post_processing",
                              error=repr(error))
                self._emit_attempt(record, request_id=request_id)
                raise QeEngineError(
                    f"scratch archive failed in {work_dir}: {error}"
                ) from error
        record["wall_time_s"] = time.perf_counter() - t0
        record.update(status="success", error=None)
        self._emit_attempt(record, request_id=request_id)
        return result

    @staticmethod
    def _mark_scratch_failed(handle, error_text: str) -> None:
        """Best-effort failed_kept marking; the real error always wins."""
        if handle is None:
            return
        try:
            scratch_mod.mark_failed(handle, error_text)
        except Exception:  # noqa: BLE001, S110 — the real error must win
            pass

    def _validated_result(self, results: dict, nat: int) -> EngineResult:
        """Numeric completeness of one executed run's results (units and
        signs are ASE's espresso reader's; the text contract is checked
        separately by check_qe_run_text)."""
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

    def _update_density_head(self, publish: dict | None,
                             input_generation: int | None) -> None:
        """Advance the persistent head after a successful compute — the
        same contract as the handwritten path: a verified publication
        moves the head to the new generation; an evaluation that consumed
        a registry generation without publishing keeps that input;
        anything else (atomic/legacy-chain input, or the ``fixed``
        external-density policy) leaves no registry dependency."""
        if self.config.density_source_policy == "fixed":
            self._density_head = None
        elif publish is not None and publish.get("status") == "published":
            self._density_head = int(publish["generation"])
        elif input_generation is not None:
            self._density_head = int(input_generation)
        else:
            self._density_head = None

    def _resolve_density(self, atoms: Atoms) -> DensitySource | None:
        """Pick a compatible known-origin density, or decide atomic up front.

        The same registry-aware order as the handwritten path over the
        same shared bridge: a resume-pinned generation first (strictly
        resolved — a failed verification refuses the evaluation, never a
        silent swap to the latest pointer); otherwise, under the default
        ``latest`` policy, the registry's latest verified generation; the
        ``fixed`` policy never consults the registry for source
        selection.  When the registry offers nothing usable under
        ``latest``, the pre-registry order runs unchanged
        (:func:`order_density_candidates`), with the registry miss noted
        in the decision record.  Without the registry the behavior is
        byte-identical to before: a fresh process re-initializes from
        ``config.density_source`` and says so, never silently claiming
        the previous process's latest density.
        """
        if self._density_registry_dir is not None:
            pinned = self._pinned_density_generation
            if pinned is not None:
                payload, reason, generation = density_publish.load_published_density(
                    self._density_registry_dir, generation=pinned,
                    reference_fingerprint=self.fingerprint, nat=len(atoms),
                    species=sorted(_species(atoms)))
                if payload is None:
                    raise QeEngineError(
                        f"the resume-pinned density generation g{pinned:06d} "
                        f"cannot be used: {reason}; refusing to substitute "
                        "another generation for the bound resume reference")
                generation_dir, save_dir, manifest = payload
                self.last_density_decision = {
                    "start": "density", "origin": str(generation_dir),
                    "via": "density_registry", "generation": generation,
                    "pinned": True}
                return DensitySource(origin_dir=generation_dir,
                                     save_dir=save_dir, manifest=manifest)
            if self.config.density_source_policy == "latest":
                payload, reason, generation = density_publish.load_published_density(
                    self._density_registry_dir, generation=None,
                    reference_fingerprint=self.fingerprint, nat=len(atoms),
                    species=sorted(_species(atoms)))
                if payload is not None:
                    generation_dir, save_dir, manifest = payload
                    self.last_density_decision = {
                        "start": "density", "origin": str(generation_dir),
                        "via": "density_registry", "generation": generation}
                    return DensitySource(origin_dir=generation_dir,
                                         save_dir=save_dir, manifest=manifest)
                # fall through to the pre-registry chain; the decision
                # record keeps the registry miss alongside the legacy
                # candidates' reasons
                registry_miss = f"density_registry (latest): {reason}"
            else:
                registry_miss = None
        else:
            registry_miss = None
        fresh_process = self._last_density_dir is None
        reasons: list[str] = ([registry_miss] if registry_miss is not None
                              else [])
        for via, origin in order_density_candidates(self.config,
                                                    self._last_density_dir):
            source, reason = load_density_source(origin, engine=self, atoms=atoms)
            if source is not None:
                decision: dict = {"start": "density",
                                  "origin": str(source.origin_dir), "via": via}
                if (via == "config.density_source" and fresh_process
                        and self.config.density_source_policy == "latest"):
                    decision["note"] = (
                        "fresh process: initialized from config.density_source; "
                        "the previous process's latest density is not recovered "
                        "automatically")
                self.last_density_decision = decision
                return source
            reasons.append(f"{via} ({origin}): {reason}")
        self.last_density_decision = {
            "start": "atomic",
            "reason": "; ".join(reasons) if reasons else "no density source configured",
        }
        return None

    def _publish_density(self, *, record: dict, handle, atoms: Atoms,
                         request_id: str, attempt_id: str) -> dict:
        """Publish this successful attempt's density into the run-owned
        registry through the shared bridge — the same save-only contract
        as the handwritten path
        (:mod:`pyraimd2.engines.density_publish`).  The attempt record
        keeps the returned status under ``density_publish``; a publication
        failure never reruns the SCF and never claims a generation."""
        work_dir = Path(record["directory"])
        save_tree = work_dir / "tmp" / f"{QE_PREFIX}.save"
        return density_publish.publish_attempt_density(
            owner_dir=self._density_registry_dir,
            source_root=work_dir,
            density_file=density_output_file(save_tree),
            scratch_handle=handle,
            request_id=request_id,
            attempt_id=attempt_id,
            reference_fingerprint=self.fingerprint,
            nat=len(atoms),
            species=sorted(_species(atoms)),
            disk_io=self.config.disk_io,
            source_desc={
                "kind": "qe-attempt",
                "engine": self.name,
                "work_dir": str(work_dir),
                "archive_dir": str(handle.archive_dir if handle is not None
                                   else work_dir),
            },
        )

    def release_consumed_scratch(self, *, evaluation_id: int) -> dict:
        """Release managed scratch under the delayed-release contract — a
        receipt, never an exception into the run loop.

        This attempt's own scratch is released only once its published
        seed has been independently consumed by a later successful
        calculation; until then the attempt carries a pending release
        receipt on its authoritative record and stays kept (a protected
        resource, not a cleanup failure).  The shared implementation
        (:func:`density_publish.release_consumed_scratch`) behaves
        identically for both QE adapters.
        """
        return density_publish.release_consumed_scratch(
            self, evaluation_id=evaluation_id,
            on_reclaimed=self._note_reclaimed_scratch)

    def _note_reclaimed_scratch(self, archive_dir: Path) -> None:
        """An attempt's scratch was reclaimed: stop claiming its save tree
        as a later evaluation's source.  The archive lives outside the
        scratch root and is untouched; only its manifest stops claiming
        the .save survives (same mark as the retention="results" path)."""
        _mark_manifest_scratch_removed(archive_dir)
        if self._last_density_dir is not None \
                and self._last_density_dir == archive_dir:
            self._last_density_dir = None


def create_ase_qe_engine(*, run_root: str | Path, command: str | None = None,
                         event_log: object | None = None,
                         density_registry_run_dir: str | Path | None = None,
                         **config_kwargs) -> AseQeEngine:
    """Registry factory: build an AseQeEngine from plain keyword settings."""
    return AseQeEngine(QeConfig(**config_kwargs), run_root, command=command,
                       event_log=event_log,
                       density_registry_run_dir=density_registry_run_dir)
