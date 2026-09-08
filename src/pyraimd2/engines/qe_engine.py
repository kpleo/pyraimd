"""Quantum ESPRESSO engine: periodic DFT labels via pw.x.

The engine is a swappable label source behind the ``Engine`` protocol;
failures raise :class:`EngineError` — never silently reuse stale output.
Each call gets its own directory ``run_root/<label>-NNNNNN`` and each
execution a numbered ``attempt-N`` subdirectory, so consecutive evaluations
never clobber one another and a failed attempt keeps its diagnostics. The
input path handed to pw.x is absolute — a relative ``run_root`` must not
become invalid once the subprocess changes its working directory.

Physical recipe: the XC functional and dispersion correction are explicit
configuration (``xc``, ``dispersion``), written into the input as
``input_dft``/``vdw_corr`` — nothing is hardcoded, and the engine name and
fingerprint state the recipe (e.g. ``qe-pbe-d3``). Changing the recipe
changes the fingerprint; old configs keep their old meaning because the
defaults are exactly the previous hardcoded values (PBE + Grimme D3).

Reference identity: the fingerprint hashes the full config plus the
pseudopotential identity — file name *and file content sha256* per species.
When ``pseudo_dir`` is unreadable the content hash degrades honestly to
``null`` (recorded as unknown, never invented), and the fingerprint still
covers the configured file names.

Energy convention: a converged pw.x ``! total energy`` is the quantity its
forces differentiate. Without smearing that is the total energy
(``energy_kind="energy"``); with smearing the ``!`` line reports the
variational free energy, so a metallic config declares
``energy_kind="free_energy"`` — both are force-consistent.

SCF success is a combination, never the returncode alone: exit code 0,
the ``JOB DONE.`` marker present, no ``convergence NOT achieved`` marker,
and a complete parse (energy + full force block, right atom count, finite
values). Output is parsed from the pw.x stdout text: the
``! total energy`` (Ry) of the last complete SCF block, the ``Forces
acting on atoms`` block (Ry/bohr) that follows it, and the 3x3 stress
block (Ry/bohr^3) after that — energy, forces and stress always come from
the same block, never mixed across blocks. Fortran ``D`` exponents are
converted.

Retries are bounded (``max_retries``) and classified: timeouts, crashes
and truncated output are retryable; deterministic input problems (missing
pseudopotential, singular cell, a QE ``Error in routine`` banner) and SCF
non-convergence of an identical input are not — rerunning them would only
burn cost. A failed chained-density start always gets its single
atomic-start retry (different input), as before. On timeout the whole
process group is killed (``start_new_session`` + ``killpg``), so an
mpirun-wrapped pw.x leaves no orphaned ranks behind.

Density warm start (``startpot_file=True``): before launch the engine
looks for a compatible density of known origin — ``density_source`` first,
then the engine's own last successful attempt. Compatibility means a
density manifest (written next to every successful attempt's ``.save``)
whose reference fingerprint, atom count and species match, with the
``.save`` tree actually present. A compatible density is *copied* into the
new attempt's writable directory — two computations never share one
writable ``.save`` — and the copy is recorded (origin, bytes, seconds) in
the attempt record and, when an event log is attached, as an ``io`` task
event in the WP02 cost ledger. Without a compatible density the first
attempt already starts atomic: no directory deliberately fails once just
to discover there is nothing to restart from.

Stress sign: pw.x reports stress with compression-positive convention; we
convert to the ASE convention (``stress_ase = -stress_qe``) and voigt order
(xx, yy, zz, yz, xz, xy). The sign is asserted against ASE's own espresso
calculator in the remote integration test before any NPT use — NVT-only
until then.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, chemical_symbols

from pyraimd2.engines.base import (
    EnergyKind,
    EngineCapabilities,
    EngineError,
    EngineResult,
)

RY_EV = units.Hartree / 2.0  # QE reports Rydbergs
RY_BOHR3_TO_EV_A3 = RY_EV / units.Bohr**3

QE_PREFIX = "pyraimd2"  # pw.x ``prefix``: the .save tree is tmp/<prefix>.save
DENSITY_MANIFEST = "density_manifest.json"
DENSITY_MANIFEST_SCHEMA = 1

DEFAULT_PSEUDOS: dict[str, str] = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "W": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",  # 14-valence PAW, scalar-relativistic
}

_ENERGY_RE = re.compile(r"^!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry", re.MULTILINE)
_FORCE_LINE_RE = re.compile(
    r"^\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
    r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$"
)

# QE prints deterministic input/physics errors inside this banner; a nonzero
# exit without it is a crash (segfault, launcher failure) and may be retried.
_QE_ERROR_BANNER_RE = re.compile(r"Error in routine\s+\S+", re.IGNORECASE)

# Short slugs so the engine name states the recipe (qe-pbe-d3, qe-pbe, ...).
_DISPERSION_SLUGS = {
    "grimme-d3": "d3",
    "grimme-d2": "d2",
    "dft-d3": "d3",
    "dft-d2": "d2",
}


def _to_float(token: str) -> float:
    """Parse a QE number, accepting Fortran ``D``/``d`` exponents."""
    return float(token.replace("D", "E").replace("d", "e"))


def valence_from_upf(path: Path) -> float:
    """z_valence from a UPF pseudopotential (PP_HEADER attribute; sits well
    past the lengthy PP_INFO/PP_INPUTFILE preamble, so read generously)."""
    text = path.read_text(errors="ignore")[:300_000]
    m = re.search(r'z_valence\s*=\s*"([0-9.eE+-]+)"', text)
    if m:
        return float(m.group(1))
    raise ValueError(f"z_valence not found in {path}")


@dataclass(frozen=True)
class QeConfig:
    """Everything a QeEngine call needs; immutable, explicit (no globals).

    ``xc``/``dispersion`` are the physical recipe: written to the input as
    ``input_dft``/``vdw_corr`` and stated in the engine name. The defaults
    (pbe, grimme-d3) are exactly the previously hardcoded recipe, so old
    configs keep their physical meaning; a PBE-only recipe must say
    ``dispersion=None`` explicitly. ``max_retries`` bounds how often a
    failed attempt may be rerun (retryable failures only; a chained-density
    start also gets one atomic-start fallback). ``density_source`` points
    at a directory holding a :data:`DENSITY_MANIFEST` from a previous
    successful attempt, for cross-run warm starts.
    """

    pseudo_dir: str
    pw_cmd: tuple[str, ...] = ("pw.x",)  # e.g. ("mpirun", "-np", "28", "pw.x")
    xc: str = "pbe"
    dispersion: str | None = "grimme-d3"
    ecutwfc: float = 50.0
    ecutrho: float = 400.0
    kpts: tuple[int, int, int] | None = None  # None -> Gamma only
    nbnd: int | None = None  # None -> QE default (nelec/2): fragile Davidson, set extras!
    metallic: bool = False  # adds Marzari-Vanderbilt smearing
    smearing: str = "mv"  # when metallic; "fd" (Fermi-Dirac) is cleaner for true metals
    degauss: float = 0.02  # Ry, smearing width when metallic
    conv_thr: float = 1e-8  # Ry, total-energy accuracy; 1e-9 never converges at ~1k electrons
    mixing_beta: float = 0.3
    mixing_mode: str = "local-TF"  # cures charge sloshing in large heterogeneous cells
    mixing_ndim: int = 8  # Broyden history length; 12+ steadies ~1k-electron liquids
    diago_david_ndim: int = 4  # Davidson subspace multiplier; 8 for ~500-band systems
    diago_full_acc: bool = False  # tightly converge ALL bands each SCF step
    electron_maxstep: int = 200
    startpot_file: bool = False  # warm start from a compatible density when one exists
    density_source: str | None = None  # directory with a density manifest (known origin)
    max_retries: int = 1  # retries after the first attempt (bounded, classified)
    timeout_s: float = 3600.0
    pseudos: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PSEUDOS))


class QeEngineError(EngineError):
    """An EngineError that states whether rerunning the attempt can help."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)


def _species(atoms: Atoms) -> list[str]:
    seen: dict[str, None] = {}
    for sym in atoms.get_chemical_symbols():
        seen.setdefault(sym)
    return list(seen)


def recipe_name(config: QeConfig) -> str:
    """``qe-<xc>[-<dispersion>]``: the engine name, stating the recipe."""
    name = f"qe-{config.xc.lower()}"
    if config.dispersion is not None:
        disp = config.dispersion.lower()
        name += f"-{_DISPERSION_SLUGS.get(disp, disp)}"
    return name


def write_qe_input(path: Path, atoms: Atoms, cfg: QeConfig) -> None:
    """Write the configured pw.x scf input for ``atoms`` (ibrav=0, explicit cell)."""
    species = _species(atoms)
    missing = [s for s in species if s not in cfg.pseudos]
    if missing:
        raise QeEngineError(f"no pseudopotential configured for species: {missing}")
    if abs(np.linalg.det(atoms.cell)) < 1e-8:
        raise QeEngineError("QeEngine requires a non-singular cell (ibrav=0)")

    lines: list[str] = []
    lines.append("&CONTROL\n")
    lines.append(f"  calculation = 'scf'\n  prefix = '{QE_PREFIX}'\n")
    lines.append(f"  pseudo_dir = '{cfg.pseudo_dir}'\n  outdir = './tmp'\n")
    lines.append("  tprnfor = .true.\n  tstress = .true.\n/\n")
    lines.append("&SYSTEM\n  ibrav = 0\n")
    lines.append(f"  nat = {len(atoms)}\n  ntyp = {len(species)}\n")
    lines.append(f"  ecutwfc = {cfg.ecutwfc}\n  ecutrho = {cfg.ecutrho}\n")
    lines.append(f"  input_dft = '{cfg.xc}'\n")
    if cfg.dispersion is not None:
        lines.append(f"  vdw_corr = '{cfg.dispersion}'\n")
    if cfg.nbnd is not None:
        lines.append(f"  nbnd = {cfg.nbnd}\n")
    if cfg.metallic:
        lines.append(
            f"  occupations = 'smearing'\n  smearing = '{cfg.smearing}'\n  degauss = {cfg.degauss}\n"
        )
    lines.append("/\n")
    lines.append("&ELECTRONS\n")
    lines.append(f"  conv_thr = {cfg.conv_thr:.1e}\n  mixing_beta = {cfg.mixing_beta}\n")
    lines.append(f"  mixing_mode = '{cfg.mixing_mode}'\n")
    lines.append(f"  mixing_ndim = {cfg.mixing_ndim}\n")
    lines.append(f"  diago_david_ndim = {cfg.diago_david_ndim}\n")
    if cfg.diago_full_acc:
        lines.append("  diago_full_acc = .true.\n")
    if cfg.startpot_file:
        lines.append("  startingpot = 'file'\n")
    lines.append(f"  electron_maxstep = {cfg.electron_maxstep}\n/\n")
    lines.append("ATOMIC_SPECIES\n")
    for sym in species:
        mass = atomic_masses[chemical_symbols.index(sym)]
        lines.append(f"  {sym} {mass:.4f} {cfg.pseudos[sym]}\n")
    lines.append("CELL_PARAMETERS angstrom\n")
    for row in atoms.cell:
        lines.append(f"  {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
    lines.append("ATOMIC_POSITIONS angstrom\n")
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"  {sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
    if cfg.kpts is None:
        lines.append("K_POINTS gamma\n")
    else:
        lines.append(f"K_POINTS automatic\n  {cfg.kpts[0]} {cfg.kpts[1]} {cfg.kpts[2]} 0 0 0\n")
    path.write_text("".join(lines))


def parse_qe_output(text: str) -> EngineResult:
    """Parse the last complete SCF block of a pw.x stdout.

    Energy, forces and stress are read from the same block: the last
    ``! total energy`` line (Ry), the force block that follows it (Ry/bohr),
    and the stress block after that (Ry/bohr^3). Concatenated outputs (e.g.
    appended restarts) must never mix values across blocks.
    """
    energies = list(_ENERGY_RE.finditer(text))
    if not energies:
        raise EngineError("pw.x output has no '! total energy' line (SCF never finished?)")
    energy_ev = _to_float(energies[-1].group(1)) * RY_EV
    if not np.isfinite(energy_ev):
        raise EngineError("pw.x reported a non-finite total energy")

    block_start = text.find("Forces acting on atoms", energies[-1].end())
    if block_start == -1:
        raise EngineError(
            "pw.x output has no force block after the last energy (tprnfor missing?)"
        )
    rows: list[list[float]] = []
    for line in text[block_start:].splitlines()[1:]:
        match = _FORCE_LINE_RE.match(line)
        if match:
            rows.append([_to_float(component) for component in match.groups()])
        elif rows:
            break  # end of the contiguous atom block
    if not rows:
        raise EngineError("pw.x force block after the last energy is empty")
    forces_ry_bohr = np.array(rows)
    if not np.isfinite(forces_ry_bohr).all():
        raise EngineError("pw.x reported non-finite forces")
    forces = forces_ry_bohr * (RY_EV / units.Bohr)

    stress = None
    stress_marker = "total   stress  (Ry/bohr**3)"
    idx = text.find(stress_marker, block_start)
    if idx != -1:
        stress_rows: list[list[float]] = []
        for line in text[idx:].splitlines()[1:4]:
            parts = line.split()
            if len(parts) >= 3:
                stress_rows.append([_to_float(parts[0]), _to_float(parts[1]), _to_float(parts[2])])
        if len(stress_rows) == 3:
            s = np.array(stress_rows)  # QE compression-positive, Ry/bohr^3
            if not np.isfinite(s).all():
                raise EngineError("pw.x reported a non-finite stress")
            voigt = -np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )  # -> ASE sign convention
            stress = voigt * RY_BOHR3_TO_EV_A3

    return EngineResult(
        energy=energy_ev, forces=forces, stress=stress, wall_time_s=float("nan")
    )


_PSEUDO_HASH_CACHE: dict[tuple[str, int, int], str | None] = {}


def _pseudo_sha256(path: Path) -> str | None:
    """Content sha256 of a pseudopotential file, or None when unreadable.

    Cached by (path, mtime, size): fingerprints are read per evaluation, so
    multi-MB UPF files must not be re-hashed every time.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _PSEUDO_HASH_CACHE:
        digest: str | None = None
        try:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            digest = None
        _PSEUDO_HASH_CACHE[key] = digest
    return _PSEUDO_HASH_CACHE[key]


def pseudo_identities(config: QeConfig) -> dict[str, dict[str, str | None]]:
    """Per-species pseudopotential identity: file name + content hash.

    ``sha256`` is None when the file cannot be read — the identity is then
    honestly degraded (names only), never invented.
    """
    base = Path(config.pseudo_dir)
    return {
        sym: {"file": config.pseudos[sym], "sha256": _pseudo_sha256(base / config.pseudos[sym])}
        for sym in sorted(config.pseudos)
    }


# Platform/execution knobs, deliberately NOT part of the reference settings
# identity: the launcher, timeouts, retry policy and warm-start plumbing do
# not change the physical label a converged SCF produces. Scientific
# parameters and platform profiles stay separate.
_EXECUTION_FIELDS = frozenset(
    {"pw_cmd", "timeout_s", "max_retries", "density_source", "startpot_file"}
)


def settings_payload(config: QeConfig, *, path_kind: str) -> dict:
    """The full reference-settings identity: recipe, scientific parameters
    and pseudo content hashes (platform/execution fields excluded)."""
    config_dict = {
        key: value
        for key, value in dataclasses.asdict(config).items()
        if key not in _EXECUTION_FIELDS
    }
    return {
        "path": path_kind,
        "engine": recipe_name(config),
        "config": config_dict,
        "pseudopotentials": pseudo_identities(config),
    }


def _settings_digest(config: QeConfig, *, path_kind: str) -> str:
    payload = json.dumps(settings_payload(config, path_kind=path_kind), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DensitySource:
    """A verified density artifact of known origin, ready to be copied."""

    origin_dir: Path  # directory holding the density manifest
    save_dir: Path  # the .save tree itself (inside origin_dir)
    manifest: dict


def write_density_manifest(attempt_dir: Path, *, engine: QeEngine, atoms: Atoms,
                           source: dict) -> Path:
    """Record what a successful attempt produced, so a later computation can
    verify origin and reference-settings compatibility before reusing it."""
    manifest = {
        "schema": DENSITY_MANIFEST_SCHEMA,
        "engine_name": engine.name,
        "reference_fingerprint": engine.fingerprint,
        "nat": len(atoms),
        "species": sorted(_species(atoms)),
        "save_dir": f"tmp/{QE_PREFIX}.save",
        "source": source,
        "created_unix": time.time(),
    }
    path = attempt_dir / DENSITY_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_density_source(origin_dir: Path, *, engine: QeEngine,
                        atoms: Atoms) -> tuple[DensitySource | None, str]:
    """Validate a candidate density directory; returns (source, reason)."""
    manifest_path = origin_dir / DENSITY_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, "no readable density manifest (origin unknown)"
    if not isinstance(manifest, dict) or manifest.get("schema") != DENSITY_MANIFEST_SCHEMA:
        return None, "unsupported density manifest schema"
    if manifest.get("reference_fingerprint") != engine.fingerprint:
        return None, "reference settings differ from the density's"
    if manifest.get("nat") != len(atoms):
        return None, f"atom count differs ({manifest.get('nat')} != {len(atoms)})"
    if manifest.get("species") != sorted(_species(atoms)):
        return None, "species differ"
    save_dir = origin_dir / str(manifest.get("save_dir", ""))
    if not save_dir.is_dir():
        return None, "density files are missing"
    return DensitySource(origin_dir=origin_dir, save_dir=save_dir, manifest=manifest), ""


class QeEngine:
    """Periodic DFT labels from Quantum ESPRESSO pw.x (Engine protocol).

    ``event_log`` is optional: any object with ``append(event_type, payload)``
    following the WP02 task-event schema. When attached, density-copy I/O is
    recorded as ``io`` task events (origin/destination/bytes in the
    ``provenance`` field); the engine only calls the interface, it does not
    depend on the runtime package.
    """

    def __init__(self, config: QeConfig, run_root: str | Path, *,
                 event_log: object | None = None) -> None:
        self.config = config
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._call_counter = 0
        self._event_log = event_log
        self._last_density_dir: Path | None = None
        self._io_counter = 0
        self.last_attempt_records: list[dict] = []
        self.last_density_decision: dict | None = None

    @property
    def name(self) -> str:
        return recipe_name(self.config)

    @property
    def capabilities(self) -> EngineCapabilities:
        # Converged pw.x forces differentiate the reported energy: with
        # smearing the "!" line is the variational free energy, otherwise the
        # total energy. Stress is always requested (tstress) and parsed when
        # printed.
        return EngineCapabilities(
            energy_kind=EnergyKind.FREE_ENERGY if self.config.metallic else EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
        )

    @property
    def fingerprint(self) -> str:
        digest = _settings_digest(self.config, path_kind="qe-subprocess")
        return f"{self.name}:{digest}"

    def compute(self, atoms: Atoms, label: str | None = None) -> EngineResult:
        base = "eval" if label is None else str(label).replace("/", "_")
        run_dir = self.run_root / f"{base}-{self._call_counter:06d}"
        self._call_counter += 1
        self.last_attempt_records = []

        density: DensitySource | None = None
        if self.config.startpot_file:
            density = self._resolve_density(atoms)

        retries_done = 0
        attempt = 1
        while True:
            use_density = density is not None and attempt == 1
            if use_density:
                attempt_config = self.config
            else:
                # No compatible density (or the density start already failed):
                # write an explicit atomic-start input instead of launching a
                # startpot='file' run that is known to have nothing to read.
                attempt_config = dataclasses.replace(self.config, startpot_file=False)
            try:
                result = self._attempt(
                    atoms, run_dir / f"attempt-{attempt}", attempt_config,
                    density=density if use_density else None,
                )
            except QeEngineError as error:
                record = self.last_attempt_records[-1]
                record.update(status="failed", error=str(error), retryable=error.retryable)
                # A failed density start gets one atomic-start retry even when
                # the failure itself is deterministic (stale density): the
                # retry runs a different input. Anything else retries only
                # when the failure is classified retryable.
                changes_input = use_density
                if retries_done >= self.config.max_retries:
                    raise
                if not (error.retryable or changes_input):
                    raise
                retries_done += 1
                attempt += 1
                reason = record["error"].splitlines()[0]
                if changes_input:
                    print(
                        f"startpot chain failed in {run_dir}/attempt-{attempt - 1} "
                        f"({reason}); retrying with an atomic-start input",
                        flush=True,
                    )
                else:
                    print(
                        f"pw.x attempt-{attempt - 1} failed in {run_dir} "
                        f"({reason}); retrying ({retries_done}/"
                        f"{self.config.max_retries})",
                        flush=True,
                    )
                continue
            record = self.last_attempt_records[-1]
            record.update(status="success", error=None, retryable=None)
            self._last_density_dir = run_dir / f"attempt-{attempt}"
            return result

    def _resolve_density(self, atoms: Atoms) -> DensitySource | None:
        """Pick a compatible known-origin density, or decide atomic up front."""
        candidates: list[tuple[str, Path]] = []
        if self.config.density_source is not None:
            candidates.append(("config.density_source", Path(self.config.density_source)))
        if self._last_density_dir is not None:
            candidates.append(("previous attempt", self._last_density_dir))
        reasons: list[str] = []
        for label, origin in candidates:
            source, reason = load_density_source(origin, engine=self, atoms=atoms)
            if source is not None:
                self.last_density_decision = {
                    "start": "density", "origin": str(source.origin_dir), "via": label,
                }
                return source
            reasons.append(f"{label} ({origin}): {reason}")
        self.last_density_decision = {
            "start": "atomic",
            "reason": "; ".join(reasons) if reasons else "no density source configured",
        }
        return None

    def _attempt(self, atoms: Atoms, attempt_dir: Path, config: QeConfig, *,
                 density: DensitySource | None = None) -> EngineResult:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        record: dict = {
            "attempt": len(self.last_attempt_records) + 1,
            "directory": str(attempt_dir),
            "start": "density" if density is not None else "atomic",
            "status": "running",
            "error": None,
            "retryable": None,
        }
        self.last_attempt_records.append(record)

        copy_elapsed_s = 0.0
        copy_bytes = 0
        if density is not None:
            copy_elapsed_s, copy_bytes = self._stage_density(density, attempt_dir)
            record["density_from"] = str(density.origin_dir)
            record["density_copy_s"] = copy_elapsed_s
            record["density_copy_bytes"] = copy_bytes

        # Absolute paths: the subprocess runs with cwd=attempt_dir, so a
        # relative run_root would otherwise stop resolving.
        in_path = (attempt_dir / "pw.in").resolve()
        out_path = (attempt_dir / "pw.out").resolve()
        write_qe_input(in_path, atoms, config)

        t0 = time.perf_counter()
        with out_path.open("w") as fh:
            proc = subprocess.Popen(
                [*config.pw_cmd, "-in", str(in_path)],
                cwd=attempt_dir,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group: killable as a unit
            )
            try:
                proc.wait(timeout=config.timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_process_group(proc)
                raise QeEngineError(
                    f"pw.x timed out after {config.timeout_s:.0f}s in {attempt_dir}; "
                    "process group killed",
                    retryable=True,
                ) from None
        wall = time.perf_counter() - t0
        text = out_path.read_text(errors="replace")
        if proc.returncode != 0:
            raise QeEngineError(
                f"pw.x exited with code {proc.returncode} in {attempt_dir}; tail:\n"
                + "\n".join(text.splitlines()[-15:]),
                # A QE error banner is a deterministic input/physics problem;
                # anything else (segfault, launcher failure) may be transient.
                retryable=_QE_ERROR_BANNER_RE.search(text) is None,
            )
        if "convergence NOT achieved" in text:
            # Identical input converges identically: rerunning only burns cost.
            raise QeEngineError(f"pw.x SCF did not converge in {attempt_dir}")
        if "JOB DONE." not in text:
            # Exit 0 without the completion marker means the output is
            # truncated (killed job, full disk): not a valid label.
            raise QeEngineError(
                f"pw.x in {attempt_dir} exited cleanly but has no JOB DONE. "
                "marker; output incomplete",
                retryable=True,
            )
        try:
            result = parse_qe_output(text)
        except EngineError as error:
            raise QeEngineError(
                f"pw.x output in {attempt_dir} did not parse completely: {error}",
                retryable=True,
            ) from error
        if result.forces.shape != (len(atoms), 3):
            raise QeEngineError(
                f"parsed force shape {result.forces.shape} != ({len(atoms)}, 3)"
            )

        write_density_manifest(
            attempt_dir, engine=self, atoms=atoms,
            source=(
                {"kind": "atomic"} if density is None else {
                    "kind": "copied",
                    "from": str(density.origin_dir),
                    "from_fingerprint": density.manifest.get("reference_fingerprint"),
                    "copy_s": copy_elapsed_s,
                    "copy_bytes": copy_bytes,
                }
            ),
        )
        record["wall_time_s"] = wall
        return EngineResult(
            energy=result.energy,
            forces=result.forces,
            stress=result.stress,
            wall_time_s=wall,
            energy_kind=self.capabilities.energy_kind,
            force_consistent=True,
        )

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """SIGKILL the attempt's whole process group (mpirun ranks included),
        then reap; no orphaned children survive a timeout."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX fallback
                proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass

    def _stage_density(self, density: DensitySource,
                       attempt_dir: Path) -> tuple[float, int]:
        """Copy the verified .save tree into this attempt's writable outdir.

        The source stays read-only by convention: two computations never share
        one writable .save. Returns (elapsed_s, bytes_copied); the copy is
        recorded as an io task event when an event log is attached.
        """
        destination = attempt_dir / "tmp" / f"{QE_PREFIX}.save"
        destination.parent.mkdir(parents=True, exist_ok=True)
        started_unix = time.time()
        t0 = time.perf_counter()
        shutil.copytree(density.save_dir, destination)
        elapsed = time.perf_counter() - t0
        nbytes = sum(
            f.stat().st_size for f in destination.rglob("*") if f.is_file()
        )
        self._io_counter += 1
        if self._event_log is not None:
            self._event_log.append(
                "task",
                {
                    "task_id": f"qe-io-{self._io_counter}",
                    "attempt": 1,
                    "operation": "io",
                    "purpose": "density_copy",
                    "status": "success",
                    "evaluation_id": None,
                    "started_unix": started_unix,
                    "elapsed_s": elapsed,
                    "cpu_cores": None,
                    "gpu": None,
                    "queue_s": None,
                    "source": "qe-engine",
                    "provenance": {
                        "from": str(density.origin_dir),
                        "from_fingerprint": density.manifest.get("reference_fingerprint"),
                        "to": str(destination),
                        "bytes": nbytes,
                    },
                },
            )
        return elapsed, nbytes


def create_qe_engine(*, run_root: str | Path, event_log: object | None = None,
                     **config_kwargs) -> QeEngine:
    """Registry factory: build a QeEngine from plain keyword settings."""
    return QeEngine(QeConfig(**config_kwargs), run_root, event_log=event_log)
