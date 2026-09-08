"""Quantum ESPRESSO engine: periodic DFT labels via pw.x.

The engine is a swappable
label source behind the ``Engine`` protocol; subprocess exit codes are always
checked and failures raise :class:`EngineError` — never silently reuse stale
output. Each call gets its own directory ``run_root/<label>-NNNNNN`` and each
execution a numbered ``attempt-N`` subdirectory, so consecutive evaluations
never clobber one another and a failed attempt keeps its diagnostics. The
input path handed to pw.x is absolute — a relative ``run_root`` must not
become invalid once the subprocess changes its working directory.

Input is written directly (simple, fixed namelist set — PBE + Grimme D3,
``ibrav=0`` with explicit cell) rather than through ASE's espresso writer, to
keep the format under our control. Output is parsed from the pw.x stdout text:
the ``! total energy`` (Ry) of the last complete SCF block, the ``Forces
acting on atoms`` block (Ry/bohr) that follows it, and the 3x3 stress block
(Ry/bohr^3) after that — energy, forces and stress always come from the same
block, never mixed across blocks. Fortran ``D`` exponents are converted.

Stress sign: pw.x reports stress with compression-positive convention; we
convert to the ASE convention (``stress_ase = -stress_qe``) and voigt order
(xx, yy, zz, yz, xz, xy). The sign is asserted against ASE's own espresso
calculator in the remote integration test before any NPT use — NVT-only until
then.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
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
    """Everything a QeEngine call needs; immutable, explicit (no globals)."""

    pseudo_dir: str
    pw_cmd: tuple[str, ...] = ("pw.x",)  # e.g. ("mpirun", "-np", "28", "pw.x")
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
    startpot_file: bool = False  # resume from <outdir>/<prefix>.save charge density
    timeout_s: float = 3600.0
    pseudos: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PSEUDOS))


def _species(atoms: Atoms) -> list[str]:
    seen: dict[str, None] = {}
    for sym in atoms.get_chemical_symbols():
        seen.setdefault(sym)
    return list(seen)


def write_qe_input(path: Path, atoms: Atoms, cfg: QeConfig) -> None:
    """Write a PBE+D3 pw.x scf input for ``atoms`` (ibrav=0, explicit cell)."""
    species = _species(atoms)
    missing = [s for s in species if s not in cfg.pseudos]
    if missing:
        raise EngineError(f"no pseudopotential configured for species: {missing}")
    if abs(np.linalg.det(atoms.cell)) < 1e-8:
        raise EngineError("QeEngine requires a non-singular cell (ibrav=0)")

    lines: list[str] = []
    lines.append("&CONTROL\n")
    lines.append("  calculation = 'scf'\n  prefix = 'pyraimd2'\n")
    lines.append(f"  pseudo_dir = '{cfg.pseudo_dir}'\n  outdir = './tmp'\n")
    lines.append("  tprnfor = .true.\n  tstress = .true.\n/\n")
    lines.append("&SYSTEM\n  ibrav = 0\n")
    lines.append(f"  nat = {len(atoms)}\n  ntyp = {len(species)}\n")
    lines.append(f"  ecutwfc = {cfg.ecutwfc}\n  ecutrho = {cfg.ecutrho}\n")
    if cfg.nbnd is not None:
        lines.append(f"  nbnd = {cfg.nbnd}\n")
    lines.append("  vdw_corr = 'grimme-d3'\n")
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


class QeEngine:
    """Periodic PBE+D3 labels from Quantum ESPRESSO pw.x (Engine protocol)."""

    name = "qe-pbe-d3"

    def __init__(self, config: QeConfig, run_root: str | Path) -> None:
        self.config = config
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._call_counter = 0

    @property
    def capabilities(self) -> EngineCapabilities:
        # Converged pw.x forces differentiate the reported total energy;
        # stress is always requested (tstress) and parsed when printed.
        return EngineCapabilities(
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
            forces_conservative=True,
            stress_available=True,
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(dataclasses.asdict(self.config), sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"qe-pbe-d3:{digest}"

    def compute(self, atoms: Atoms, label: str | None = None) -> EngineResult:
        base = "eval" if label is None else str(label).replace("/", "_")
        run_dir = self.run_root / f"{base}-{self._call_counter:06d}"
        self._call_counter += 1
        try:
            return self._attempt(atoms, run_dir / "attempt-1", self.config)
        except EngineError:
            if not self.config.startpot_file:
                raise
        # Chained-density start failed (stale/corrupt .save after a killed
        # run, or a frame too far from the last converged one): retry once
        # with an explicit atomic-start input. The failed attempt keeps its
        # directory and output for diagnosis — nothing is wiped.
        atomic_config = dataclasses.replace(self.config, startpot_file=False)
        print(
            f"startpot chain failed in {run_dir}/attempt-1; "
            "retrying with an atomic-start input",
            flush=True,
        )
        return self._attempt(atoms, run_dir / "attempt-2", atomic_config)

    def _attempt(self, atoms: Atoms, attempt_dir: Path, config: QeConfig) -> EngineResult:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        # Absolute paths: the subprocess runs with cwd=attempt_dir, so a
        # relative run_root would otherwise stop resolving.
        in_path = (attempt_dir / "pw.in").resolve()
        out_path = (attempt_dir / "pw.out").resolve()
        write_qe_input(in_path, atoms, config)

        t0 = time.perf_counter()
        try:
            with out_path.open("w") as fh:
                proc = subprocess.run(
                    [*config.pw_cmd, "-in", str(in_path)],
                    cwd=attempt_dir,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=config.timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                f"pw.x timed out after {config.timeout_s:.0f}s in {attempt_dir}"
            ) from exc
        wall = time.perf_counter() - t0
        text = out_path.read_text(errors="replace")
        if proc.returncode != 0:
            raise EngineError(
                f"pw.x exited with code {proc.returncode} in {attempt_dir}; tail:\n"
                + "\n".join(text.splitlines()[-15:])
            )
        if "convergence NOT achieved" in text:
            raise EngineError(f"pw.x SCF did not converge in {attempt_dir}")
        result = parse_qe_output(text)
        if result.forces.shape != (len(atoms), 3):
            raise EngineError(
                f"parsed force shape {result.forces.shape} != ({len(atoms)}, 3)"
            )
        return EngineResult(
            energy=result.energy,
            forces=result.forces,
            stress=result.stress,
            wall_time_s=wall,
            energy_kind=EnergyKind.ENERGY,
            force_consistent=True,
        )
