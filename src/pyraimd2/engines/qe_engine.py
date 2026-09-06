"""Quantum ESPRESSO engine (M3): periodic DFT labels via pw.x.

Design-doc contract (design-phase-b.md §3, §4): the engine is a swappable
label source behind the ``Engine`` protocol; subprocess exit codes are always
checked and failures raise :class:`EngineError` — never silently reuse stale
output. Each call runs in its own directory ``run_root/label`` (no CWD
coupling, no clobbering of other runs).

Input is written directly (simple, fixed namelist set — PBE + Grimme D3,
``ibrav=0`` with explicit cell) rather than through ASE's espresso writer, to
keep the format under our control. Output is parsed from the pw.x stdout text:
``! total energy`` (Ry), the ``Forces acting on atoms`` block (Ry/bohr), and
the 3x3 stress block (Ry/bohr^3).

Stress sign: pw.x reports stress with compression-positive convention; we
convert to the ASE convention (``stress_ase = -stress_qe``) and voigt order
(xx, yy, zz, yz, xz, xy). The sign is asserted against ASE's own espresso
calculator in the remote integration test before any NPT use — NVT-only until
then.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, chemical_symbols

from pyraimd2.engines.base import EngineError, EngineResult

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
_FORCE_RE = re.compile(
    r"^\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
    r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$",
    re.MULTILINE,
)


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
    """Parse pw.x stdout: total energy (Ry), forces (Ry/bohr), stress (Ry/bohr^3)."""
    energies = _ENERGY_RE.findall(text)
    if not energies:
        raise EngineError("pw.x output has no '! total energy' line (SCF never finished?)")
    energy_ev = float(energies[-1]) * RY_EV

    forces_ry_bohr = np.array([tuple(map(float, m)) for m in _FORCE_RE.findall(text)])
    if forces_ry_bohr.size == 0:
        raise EngineError("pw.x output has no force block (tprnfor missing?)")
    forces = forces_ry_bohr * (RY_EV / units.Bohr)

    stress = None
    stress_marker = "total   stress  (Ry/bohr**3)"
    idx = text.find(stress_marker)
    if idx != -1:
        rows: list[list[float]] = []
        for line in text[idx:].splitlines()[1:4]:
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) == 3:
            s = np.array(rows)  # QE compression-positive, Ry/bohr^3
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

    def compute(self, atoms: Atoms, label: str = "step") -> EngineResult:
        run_dir = self.run_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._attempt(atoms, run_dir)
        except EngineError:
            if not self.config.startpot_file:
                raise
            # Chained-density restart failed (stale/corrupt .save after a
            # killed run, or a frame too far from the last converged one):
            # wipe the saved density and retry once from scratch — the same
            # wipe-on-failure semantics as the bootstrap labeling chain.
            shutil.rmtree(run_dir / "tmp", ignore_errors=True)
            print(
                f"startpot chain failed in {run_dir}; density wiped, "
                "retrying from scratch",
                flush=True,
            )
            return self._attempt(atoms, run_dir)

    def _attempt(self, atoms: Atoms, run_dir: Path) -> EngineResult:
        in_path = run_dir / "pw.in"
        out_path = run_dir / "pw.out"
        write_qe_input(in_path, atoms, self.config)

        t0 = time.perf_counter()
        try:
            with out_path.open("w") as fh:
                proc = subprocess.run(
                    [*self.config.pw_cmd, "-in", str(in_path)],
                    cwd=run_dir,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                f"pw.x timed out after {self.config.timeout_s:.0f}s in {run_dir}"
            ) from exc
        wall = time.perf_counter() - t0
        text = out_path.read_text(errors="replace")
        if proc.returncode != 0:
            raise EngineError(
                f"pw.x exited with code {proc.returncode} in {run_dir}; tail:\n"
                + "\n".join(text.splitlines()[-15:])
            )
        if "convergence NOT achieved" in text:
            raise EngineError(f"pw.x SCF did not converge in {run_dir}")
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
        )
