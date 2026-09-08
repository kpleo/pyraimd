"""Build a Li(100)|LP30-electrolyte dual-interface box.

Default structure:

- Li(100) slab, 3x3 surface x 4 layers = 36 Li (a = 3.51 A, experimental room-
  temperature bcc value), bottom layer fixed downstream, two interfaces per slab
- electrolyte: 21 EC + 17 DMC + 3 LiPF6 (438 atoms, LP30-class: 1 M LiPF6 in
  EC:DMC 1:1 v/v) at liquid-region density 1.28 g/cm3 (Dinh-Nguyen et al.,
  J. Electrochem. Soc. 2016, DOI 10.1149/2.0771605jes)
- box ~10.5 x 10.5 x ~52 A (lateral exactly 3*a = 10.53 A), PBC everywhere
- packing: Packmol, tolerance 2.0 A (see arXiv:2606.09422), seeded. Classical-FF NPT
  pre-equilibration and the 450 K
  AIMD seeding (Leung & Budzien PCCP 2010; Leung et al. JPCC 2016) happen
  downstream — this script only produces the packed starting structure.

Molecular geometries are embedded below (no RDKit dependency): they are the
git-tracked RDKit ETKDGv3 + UFF minimized packing geometries from
hpc/neimeng/scripts/build_molecules.py (seed 20250819), identical to the ones
used for the bulk LP30 box. They are chemically reasonable and will be relaxed
by the classical/DFT pre-equilibration.

The Li(100) slab is stacked manually (ABAB registry, layer spacing a/2); the
stacking was verified to reproduce ase.build.bcc100("Li", a=3.51, size=(3,3,4))
positions exactly, so no ASE dependency either — numpy only. Packmol is the
single external tool (reviewer-expected standard); everything else in this file
is pure Python and unit-tested without packmol.

Every parameter is a CLI argument; a JSON sidecar next to the output records
all parameters, provenance, and validation numbers (project constitution:
seeded, restart-complete). Conventions follow hpc/neimeng/scripts/make_w_spike.py.

Usage:
  python build_interface_box.py --out interface_li100_lp30_474.xyz \
      --packmol /path/to/packmol [--seed 1] [--density 1.28] \
      [--n-ec 21] [--n-dmc 17] [--n-lipf6 3] [--surface 3] [--layers 4] \
      [--li-a 3.51] [--interface-gap 1.0] [--tolerance 2.0] [--nloop 500] \
      [--lz None] [--workdir None]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Embedded molecular templates (packing geometries; see module docstring).
# Element order matches hpc/neimeng/inputs/molecules/{ec,dmc,pf6,li}.xyz.
# ---------------------------------------------------------------------------

EC_SYMBOLS = ["O", "C", "O", "C", "C", "O", "H", "H", "H", "H"]
EC_POSITIONS = np.array(
    [
        [2.74535828, -0.10559422, 0.31827318],
        [1.49656363, -0.05756197, 0.17349792],
        [0.78557999, 1.11228524, 0.05425638],
        [-0.59205674, 0.79657153, -0.09357091],
        [-0.65640278, -0.74855211, -0.05116452],
        [0.69057315, -1.16906208, 0.11687626],
        [-0.96498892, 1.18313348, -1.06599839],
        [-1.17242992, 1.24141245, 0.74273476],
        [-1.06237158, -1.15545592, -1.00181792],
        [-1.26982509, -1.09717639, 0.80691324],
    ]
)

DMC_SYMBOLS = ["O", "C", "O", "C", "O", "C", "H", "H", "H", "H", "H", "H"]
DMC_POSITIONS = np.array(
    [
        [0.00473474, -0.08015805, -1.25787661],
        [-0.00009649, 0.00163383, 0.00210112],
        [-1.20627451, 0.03396791, 0.70582062],
        [-2.47247347, -0.02204878, 0.05969986],
        [1.20065622, 0.06114981, 0.71328522],
        [2.47178010, 0.03378805, 0.07503352],
        [-3.27448855, 0.01876796, 0.82498439],
        [-2.56699877, -0.97055545, -0.51036258],
        [-2.58710838, 0.84242508, -0.62813383],
        [2.57112547, 0.90067693, -0.61213844],
        [2.59123618, -0.91230290, -0.49436401],
        [3.26790746, 0.09265563, 0.84527430],
    ]
)

PF6_SYMBOLS = ["F", "P", "F", "F", "F", "F", "F"]
PF6_POSITIONS = np.array(
    [
        [1.26159810, -1.07586042, -0.23123812],
        [0.00000005, 0.00000023, -0.00000013],
        [-1.26159831, 1.07586000, 0.23123800],
        [0.91834514, 0.83552018, 1.12299824],
        [0.60629069, 0.97314310, -1.21982790],
        [-0.91834523, -0.83551958, -1.12299799],
        [-0.60629044, -0.97314351, 1.21982789],
    ]
)

LI_ION_SYMBOLS = ["Li"]
LI_ION_POSITIONS = np.array([[0.0, 0.0, 0.0]])

TEMPLATES: dict[str, tuple[list[str], np.ndarray]] = {
    "ec": (EC_SYMBOLS, EC_POSITIONS),
    "dmc": (DMC_SYMBOLS, DMC_POSITIONS),
    "li_ion": (LI_ION_SYMBOLS, LI_ION_POSITIONS),
    "pf6": (PF6_SYMBOLS, PF6_POSITIONS),
}

# Molecular formulas of the templates, used for element bookkeeping.
FORMULAS: dict[str, dict[str, int]] = {
    "ec": {"C": 3, "H": 4, "O": 3},
    "dmc": {"C": 3, "H": 6, "O": 3},
    "li_ion": {"Li": 1},
    "pf6": {"P": 1, "F": 6},
}

# CIAAW conventional atomic weights (g/mol == amu per atom).
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
    "F": 18.998403163,
    "P": 30.973761998,
    "Li": 6.94,
}

AMU_TO_G = 1.66053906660e-24  # g per atomic mass unit
AVOGADRO = 6.02214076e23

# Hard validation thresholds (the density tolerance is a CLI-side check).
MIN_INTERMOLECULAR_A = 1.5  # spec: no intermolecular pair closer than this
MIN_SLAB_SOLVENT_A = 1.5  # spec: slab Li to nearest solvent atom
MAX_NN_INTRAMOLECULAR_A = 1.9  # every atom keeps a neighbor -> molecule intact
MIN_NN_INTRAMOLECULAR_A = 0.9  # no crushed intramolecular pair
MAX_SIDE_MASS_IMBALANCE = 0.10  # dual-interface symmetry (integer molecules)

# Species packing order; the odd molecule of alternating species goes to the
# opposite side so the two liquid slabs stay mass-balanced.
SPECIES_ORDER = ["ec", "dmc", "li_ion", "pf6"]

PROVENANCE = {
    "spec": "Li(100) 3x3 x 4 layers "
    "= 36 Li + 21 EC + 17 DMC + 3 LiPF6 = 474 atoms, dual interface, PBC",
    "li_lattice_a": "3.51 A, experimental room-temperature bcc Li value "
    "(CRC Handbook of Chemistry and Physics). "
    "Li(100) orientation precedents: Leung et al. JPCC 2016 "
    "(DOI 10.1021/acs.jpcc.5b11719), Kundu et al. arXiv:2509.14067, "
    "DP-QEq Nat. Commun. 16:7379 (2025). Four slab layers are used here. "
    "Slab-thickness convergence is a separate calculation.",
    "density": "1.28 g/cm3 for 1 M LiPF6 in EC:DMC 1:1, Dinh-Nguyen et al., "
    "J. Electrochem. Soc. 2016, DOI 10.1149/2.0771605jes",
    "molecular_geometries": "RDKit ETKDGv3 embed + UFF minimize, git-tracked at "
    "hpc/neimeng/inputs/molecules/*.xyz (hpc/neimeng/scripts/build_molecules.py, "
    "seed 20250819) — the same packing geometries as the bulk LP30 box; measured "
    "template bond lengths are recorded under template_bonds_A. UFF bond lengths "
    "run slightly long vs experiment (carbonate C=O ~1.21-1.23 A, PF6- P-F "
    "~1.6 A; cf. Allen et al., J. Chem. Soc. Perkin Trans. II 1987 bond tables) "
    "and are relaxed by the downstream classical/DFT pre-equilibration",
    "packing": "Packmol, tolerance 2.0 A ("
    "see arXiv:2606.09422); slab as fixed structure, solvent constrained "
    "to the two liquid regions inset by pack_margin from the periodic cell "
    "faces (packmol is not PBC-aware; same inset convention as "
    "hpc/neimeng/inputs/packmol_bulk.inp)",
    "composition_rationale": "LP30-class 1 M LiPF6 in EC:DMC 1:1 v/v; EC:DMC "
    "mole ratio 21:17 = 1.235 vs ideal ~1.26 from pure-solvent densities; "
    "salt molarity at the target density is ~1.0 M (recorded below)",
    "downstream_protocol": "classical-FF bulk NPT pre-equilibration to measured "
    "density (UFF/Forcite: Leung 2016; APPLE&P: Borodin & Smith JPCB 110, 4971 "
    "(2006)), then NVT 300 K dt=0.5 fs + DP-QEq anisotropic ConstQ stage; AIMD "
    "seeding at 450 K (Leung & Budzien PCCP 2010; Leung et al. JPCC 2016 — "
    "'to avoid EC freezing')",
}


@dataclass
class BoxGeometry:
    """Orthorhombic dual-interface cell layout along z.

    [liquid bottom: 0 .. t_side] [gap] [slab: span] [gap] [liquid top: t_side]
    """

    lx: float
    ly: float
    lz: float
    slab_span: float
    gap: float
    t_side: float
    z_slab_bottom: float
    z_slab_top: float

    @property
    def region_bottom(self) -> tuple[float, float]:
        return (0.0, self.t_side)

    @property
    def region_top(self) -> tuple[float, float]:
        return (self.lz - self.t_side, self.lz)

    @property
    def cell(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.lz])

    @property
    def liquid_volume_a3(self) -> float:
        return 2.0 * self.t_side * self.lx * self.ly


@dataclass
class BuildPlan:
    composition: dict[str, int]  # species -> molecule count
    surface: int
    layers: int
    li_a: float
    density_target: float
    gap: float
    pack_margin: float
    tolerance: float
    seed: int
    nloop: int
    geometry: BoxGeometry
    # (species, count, region) blocks in packmol structure order.
    region_split: list[tuple[str, int, str]]


def count_elements(composition: dict[str, int], surface: int, layers: int) -> dict[str, int]:
    """Total element counts: slab Li plus all solvent molecules."""
    counts: dict[str, int] = {"Li": surface * surface * layers}
    for species, n in composition.items():
        if n == 0:
            continue
        for element, k in FORMULAS[species].items():
            counts[element] = counts.get(element, 0) + k * n
    return counts


def electrolyte_mass_amu(composition: dict[str, int]) -> float:
    return sum(
        n * sum(ATOMIC_MASSES[e] * k for e, k in FORMULAS[species].items())
        for species, n in composition.items()
    )


def liquid_density_g_cm3(mass_amu: float, volume_a3: float) -> float:
    return mass_amu * AMU_TO_G / (volume_a3 * 1.0e-24)


def salt_molarity(n_salt: int, volume_a3: float) -> float:
    return n_salt / (AVOGADRO * volume_a3 * 1.0e-27)


def slab_positions(a: float, surface: int, layers: int, z_bottom: float) -> np.ndarray:
    """Manual Li(100) stacking: ABAB registry, layer spacing a/2, in-plane
    spacing a; consecutive layers shifted by (a/2, a/2). Equivalent to
    ase.build.bcc100("Li", a=a, size=(surface, surface, layers)) up to a
    uniform lateral shift of (a/4, a/4) that centers the atom columns in the
    surface cell (maximizes the clearance to the lateral cell faces, which
    matters because packmol is not PBC-aware)."""
    pos = []
    for k in range(layers):
        shift = a / 4.0 + (a / 2.0 if (k % 2) else 0.0)
        for i in range(surface):
            for j in range(surface):
                pos.append([i * a + shift, j * a + shift, z_bottom + k * a / 2.0])
    return np.array(pos, dtype=float)


def make_plan(
    n_ec: int = 21,
    n_dmc: int = 17,
    n_lipf6: int = 3,
    surface: int = 3,
    layers: int = 4,
    li_a: float = 3.51,
    density_target: float = 1.28,
    gap: float = 1.0,
    pack_margin: float = 1.0,
    tolerance: float = 2.0,
    seed: int = 1,
    nloop: int = 500,
    lz: float | None = None,
) -> BuildPlan:
    composition = {"ec": n_ec, "dmc": n_dmc, "li_ion": n_lipf6, "pf6": n_lipf6}
    lateral = surface * li_a
    span = (layers - 1) * li_a / 2.0
    area = lateral * lateral
    mass = electrolyte_mass_amu(composition)
    if lz is None:
        # Size the liquid region so its density equals the target exactly.
        # rho = mass*AMU_TO_G / (V*1e-24)  ->  V = mass*AMU_TO_G / (rho*1e-24)
        volume_total = mass * AMU_TO_G / (density_target * 1.0e-24)  # A^3
        t_side = volume_total / area / 2.0
        lz = span + 2.0 * gap + 2.0 * t_side
    else:
        t_side = (lz - span - 2.0 * gap) / 2.0
        if t_side <= 0.0:
            raise ValueError(f"lz={lz} leaves no room for liquid (span+2*gap={span + 2 * gap:.3f})")
    geometry = BoxGeometry(
        lx=lateral,
        ly=lateral,
        lz=lz,
        slab_span=span,
        gap=gap,
        t_side=t_side,
        z_slab_bottom=t_side + gap,
        z_slab_top=t_side + gap + span,
    )
    region_split: list[tuple[str, int, str]] = []
    for i, species in enumerate(SPECIES_ORDER):
        n = composition[species]
        bottom = (n + 1) // 2 if i % 2 == 0 else n // 2
        region_split.append((species, bottom, "bottom"))
        region_split.append((species, n - bottom, "top"))
    return BuildPlan(
        composition=composition,
        surface=surface,
        layers=layers,
        li_a=li_a,
        density_target=density_target,
        gap=gap,
        pack_margin=pack_margin,
        tolerance=tolerance,
        seed=seed,
        nloop=nloop,
        geometry=geometry,
        region_split=region_split,
    )


def pack_region(plan: BuildPlan, region: str) -> tuple[float, ...]:
    """Atom-confinement box for packmol. Inset by pack_margin from every cell
    face that is periodic: packmol is not PBC-aware, so atoms at the region
    edges would otherwise clash through the cell boundaries (precedent:
    hpc/neimeng/inputs/packmol_bulk.inp packs into an inset sub-cube for the
    same reason). With margin m on both sides of every seam the cross-boundary
    clearance is >= 2m; the inner z edge stays at the accounting boundary
    (clearance to the slab is enforced by packmol's tolerance in-box)."""
    g = plan.geometry
    m = plan.pack_margin
    if region == "bottom":
        z0, z1 = m, g.t_side
    elif region == "top":
        z0, z1 = g.lz - g.t_side, g.lz - m
    else:
        raise ValueError(f"unknown region {region}")
    return (m, m, z0, g.lx - m, g.ly - m, z1)


def render_packmol_input(plan: BuildPlan) -> str:
    """Pure string generation — unit-tested without running packmol."""
    lines = [
        "# Li(100)|LP30 dual-interface box",
        "# generated by experiments/build_interface_box.py -- do not edit",
        f"tolerance {plan.tolerance:.1f}",
        "filetype xyz",
        "output packed.xyz",
        f"seed {plan.seed}",
        f"nloop {plan.nloop}",
        "",
        "structure slab.xyz",
        "  number 1",
        "  fixed 0. 0. 0. 0. 0. 0.",
        "end structure",
    ]
    for species, count, region in plan.region_split:
        if count == 0:
            continue
        x0, y0, z0, x1, y1, z1 = pack_region(plan, region)
        lines += [
            "",
            f"structure {species}.xyz",
            f"  number {count}",
            f"  inside box {x0:.6f} {y0:.6f} {z0:.6f} {x1:.6f} {y1:.6f} {z1:.6f}",
            "end structure",
        ]
    return "\n".join(lines) + "\n"


def write_xyz(path: Path, symbols: list[str], positions: np.ndarray, comment: str = "") -> None:
    lines = [f"{len(symbols)}\n", f"{comment}\n"]
    for s, (x, y, z) in zip(symbols, positions):
        lines.append(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")
    path.write_text("".join(lines))


def write_extxyz(
    path: Path, symbols: list[str], positions: np.ndarray, cell: np.ndarray, comment: str
) -> None:
    lattice = " ".join(
        f"{v:.6f}"
        for v in [cell[0], 0.0, 0.0, 0.0, cell[1], 0.0, 0.0, 0.0, cell[2]]
    )
    header = (
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T" {comment}\n'
    )
    lines = [f"{len(symbols)}\n", header]
    for s, (x, y, z) in zip(symbols, positions):
        lines.append(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")
    path.write_text("".join(lines))


def parse_xyz_text(text: str) -> tuple[list[str], np.ndarray]:
    lines = text.strip().splitlines()
    n = int(lines[0].split()[0])
    symbols, positions = [], []
    for line in lines[2 : 2 + n]:
        fields = line.split()
        symbols.append(fields[0])
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if len(symbols) != n:
        raise ValueError(f"xyz header says {n} atoms but {len(symbols)} lines parsed")
    return symbols, np.array(positions, dtype=float)


def run_packmol(executable: str, workdir: Path, input_path: Path) -> str:
    """Run packmol in workdir and return its stdout. stdin is the input
    *file*, not a pipe: packmol rewinds unit 5 while reading (setsizes.f90)
    and dies with 'Illegal seek' on a non-seekable pipe. Only this function
    touches the external binary."""
    with open(input_path) as fh:
        proc = subprocess.run(
            [executable],
            stdin=fh,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"packmol exited with code {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def minimum_image_distances(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Full pairwise minimum-image distance matrix for an orthorhombic cell."""
    delta = positions[None, :, :] - positions[:, None, :]
    delta -= cell[None, None, :] * np.round(delta / cell[None, None, :])
    return np.linalg.norm(delta, axis=2)


def check_overlaps(
    positions: np.ndarray,
    cell: np.ndarray,
    mol_ids: np.ndarray,
    cutoff: float = MIN_INTERMOLECULAR_A,
) -> dict:
    """Intermolecular overlap check under PBC. Returns the minimum
    intermolecular distance, the number of pairs below cutoff, and the worst
    pair indices."""
    dist = minimum_image_distances(positions, cell)
    different_mol = mol_ids[None, :] != mol_ids[:, None]
    iu = np.triu_indices(len(positions), k=1)
    d = dist[iu]
    mask = different_mol[iu]
    d_inter = d[mask]
    pairs = int((mask & (d < cutoff)).sum())
    if d_inter.size == 0:
        return {
            "min_intermolecular_A": float("inf"),
            "n_pairs_below_cutoff": 0,
            "cutoff_A": cutoff,
            "closest_pair": None,
        }
    k = int(np.argmin(d_inter))
    ij = (int(iu[0][mask][k]), int(iu[1][mask][k]))
    return {
        "min_intermolecular_A": float(d_inter.min()),
        "n_pairs_below_cutoff": pairs,
        "cutoff_A": cutoff,
        "closest_pair": ij,
    }


def slab_solvent_distance(
    positions: np.ndarray, cell: np.ndarray, mol_ids: np.ndarray, slab_mol_id: int = 0
) -> float:
    dist = minimum_image_distances(positions, cell)
    cross = (mol_ids[None, :] == slab_mol_id) & (mol_ids[:, None] > slab_mol_id)
    return float(dist[cross].min())


def intramolecular_nn_stats(
    positions: np.ndarray, mol_ids: np.ndarray, slab_mol_id: int = 0
) -> dict:
    """Per-molecule nearest-neighbor distances: min over molecules of the
    smallest NN distance, and max over molecules of the largest NN distance
    (every atom must retain one close neighbor -> molecule survived packing).
    The slab (mol_id 0) is excluded — its crystal NN is checked separately."""
    nn_min, nn_max = np.inf, 0.0
    per_mol = {}
    for m in np.unique(mol_ids):
        if m == slab_mol_id:
            continue
        idx = np.flatnonzero(mol_ids == m)
        if len(idx) == 1:
            per_mol[int(m)] = None  # single-atom ion
            continue
        p = positions[idx]
        d = np.linalg.norm(p[None, :, :] - p[:, None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        per_mol[int(m)] = {"min": float(nn.min()), "max": float(nn.max())}
        nn_min = min(nn_min, float(nn.min()))
        nn_max = max(nn_max, float(nn.max()))
    return {"nn_min_A": nn_min, "nn_max_A": nn_max, "per_molecule": per_mol}


def template_bonds() -> dict:
    """Measured nearest-neighbor bond lengths of the embedded templates —
    recorded in the sidecar as the geometry-provenance numbers."""
    out = {}
    for name, (symbols, pos) in TEMPLATES.items():
        if len(symbols) == 1:
            out[name] = None
            continue
        d = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        pairs = {}
        for i in range(len(symbols)):
            j = int(np.argmin(d[i]))
            key = "-".join(sorted([symbols[i], symbols[j]]))
            pairs.setdefault(key, []).append(round(float(d[i, j]), 4))
        out[name] = {k: sorted(set(v)) for k, v in sorted(pairs.items())}
    return out


def molecule_block_map(plan: BuildPlan) -> tuple[np.ndarray, np.ndarray]:
    """mol_ids and region labels for every atom of a packed structure that
    follows the packmol structure order (slab first, then region blocks)."""
    mol_ids = [0] * (plan.surface * plan.surface * plan.layers)
    regions = ["slab"] * len(mol_ids)
    next_mol = 1
    for species, count, region in plan.region_split:
        n_atoms = len(TEMPLATES[species][0])
        for _ in range(count):
            mol_ids += [next_mol] * n_atoms
            regions += [region] * n_atoms
            next_mol += 1
    return np.array(mol_ids), np.array(regions)


def expected_symbols(plan: BuildPlan) -> list[str]:
    symbols = ["Li"] * (plan.surface * plan.surface * plan.layers)
    for species, count, _ in plan.region_split:
        symbols += TEMPLATES[species][0] * count
    return symbols


def validate(
    symbols: list[str],
    positions: np.ndarray,
    plan: BuildPlan,
    mol_ids: np.ndarray,
    regions: np.ndarray,
) -> dict:
    """The full acceptance checklist; returns numbers plus pass/fail."""
    g = plan.geometry
    cell = g.cell
    failures: list[str] = []

    # 1. element counts vs molecular formulas
    expected = count_elements(plan.composition, plan.surface, plan.layers)
    actual: dict[str, int] = {}
    for s in symbols:
        actual[s] = actual.get(s, 0) + 1
    if actual != expected:
        failures.append(f"element counts {actual} != expected {expected}")

    # 2. intramolecular integrity (packmol only translates/rotates); the slab
    # crystal NN must stay at a*sqrt(3)/2.
    nn = intramolecular_nn_stats(positions, mol_ids)
    if nn["nn_min_A"] < MIN_NN_INTRAMOLECULAR_A:
        failures.append(f"crushed intramolecular pair: {nn['nn_min_A']:.3f} A")
    if nn["nn_max_A"] > MAX_NN_INTRAMOLECULAR_A:
        failures.append(f"broken molecule: max NN distance {nn['nn_max_A']:.3f} A")
    slab_idx = np.flatnonzero(mol_ids == 0)
    sp = positions[slab_idx]
    sd = minimum_image_distances(sp, cell)
    np.fill_diagonal(sd, np.inf)
    slab_nn = float(sd.min())
    slab_nn_expected = plan.li_a * np.sqrt(3.0) / 2.0
    if abs(slab_nn - slab_nn_expected) > 1e-3:
        failures.append(f"slab NN {slab_nn:.4f} A != bcc NN {slab_nn_expected:.4f} A")

    # 3. intermolecular overlaps and slab clearance
    ov = check_overlaps(positions, cell, mol_ids)
    if ov["n_pairs_below_cutoff"] > 0:
        failures.append(
            f"{ov['n_pairs_below_cutoff']} intermolecular pairs below "
            f"{ov['cutoff_A']} A (min {ov['min_intermolecular_A']:.3f} A)"
        )
    d_slab = slab_solvent_distance(positions, cell, mol_ids)
    if d_slab < MIN_SLAB_SOLVENT_A:
        failures.append(f"slab-solvent distance {d_slab:.3f} A < {MIN_SLAB_SOLVENT_A}")

    # 4. liquid-region density and box dimensions
    volume = g.liquid_volume_a3
    density = liquid_density_g_cm3(electrolyte_mass_amu(plan.composition), volume)
    molarity = salt_molarity(plan.composition["pf6"], volume)
    if abs(density - plan.density_target) > 0.05:
        failures.append(f"liquid density {density:.4f} off target {plan.density_target}")

    # 5. dual-interface symmetry: equal construction thickness, mass balance
    mass = {r: 0.0 for r in ("bottom", "top")}
    for species, count, region in plan.region_split:
        m = count * sum(ATOMIC_MASSES[e] * k for e, k in FORMULAS[species].items())
        mass[region] += m
    imbalance = abs(mass["bottom"] - mass["top"]) / (mass["bottom"] + mass["top"])
    if imbalance > MAX_SIDE_MASS_IMBALANCE:
        failures.append(f"side mass imbalance {imbalance:.3f} > {MAX_SIDE_MASS_IMBALANCE}")
    z_extents = {}
    for region, (z0, z1) in (("bottom", g.region_bottom), ("top", g.region_top)):
        idx = np.flatnonzero(regions == region)
        z_extents[region] = {
            "region_A": [round(z0, 4), round(z1, 4)],
            "occupied_z_A": [round(float(positions[idx, 2].min()), 4), round(float(positions[idx, 2].max()), 4)],
        }

    return {
        "n_atoms": len(symbols),
        "element_counts": actual,
        "expected_element_counts": expected,
        "box_A": {"lx": g.lx, "ly": g.ly, "lz": g.lz},
        "slab_span_A": g.slab_span,
        "liquid_thickness_per_side_A": g.t_side,
        "interface_gap_A": g.gap,
        "intramolecular_nn_min_A": round(nn["nn_min_A"], 4),
        "intramolecular_nn_max_A": round(nn["nn_max_A"], 4),
        "slab_nn_min_A": round(slab_nn, 4),
        "min_intermolecular_A": round(ov["min_intermolecular_A"], 4),
        "n_pairs_below_cutoff": ov["n_pairs_below_cutoff"],
        "min_slab_solvent_A": round(d_slab, 4),
        "liquid_density_g_cm3": round(density, 5),
        "salt_molarity_mol_L": round(molarity, 4),
        "side_mass_amu": {k: round(v, 3) for k, v in mass.items()},
        "side_mass_imbalance": round(imbalance, 4),
        "solvent_z_extents": z_extents,
        "pass": not failures,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path, help="output xyz path; the JSON sidecar is written next to it")
    p.add_argument("--packmol", default="packmol", help="packmol executable (on Neimeng A: $PROJECT/envs/packmol/bin/packmol)")
    p.add_argument("--n-ec", type=int, default=21)
    p.add_argument("--n-dmc", type=int, default=17)
    p.add_argument("--n-lipf6", type=int, default=3)
    p.add_argument("--surface", type=int, default=3, help="NxN surface supercell")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--li-a", type=float, default=3.51, help="bcc Li lattice constant, Angstrom")
    p.add_argument("--density", type=float, default=1.28, help="target liquid-region density, g/cm3")
    p.add_argument("--interface-gap", type=float, default=1.0, help="solvent-center standoff from the outermost Li planes, Angstrom")
    p.add_argument("--pack-margin", type=float, default=1.0, help="inset of packmol confinement boxes from periodic cell faces, Angstrom")
    p.add_argument("--tolerance", type=float, default=2.0, help="packmol tolerance, Angstrom")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nloop", type=int, default=500, help="packmol nloop")
    p.add_argument("--lz", type=float, default=None, help="override box z length (density is then recomputed, not targeted)")
    p.add_argument("--workdir", type=Path, default=None, help="keep the packmol scratch directory here instead of a tempdir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan = make_plan(
        n_ec=args.n_ec,
        n_dmc=args.n_dmc,
        n_lipf6=args.n_lipf6,
        surface=args.surface,
        layers=args.layers,
        li_a=args.li_a,
        density_target=args.density,
        gap=args.interface_gap,
        pack_margin=args.pack_margin,
        tolerance=args.tolerance,
        seed=args.seed,
        nloop=args.nloop,
        lz=args.lz,
    )
    g = plan.geometry
    n_total = sum(count_elements(plan.composition, plan.surface, plan.layers).values())
    print(
        f"plan: {n_total} atoms, box {g.lx:.3f} x {g.ly:.3f} x {g.lz:.3f} A, "
        f"liquid 2 x {g.t_side:.3f} A at {plan.density_target} g/cm3, "
        f"gap {g.gap} A, seed {plan.seed}"
    )

    scratch = args.workdir or Path(tempfile.mkdtemp(prefix="packmol_if_"))
    scratch.mkdir(parents=True, exist_ok=True)

    slab_pos = slab_positions(plan.li_a, plan.surface, plan.layers, g.z_slab_bottom)
    write_xyz(scratch / "slab.xyz", ["Li"] * len(slab_pos), slab_pos, "Li(100) slab (fixed)")
    for species, (symbols, pos) in TEMPLATES.items():
        write_xyz(scratch / f"{species}.xyz", symbols, pos, f"{species} packing geometry")

    packmol_inp = render_packmol_input(plan)
    inp_path = scratch / "packmol.inp"
    inp_path.write_text(packmol_inp)
    stdout = run_packmol(args.packmol, scratch, inp_path)
    success = "Success!" in stdout
    print(f"packmol finished (success flag: {success}); scratch: {scratch}")
    if not success:
        print("WARNING: packmol did not report 'Success!' — check constraints", file=sys.stderr)

    symbols, positions = parse_xyz_text((scratch / "packed.xyz").read_text())
    expected = expected_symbols(plan)
    if symbols != expected:
        raise RuntimeError("packmol output atom order does not match the structure plan")
    mol_ids, regions = molecule_block_map(plan)

    # Wrap into the primary cell and write the final extended xyz.
    positions = positions % g.cell
    args.out.parent.mkdir(parents=True, exist_ok=True)
    comment = (
        "Li(100)|LP30 dual interface; packed with packmol "
        f"tol={plan.tolerance} seed={plan.seed}"
    )
    write_extxyz(args.out, symbols, positions, g.cell, comment)

    result = validate(symbols, positions, plan, mol_ids, regions)
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "builder": "experiments/build_interface_box.py",
                "provenance": PROVENANCE,
                "parameters": {
                    "composition": plan.composition,
                    "surface": plan.surface,
                    "layers": plan.layers,
                    "li_a_A": plan.li_a,
                    "density_target_g_cm3": plan.density_target,
                    "interface_gap_A": plan.gap,
                    "pack_margin_A": plan.pack_margin,
                    "packmol_tolerance_A": plan.tolerance,
                    "packmol_nloop": plan.nloop,
                    "lz_override_A": args.lz,
                    "seed": plan.seed,
                },
                "geometry": asdict(g),
                "region_split": plan.region_split,
                "template_bonds_A": template_bonds(),
                "packmol": {
                    "executable": args.packmol,
                    "input": packmol_inp,
                    "reported_success": success,
                },
                "validation": result,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} + {sidecar}")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(f"validation FAILED: {result['failures']}")


if __name__ == "__main__":
    main()
