"""Contract FD verification: QE metallic free_energy vs forces (Al slab).

The refined contract (INDEPENDENT_REVIEW_20260909 section 5) allows a
cross-kind combination only when each side's reported scalar is consistent
with its forces.  For the QE metallic reference this script checks the
correspondence directly: the variational free energy reported on the "!"
line must differentiate to the reported forces.  Two internal directions
(free-layer atoms 8 and 16; the fixed bottom layer is not displaced),
central difference h = 1e-3 A, one residual per direction:

    residual_i = |(E(+h u_i) - E(-h u_i)) / (2h) + F0 . u_i|

Preset tolerance (fixed BEFORE running, not adjusted after):
    residual_i <= 5e-4 eV/A per direction.

Justification: conv_thr = 1e-8 Ry puts the total-energy noise near
1.4e-7 eV, i.e. FD noise ~1.4e-4 eV/A at h = 1e-3 A; truncation at this
step on a smooth slab potential is well below that.  5e-4 eV/A leaves
~3x headroom over the estimated noise without being vacuous.

SCF budget: 1 base + 4 displaced = 5 executions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from pyraimd2.engines.qe_engine import QeConfig, QeEngine

TOLERANCE_EV_A = 5e-4
H_A = 1e-3

work = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
atoms = ase_read(work / "structure.extxyz")
config = QeConfig(
    pseudo_dir="/tmp/ps",
    pw_cmd=("mpirun", "-np", "28", "pw.x"),
    xc="pbe",
    dispersion="grimme-d3",
    ecutwfc=50.0,
    ecutrho=400.0,
    kpts=(4, 4, 1),
    metallic=True,
    smearing="mv",
    degauss=0.02,
    conv_thr=1e-8,
    pseudos={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
)
engine = QeEngine(config, run_root=work / "fd_runs")

base = engine.compute(atoms)
forces = np.asarray(base.forces, dtype=float)
report = {
    "engine_name": engine.name,
    "fingerprint": engine.fingerprint,
    "energy_kind": str(engine.capabilities.energy_kind),
    "force_consistent": engine.capabilities.force_consistent,
    "base_energy_eV": float(base.energy),
    "h_A": H_A,
    "tolerance_eV_A": TOLERANCE_EV_A,
    "directions": [],
    "n_scf": 1,
}
for atom_index, axis in ((8, 0), (16, 1)):
    direction = np.zeros_like(atoms.positions)
    direction[atom_index, axis] = 1.0
    energies = []
    for sign in (1.0, -1.0):
        probe = atoms.copy()
        probe.positions = atoms.positions + sign * H_A * direction
        energies.append(float(engine.compute(probe).energy))
        report["n_scf"] += 1
    fd = (energies[0] - energies[1]) / (2.0 * H_A)
    residual = abs(fd + float(np.sum(forces * direction)))
    report["directions"].append({
        "atom": atom_index, "axis": axis,
        "fd_eV_A": fd, "minus_F_dot_u_eV_A": float(-np.sum(forces * direction)),
        "residual_eV_A": residual,
        "passed": bool(residual <= TOLERANCE_EV_A),
    })

report["passed"] = all(d["passed"] for d in report["directions"])
out = work / "contract_fd_result.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2, sort_keys=True))
sys.exit(0 if report["passed"] else 1)
