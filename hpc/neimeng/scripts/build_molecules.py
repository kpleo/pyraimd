"""Build the molecular components of an LP30-class electrolyte.

Generates MM/UFF-optimized gas-phase geometries with RDKit and writes them as
git-tracked xyz files under hpc/neimeng/inputs/molecules/:

- EC   (ethylene carbonate, C3H4O3), SMILES O=C1OCCO1
- DMC  (dimethyl carbonate, C3H6O3), SMILES O=C(OC)OC
- PF6- (hexafluorophosphate),      SMILES F[P-](F)(F)(F)(F)F
- Li+  (single atom)

These are only *packing* geometries — the DFT/MLIP equilibration relaxes them.

Usage:  uv run python hpc/neimeng/scripts/build_molecules.py
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

OUT = Path(__file__).parents[1] / "inputs" / "molecules"

SPECIES = {
    "ec": ("O=C1OCCO1", "ethylene carbonate"),
    "dmc": ("O=C(OC)OC", "dimethyl carbonate"),
    "pf6": ("F[P-](F)(F)(F)(F)F", "hexafluorophosphate"),
}


def build(smiles: str, name: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES for {name}: {smiles}")
    params = AllChem.ETKDGv3()
    params.randomSeed = 20250819
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"embedding failed for {name}")
    if AllChem.UFFOptimizeMolecule(mol, maxIters=500) != 0:
        raise RuntimeError(f"UFF optimization did not converge for {name}")
    return mol


def write_xyz(mol: Chem.Mol, path: Path, comment: str) -> None:
    conf = mol.GetConformer()
    lines = [f"{mol.GetNumAtoms()}\n", f"{comment}\n"]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}\n")
    path.write_text("".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (smiles, comment) in SPECIES.items():
        mol = build(smiles, name)
        write_xyz(mol, OUT / f"{name}.xyz", f"{comment} ({smiles}), UFF-optimized")
        print(f"{name}: {mol.GetNumAtoms()} atoms -> {OUT / (name + '.xyz')}")
    (OUT / "li.xyz").write_text("1\nlithium ion\nLi 0.0 0.0 0.0\n")
    print(f"li: 1 atom -> {OUT / 'li.xyz'}")


if __name__ == "__main__":
    main()
