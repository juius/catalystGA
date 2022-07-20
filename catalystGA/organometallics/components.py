from typing import List

from rdkit import Chem
from rdkit.Chem.rdMolHash import HashFunction, MolHash


class BaseCatalyst:

    n_ligands = None

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.metal = metal
        self.ligands = ligands
        self.health_check()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metal},{self.ligands})"

    def __hash__(self) -> int:
        return hash(
            ",".join(
                [MolHash(self.metal.atom, HashFunction.CanonicalSmiles)]
                + [MolHash(lig.mol, HashFunction.CanonicalSmiles) for lig in self.ligands]
            )
        )

    def __eq__(self, other):
        if isinstance(other, BaseCatalyst):
            if self.__hash__() == other.__hash__():
                return True
        return False

    def health_check(self):
        pass
        # assert self.n_ligands == len(
        #     self.ligands
        # ), f"Wrong number of ligands. Got {len(self.ligands)}, expected {self.n_ligands}"

    def assemble(self):
        # Initialize Mol
        tmp = self.metal.atom
        # Add ligands
        for ligand in self.ligands:
            tmp = Chem.CombineMols(tmp, ligand.mol)
        # Attach ligands to central atom
        emol = Chem.EditableMol(tmp)
        atom_ids = Chem.GetMolFrags(tmp)
        self.donor_ids = []
        for i, ligand in enumerate(self.ligands):
            # Get donor id in combined mol
            donor_id = atom_ids[i + 1][ligand.donor_id]
            self.donor_ids.append(donor_id)
            # Add Bond to Central Atom
            emol.AddBond(donor_id, 0, Chem.BondType.DATIVE)
        self.mol = emol.GetMol()
        Chem.SanitizeMol(self.mol)

    def embed(self, STemplate, CGenerator, constrain_all=True):
        # Embed one conformer and set positions
        assert len(self.ligands) == len(STemplate.dummy_ligands), (
            f"Mismatch between number of ligands in catalyst and template: "
            f"{len(self.ligands)} vs {len(STemplate.dummy_ligands)}"
        )
        mol3d = Chem.AddHs(self.mol)
        assert (
            Chem.rdDistGeom.EmbedMolecule(mol3d, ETversion=2, useRandomCoords=True) == 0
        ), "Initial embedding failed"
        conf = mol3d.GetConformer()
        # Set position of central atom
        conf.SetAtomPosition(self.metal.atom.GetAtoms()[0].GetIdx(), STemplate.central_atom)
        # Set position of donor atoms
        for i, point in enumerate(STemplate.dummy_ligands):
            conf.SetAtomPosition(self.donor_ids[i], point)

        # Set fixed ligands
        fixed_atoms = []
        if STemplate.fixed_ligands:
            for ligand in STemplate.fixed_ligands:
                tmp_last_idx = mol3d.GetNumAtoms()
                mol3d = Chem.CombineMols(mol3d, ligand.mol)
                # set positions
                conf = mol3d.GetConformer()
                for i, atom in enumerate(ligand.mol.GetAtoms()):
                    new_id = tmp_last_idx + atom.GetIdx()
                    conf.SetAtomPosition(new_id, ligand.positions[i])
                    fixed_atoms.append(new_id)
                # set bond to central atom
                emol = Chem.EditableMol(mol3d)
                emol.AddBond(
                    int(ligand.donor_id + tmp_last_idx),
                    0,
                    Chem.BondType.DATIVE,
                )
                mol3d = emol.GetMol()
        # Set constraints
        if constrain_all:
            constrained_atoms = (
                [self.metal.atom.GetAtoms()[0].GetIdx()] + self.donor_ids + fixed_atoms
            )
        else:
            constrained_atoms = None
        Chem.SanitizeMol(mol3d)
        # Generate more conformers
        return CGenerator.generate(mol3d, constrain_atoms=constrained_atoms)


class Ligand:
    def __init__(self, mol, donor_id=None, fixed=False):
        self.mol = mol
        if not donor_id:
            donor_id = self.find_donor_atom()
        self.donor_id = donor_id
        self.fixed = fixed

    def __repr__(self):
        return f"{MolHash(self.mol, HashFunction.CanonicalSmiles)}"

    def __hash__(self) -> int:
        return hash(MolHash(self.mol, HashFunction.CanonicalSmiles))

    def __eq__(self, other):
        if isinstance(other, Ligand):
            if self.__hash__() == other.__hash__():
                return True
        return False

    @classmethod
    def from_smiles(cls, smiles: str, donor_id=None):
        mol = Chem.MolFromSmiles(smiles)
        return cls(mol, donor_id)

    def find_donor_atom(self):
        # REDO THIS PART
        for atype in ["P", "N", "C", "Cl", "Br"]:
            match = self.mol.GetSubstructMatch(Chem.MolFromSmarts(f"[{atype}]"))
            if len(match) > 0:
                return match[0]
        raise Warning(f"No donor atom found for Ligand {Chem.MolToSmiles(self.mol)}")

    def set_positions(self, positions):
        if self.fixed:
            self.positions = positions


class Metal:
    def __init__(self, atom, coordination_number=None):
        if isinstance(atom, str):
            self.atom = Chem.MolFromSmiles(f"[{atom}]")
        elif isinstance(atom, Chem.Atom):
            self.atom = Chem.MolFromSmiles(f"[{atom.GetSymbol()}]")
        elif isinstance(atom, Chem.Mol):
            self.atom = atom
        else:
            raise TypeError(f"Invalid type for atom: {type(atom)}")

        self.coordination_number = coordination_number

    def __repr__(self):
        return f"{self.atom.GetAtoms()[0].GetSymbol()}"
