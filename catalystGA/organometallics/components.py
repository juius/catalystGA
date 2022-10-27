import math
from typing import List

from rdkit import Chem
from rdkit.Chem.rdMolHash import HashFunction, MolHash
from rdkit.Chem import rdChemReactions, rdDistGeom

CARBENE = "#6&v2H0"
PHOSPHINE = "#15&v3"
AMINE = "#7&v3"
OXYGEN = "#8&v2"
DONORS = [CARBENE, PHOSPHINE, AMINE, OXYGEN]


class BaseCatalyst:
    """Base Class for Metal-Organic Catalysts"""

    n_ligands = None

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.metal = metal
        self.ligands = ligands
        self.score = math.nan
        self.error = ""
        self.health_check()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metal},{self.ligands})"

    def __hash__(self) -> int:
        return hash(self.smiles)

    def __eq__(self, other):
        if isinstance(other, BaseCatalyst):
            if self.__hash__() == other.__hash__():
                return True
        return False

    @property
    def mol(self):
        return self.assemble()

    @property
    def smiles(self):
        self.assemble()
        return MolHash(self.mol, HashFunction.CanonicalSmiles)

    def health_check(self):
        pass

    def assemble(self, extraLigands=None, chiralTag=None, permutationOrder=None):
        """Forms dative bonds from Ligands to Metal Center, adds extra Ligands from Reaction SMARTS and sets the chiral tag of the metal center and permutation order of the Ligands

        Args:
            extraLigands (str, optional): Reaction SMARTS to add ligands to the molecule. Defaults to None.
            chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom. Defaults to None.
            permutationOrder (int, optional): Permutation order of ligands. Defaults to None.

        Returns:
            Chem.Mol: Catalyst Molecule
        """
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
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        # Add Extra Ligands
        if extraLigands:
            rxn = rdChemReactions.ReactionFromSmarts(extraLigands)
            mol = rxn.RunReactants((mol,))[0][0]
        # Set Chiral Tag and Psermutation Order
        if chiralTag:
            metal = mol.GetAtomWithIdx(mol.GetSubstructMatch(self.metal.atom)[0])
            self._setChiralTagAndOrder(metal, chiralTag, permutationOrder)
        return mol

    @staticmethod
    def _setChiralTagAndOrder(atom, chiralTag, chiralPermutation=None):
        """Sets the chiral tag of an atom and the permutation order of attached ligands

        Args:
            atom (Chem.Atom): Atom for which to set the chiral tag/permutation order properties
            chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom. Defaults to None.
            permutationOrder (int, optional): Permutation order of ligands. Defaults to None.
        """
        atom.SetChiralTag(chiralTag)
        if chiralPermutation:
            atom.SetIntProp("_chiralPermutation", chiralPermutation)

    def embed(
        self,
        extraLigands=None,
        chiralTag=None,
        permutationOrder=None,
        numConfs=10,
        useRandomCoords=True,
        pruneRmsThresh=-1,
        **kwargs,
    ):
        """Embed the Catalyst Molecule using ETKDG

        Args:
            extraLigands (str, optional): Reaction SMARTS to add ligands to the molecule. Defaults to None.
            chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom. Defaults to None.
            permutationOrder (int, optional): Permutation order of ligands. Defaults to None.
            numConfs (int, optional): Number of Conformers to embed. Defaults to 10.
            useRandomCoords (bool, optional): Embedding option. Defaults to True.
            pruneRmsThresh (int, optional): Conformers within this threshold will be removed. Defaults to -1.

        Returns:
            Chem.Mol: Catalyst Molecule with conformers embedded
        """
        mol3d = self.assemble(extraLigands, chiralTag, permutationOrder)
        Chem.SanitizeMol(mol3d)
        mol3d = Chem.AddHs(mol3d)
        # Embed with ETKDG
        cids = rdDistGeom.EmbedMultipleConfs(
            mol3d,
            numConfs=numConfs,
            useRandomCoords=useRandomCoords,
            pruneRmsThresh=pruneRmsThresh,
            **kwargs,
        )
        return mol3d


carben = Chem.MolFromSmarts("[C;v2-0]")
phosphor = Chem.MolFromSmarts("[P,p;v3]")
nitrogen = Chem.MolFromSmarts("[N,n;v3]")
oxygen = Chem.MolFromSmarts("[O,o;v2]")
priority = [carben, phosphor, nitrogen, oxygen]


class Ligand:
    """Organic Ligands"""

    def __init__(self, mol, donor_id=None, fixed=False):
        self.mol = mol
        if not donor_id:
            donor_id = self._find_donor_atom()
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

    def _find_donor_atom(self):
        # REDO THIS PART
        for p in priority:
            match = self.mol.GetSubstructMatch(p)
            if len(match) > 0:
                return match[0]

        raise Warning(f"No donor atom found for Ligand {Chem.MolToSmiles(self.mol)}")

    def set_positions(self, positions):
        if self.fixed:
            self.positions = positions


class Metal:
    """Transition Metal"""

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
