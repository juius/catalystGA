import logging
import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDistGeom
from rdkit.Chem.rdMolHash import HashFunction, MolHash

from catalystGA.xtb import ac2mol, xtb_calculate

TRANSITION_METALS = (
    "[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg]"
)

###  Dative bonds  ###

CARBENE = "#6&v2H0"
PHOSPHINE = "#15&v3"
AMINE = "#7&v3"
OXYGEN = "#8&v2"
CO = "C-,v5"
DONORS_dative = [CARBENE, PHOSPHINE, AMINE, OXYGEN, CO]
priority_dative = [Chem.MolFromSmarts("[" + pattern + "]") for pattern in DONORS_dative]

###  Covalent bonds  ###

# Halogens
HALOGENS = "#9,#17,#35"
# Hydroxide
HYDROXIDE = "O;H1"
# SP3 hybridized carbon
SP3_CARBON = "#6X4;!H0"
# SP3 hybridized carbon
SP2_CARBON = "#6X3;!H0"
# Sulphur TODO
sulphur = None

DONORS_covalent = [HALOGENS, HYDROXIDE, SP3_CARBON, SP2_CARBON]


class BaseCatalyst:
    """Base Class for Metal-Organic Catalysts."""

    save_attributes = {}

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.metal = metal
        self.ligands = ligands
        self.n_ligands = len(ligands)
        self.score = math.nan
        self.fitness = math.nan
        self.error = ""
        self.idx = (-1, -1)
        self.timing = math.nan
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

    @classmethod
    def from_smiles(cls, smiles: str):
        """Create a catalyst from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol, "Could not parse SMILES string"

        # get transition metal
        metal_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(TRANSITION_METALS))
        assert len(metal_matches) > 0, "No transition metal found in molecule"
        assert len(metal_matches) < 2, "More than one transition metal found in molecule"
        metal_id = metal_matches[0][0]
        metal = Metal(mol.GetAtomWithIdx(metal_id).GetSymbol())

        # label donor atoms
        for atom in mol.GetAtomWithIdx(metal_id).GetNeighbors():
            atom.SetBoolProp("donor_atom", True)

        # fragment ligands
        fragments = Chem.FragmentOnBonds(
            mol,
            [bond.GetIdx() for bond in mol.GetAtomWithIdx(metal_id).GetBonds()],
            addDummies=False,
        )
        ligands = []
        for ligand_mol in Chem.GetMolFrags(fragments, asMols=True):
            if not ligand_mol.HasSubstructMatch(Chem.MolFromSmarts(TRANSITION_METALS)):
                # find donor atom
                for atom in ligand_mol.GetAtoms():
                    if atom.HasProp("donor_atom"):
                        connection_atom_id = atom.GetIdx()
                        break
                ligand_mol = Chem.AddHs(ligand_mol)
                Chem.SanitizeMol(ligand_mol)
                ligands.append(Ligand(ligand_mol, connection_atom_id=connection_atom_id))

        cat = cls(metal, ligands)
        assert Chem.MolToSmiles(cat.mol) == smiles, "SMILES string does not match input SMILES"
        return cat

    @property
    def mol(self):
        return self.assemble()

    @property
    def smiles(self):
        self.assemble()
        return MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)

    # TODO
    def health_check(self):
        pass

    def assemble(self, extraLigands=None, chiralTag=None, permutationOrder=None):
        """Forms bonds from Ligands to Metal Center, adds extra Ligands from
        Reaction SMARTS and sets the chiral tag of the metal center and
        permutation order of the Ligands.

        Args:
            extraLigands (str, optional): Reaction SMARTS to add ligands to the molecule. Defaults to None.
            chiralTag (Chem.rdchem.ChiralType, optional): Chiral Tag of Metal Atom. Defaults to None.
            permutationOrder (int, optional): Permutation order of ligands. Defaults to None.

        Returns:
            Chem.Mol: Catalyst Molecule
        """
        # Initialize Mol
        tmp = self.metal.atom

        # Add Extra Ligands
        if extraLigands:
            rxn = rdChemReactions.ReactionFromSmarts(extraLigands)
            tmp = rxn.RunReactants((tmp,))[0][0]

        # Add hydrogens
        Chem.SanitizeMol(tmp)
        tmp = Chem.AddHs(tmp)

        # Add ligands
        for ligand in self.ligands:
            tmp = Chem.CombineMols(tmp, ligand.mol)

        # Start editing mol
        emol = Chem.RWMol(tmp)
        emol.BeginBatchEdit()

        atom_ids = Chem.GetMolFrags(tmp)
        self.donor_idxs = []
        for i, ligand in enumerate(self.ligands):
            # Get donor id in combined mol
            connection_atom_id = atom_ids[i + 1][ligand.connection_atom_id]
            self.donor_idxs.append(connection_atom_id)
            # Add Bond to Central Atom
            emol.AddBond(connection_atom_id, 0, ligand.bond_type)

            # Remove halogen atom if the ligand is covalent and with halogen selected
            if isinstance(ligand, CovalentLigand):
                if ligand.pattern == HALOGENS:
                    halogen_idx = atom_ids[i + 1][ligand.halogen_idx]
                    emol.RemoveAtom(halogen_idx)
                # TODO NB! IF ATOMS ARE REMOVED THE DONOR IDS THAT ARE SAVED ARE WRONG

        # Commit changes made and get mol
        emol.CommitBatchEdit()
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)
        # Set Chiral Tag and Permutation Order
        if chiralTag:
            metal = mol.GetAtomWithIdx(mol.GetSubstructMatch(self.metal.atom)[0])
            self._setChiralTagAndOrder(metal, chiralTag, permutationOrder)
        Chem.SanitizeMol(mol)
        return mol

    @staticmethod
    def _setChiralTagAndOrder(atom, chiralTag, chiralPermutation=None):
        """Sets the chiral tag of an atom and the permutation order of attached
        ligands.

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
        """Embed the Catalyst Molecule using ETKDG.

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
        _ = rdDistGeom.EmbedMultipleConfs(
            mol3d,
            numConfs=numConfs,
            useRandomCoords=useRandomCoords,
            pruneRmsThresh=pruneRmsThresh,
            **kwargs,
        )
        return mol3d


_logger = logging.getLogger("ligand")


class Ligand(ABC):
    """Ligand base class."""

    def __init__(self, mol, connection_atom_id=None, fixed=False):
        self.mol = mol
        if not connection_atom_id:
            self.find_donor_atom(smarts_match=True)
        else:
            self.connection_atom_id = connection_atom_id
        self.fixed = fixed

    def __repr__(self):
        return f"{MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)}"

    def __hash__(self) -> int:
        return hash(MolHash(self.mol, HashFunction.CanonicalSmiles))

    def __eq__(self, other):
        if isinstance(other, Ligand):
            if self.__hash__() == other.__hash__():
                return True
        return False

    @classmethod
    def from_smiles(cls, smiles: str, connection_atom_id=None):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        Chem.SanitizeMol(mol)
        return cls(mol, connection_atom_id)

    @abstractmethod
    def find_donor_atom(
        self, smarts_match=True, reference_smiles="[Pd]<-P", n_cores=1, calc_dir="."
    ):
        pass

    def set_positions(self, positions):
        if self.fixed:
            self.positions = positions


class CovalentLigand(Ligand):
    """Covalently bound ligands."""

    def __init__(self, mol, connection_atom_id=None, fixed=False):
        super().__init__(mol=mol, connection_atom_id=connection_atom_id)
        self.bond_type = Chem.BondType.SINGLE

    def find_donor_atom(
        self, smarts_match=True, reference_smiles="[Mo]<-C", n_cores=1, calc_dir="."
    ):

        if smarts_match:
            connection_atom_id = None
            for pattern in DONORS_covalent:

                p = Chem.MolFromSmarts("[" + pattern + "]")
                match = self.mol.GetSubstructMatch(p)

                if len(match) > 0:
                    # Ensure that neighbors to HALOGENS are donors
                    if pattern == HALOGENS:
                        # Get Neighbor atom
                        neighbours = self.mol.GetAtomWithIdx(match[0]).GetNeighbors()
                        connection_atom_id = neighbours[0].GetIdx()

                        # Save halogen atom idx
                        self.halogen_idx = match[0]
                    else:
                        connection_atom_id = match[0]
                    # Set the type of donor atom that was found
                    self.pattern = pattern
                    break
            if not isinstance(connection_atom_id, int):
                raise Warning(
                    f"No donor atom found for Ligand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            raise NotImplementedError("Bonding site selection not implemented yet")

        # Set donor id on ligand
        self.connection_atom_id = connection_atom_id


class DativeLigand(Ligand):
    """Dative bound ligands."""

    def __init__(self, mol, connection_atom_id=None, fixed=False):
        super().__init__(mol=mol, connection_atom_id=connection_atom_id)
        self.bond_type = Chem.BondType.DATIVE

    def find_donor_atom(
        self, smarts_match=True, reference_smiles="[Pd]<-P", n_cores=1, calc_dir="."
    ):
        if smarts_match:
            connection_atom_id = None
            for p in priority_dative:
                match = self.mol.GetSubstructMatch(p)
                if len(match) > 0:
                    connection_atom_id = match[0]
                    break
            if not isinstance(connection_atom_id, int):
                raise Warning(
                    f"No donor atom found for Ligand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            # Find all possible donor atoms
            pattern = Chem.MolFromSmarts("[" + ",".join(DONORS_dative) + "]")
            matches = self.mol.GetSubstructMatches(pattern)

            if len(matches) == 0:
                raise ValueError("No donor atoms found in ligand.")
            elif len(matches) == 1:
                connection_atom_id = matches[0][0]
            else:
                # Make all possible constitutional isomers
                _logger.info(f"Found {len(matches)} possible donor atoms in ligand.")
                _logger.info(
                    "Generating all possible constitutional isomers and calculating binding energies."
                )
                binding_energies = []
                reference_mol = Chem.AddHs(Chem.MolFromSmiles(reference_smiles))
                central_id = reference_mol.GetSubstructMatch(
                    Chem.MolFromSmarts(TRANSITION_METALS)
                )[0]
                for match in matches:
                    connection_atom_id = match[0]
                    tmp = Chem.CombineMols(reference_mol, self.mol)
                    # Attach ligands to central atom
                    emol = Chem.EditableMol(tmp)
                    atom_ids = Chem.GetMolFrags(tmp)
                    # Get donor id in combined mol
                    comb_donor_id = atom_ids[1][match[0]]
                    # Add Bond to Central Atom
                    emol.AddBond(comb_donor_id, central_id, Chem.BondType.DATIVE)
                    mol = emol.GetMol()
                    Chem.SanitizeMol(mol)
                    metal = mol.GetAtomWithIdx(central_id)
                    metal.SetChiralTag(Chem.rdchem.ChiralType.CHI_SQUAREPLANAR)
                    metal.SetIntProp("_chiralPermutation", 2)
                    Chem.SanitizeMol(mol)
                    _logger.info(f"\nIsomer: {Chem.MolToSmiles(Chem.RemoveHs(mol))}")

                    # Embed test molecule
                    _ = rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=25,
                        useRandomCoords=True,
                        pruneRmsThresh=0.1,
                        randomSeed=42,
                    )

                    # Get adjacency matrix
                    adj = Chem.GetAdjacencyMatrix(mol)

                    # Find lowest energy conformer
                    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    results = []
                    _logger.info(
                        ("{:>10}{:>10}{:>25}{:>25}").format(
                            "Donor ID", "Conf-ID", "GFN-FF OPT [Hartree]", "GFN-2 SP [Hartree]"
                        )
                    )
                    for conf in mol.GetConformers():
                        cid = conf.GetId()
                        coords = conf.GetPositions()
                        # GFN-FF optimization
                        _, opt_coords, ff_energy = xtb_calculate(
                            atoms=atoms,
                            coords=coords,
                            options={"gfn": "ff", "opt": True},
                            scr=calc_dir,
                            n_cores=n_cores,
                        )
                        # Check adjacency matrix after optimization
                        opt_adj = Chem.GetAdjacencyMatrix(ac2mol(atoms, opt_coords))

                        if not np.array_equal(adj, opt_adj):
                            _logger.warning(
                                f"\tChange in adjacency matrix after GFN-FF optimization. Skipping conformer {cid}."
                            )
                            continue

                        # GFN-2 Single Point calculation
                        _, _, sp_energy = xtb_calculate(
                            atoms=atoms,
                            coords=opt_coords,
                            options={"gfn": 2},
                            scr=calc_dir,
                            n_cores=n_cores,
                        )
                        results.append((connection_atom_id, sp_energy))
                        _logger.info(
                            f"{('{:>10}{:>10}{:>25}{:>25}').format(connection_atom_id, cid, round(ff_energy, 4), round(sp_energy, 4))}"
                        )

                    if len(results) == 0:
                        results.append((connection_atom_id, np.nan))

                    results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
                    binding_energies.append(results[0])
                binding_energies.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])

                _logger.info("\n\nBinding energies:")
                _logger.info(
                    ("{:>12}{:>12}{:>27}").format(
                        "Donor ID", "Atom Type", "Binding Energy [Hartree]"
                    )
                )
                for connection_atom_id, energy in binding_energies:
                    _logger.info(
                        ("{:>12}{:>12}{:>27}").format(
                            connection_atom_id,
                            self.mol.GetAtomWithIdx(connection_atom_id).GetSymbol(),
                            round(energy, 4),
                        )
                    )

                connection_atom_id = binding_energies[0][0]

                _logger.info(
                    f"\nDonor atom: {connection_atom_id} ({self.mol.GetAtomWithIdx(connection_atom_id).GetSymbol()})"
                )
        self.connection_atom_id = connection_atom_id

    def set_positions(self, positions):
        if self.fixed:
            self.positions = positions


class Metal:
    """Transition Metal."""

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
