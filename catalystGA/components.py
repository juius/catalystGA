import logging
import math
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDistGeom
from rdkit.Chem.rdMolHash import HashFunction, MolHash
from .xtb_new import ac2mol
from .xtb_new import xtb_calculate

CARBENE = "#6&v2H0"
PHOSPHINE = "#15&v3"
AMINE = "#7&v3"
OXYGEN = "#8&v2"
CO = "C-,v5"
DONORS = [CARBENE, PHOSPHINE, AMINE, OXYGEN, CO]
priority = [Chem.MolFromSmarts("[" + pattern + "]") for pattern in DONORS]


class BaseCatalyst:
    """Base Class for Metal-Organic Catalysts."""

    n_ligands = None
    save_attributes = {}

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.metal = metal
        self.ligands = ligands
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

    @property
    def mol(self):
        return self.assemble()

    @property
    def smiles(self):
        self.assemble()
        return MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)

    def health_check(self):
        pass

    def assemble(self, extraLigands=None, chiralTag=None, permutationOrder=None):
        """Forms dative bonds from Ligands to Metal Center, adds extra Ligands
        from Reaction SMARTS and sets the chiral tag of the metal center and
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
        # Set Chiral Tag and Psermutation Order
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


class Ligand:
    """Organic Ligands."""

    def __init__(self, mol, donor_id=None, fixed=False):
        self.mol = mol
        if not donor_id:
            self.find_donor_atom()
        else:
            self.donor_id = donor_id
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
    def from_smiles(cls, smiles: str, donor_id=None):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        Chem.SanitizeMol(mol)
        return cls(mol, donor_id)

    def find_donor_atom(
        self, smarts_match=True, reference_smiles="[Mo]<-N", n_cores=1, calc_dir="."
    ):
        if smarts_match:
            donor_id = None
            for p in priority:
                match = self.mol.GetSubstructMatch(p)
                if len(match) > 0:
                    donor_id = match[0]
                    break
            if not isinstance(donor_id, int):
                raise Warning(
                    f"No donor atom found for Ligand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            # Find all possible donor atoms
            pattern = Chem.MolFromSmarts("[" + ",".join(DONORS) + "]")
            matches = self.mol.GetSubstructMatches(pattern)

            if len(matches) == 0:
                raise ValueError("No donor atoms found in ligand.")
            elif len(matches) == 1:
                donor_id = matches[0][0]
            else:
                # Make all possible constitutional isomers
                _logger.info(f"Found {len(matches)} possible donor atoms in ligand.")
                _logger.info(
                    "Generating all possible constitutional isomers and calculating binding energies."
                )
                binding_energies = []
                reference_mol = Chem.AddHs(Chem.MolFromSmiles(reference_smiles))
                central_id = reference_mol.GetSubstructMatch(Chem.MolFromSmarts("[Mo]"))[0]
                for match in matches:
                    donor_id = match[0]
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
                        numConfs=50,
                        useRandomCoords=True,
                        pruneRmsThresh=0.1,
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
                        results.append((donor_id, sp_energy))
                        _logger.info(
                            f"{('{:>10}{:>10}{:>25}{:>25}').format(donor_id, cid, round(ff_energy, 4), round(sp_energy, 4))}"
                        )

                    if len(results) == 0:
                        results.append((donor_id, np.nan))

                    results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
                    binding_energies.append(results[0])
                binding_energies.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])

                _logger.info("\n\nBinding energies:")
                _logger.info(
                    ("{:>12}{:>12}{:>27}").format(
                        "Donor ID", "Atom Type", "Binding Energy [Hartree]"
                    )
                )
                for donor_id, energy in binding_energies:
                    _logger.info(
                        ("{:>12}{:>12}{:>27}").format(
                            donor_id,
                            self.mol.GetAtomWithIdx(donor_id).GetSymbol(),
                            round(energy, 4),
                        )
                    )

                donor_id = binding_energies[0][0]

                _logger.info(
                    f"\nDonor atom: {donor_id} ({self.mol.GetAtomWithIdx(donor_id).GetSymbol()})"
                )
        self.donor_id = donor_id

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
