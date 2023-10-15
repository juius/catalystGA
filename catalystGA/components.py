import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDistGeom
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolHash import HashFunction, MolHash

from catalystGA.utils import optimize
from catalystGA.xtb import ac2mol

TRANSITION_METALS = (
    "[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg]"
)

###  Dative bond patterns  ###

CARBENE = "#6&v2H0"
PHOSPHINE = "#15&v3"
AMINE = "#7&v3"
OXYGEN = "#8&v2"
CO = "C-,v5"
DONORS_dative = [CARBENE, PHOSPHINE, AMINE, OXYGEN, CO]
priority_dative = [Chem.MolFromSmarts("[" + pattern + "]") for pattern in DONORS_dative]

###  Covalent bond patterns ###

HALOGENS = "#9,#17,#35"
HYDROXIDE = "O;H1"
SECONDARY_AMINE = "#7X3;H1"
PRIMARY_AMINE = "#7X3;H2"
SP3_CARBON = "#6X4;!H0"
SP2_CARBON = "#6X3;!H0"

DONORS_covalent = [HYDROXIDE, SECONDARY_AMINE, PRIMARY_AMINE, SP3_CARBON, SP2_CARBON]
priority_covalent = [Chem.MolFromSmarts("[" + pattern + "]") for pattern in DONORS_covalent]


class BaseCatalyst:
    """Base Class for Metal-Organic Catalysts."""

    save_attributes = {}

    def __init__(self, metal: Chem.Mol, ligands: List) -> None:
        self.metal = metal
        self.ligands = ligands
        self.score = math.nan
        self.energy = math.nan
        self.fitness = math.nan
        self.error = ""
        self.idx = (-1, -1)
        self.timing = math.nan
        self.health_check()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metal},{self.ligands})"

    def __hash__(self) -> int:
        return hash(self.smiles)

    def __eq__(self, other) -> bool:
        if isinstance(other, BaseCatalyst):
            if self.__hash__() == other.__hash__():
                return True
        return False

    # https://stackoverflow.com/questions/10254594/what-makes-a-user-defined-class-unhashable

    @classmethod
    def from_smiles(cls, smiles: str):
        """Generate Catalyst from SMILES string. Requires a custom version of
        RDKit to work with dative bonds from atoms with unpaired electrons. See
        Github issue #6287 and pull request #6288.

        Args:
            smiles (str): SMILES string of Catalyst

        Returns:
            Instance of Catalyst Class
        """
        mol = Chem.MolFromSmiles(smiles)
        test_smiles = MolHash(Chem.RemoveHs(mol), HashFunction.CanonicalSmiles)
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
                Chem.SanitizeMol(ligand_mol)
                # find donor atom
                for atom in ligand_mol.GetAtoms():
                    if atom.HasProp("donor_atom"):
                        connection_atom_id = atom.GetIdx()
                        break
                ligand_mol = Chem.AddHs(ligand_mol)
                Chem.SanitizeMol(ligand_mol)
                ligands.append(Ligand(ligand_mol, connection_atom_id=connection_atom_id))

        cat = cls(metal, ligands)
        assert (
            cat.smiles == test_smiles
        ), f"SMILES string does not match input SMILES: {cat.smiles} != {test_smiles}"
        return cat

    @property
    def mol(self) -> Mol:
        return self.assemble()

    @property
    def smiles(self) -> str:
        self.assemble()
        return MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)

    # TODO
    def health_check(self) -> None:
        pass

    def assemble(
        self,
        extraLigands: None = None,
        chiralTag: None = None,
        permutationOrder: None = None,
    ) -> Mol:
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
        for i, ligand in enumerate(self.ligands):
            # Add bonds. If the ligand is bidentate, two bonds are added
            if isinstance(ligand, BidentateLigand):
                connection_atom_ids = [atom_ids[i + 1][id] for id in ligand.connection_atom_id]
                for id in connection_atom_ids:
                    emol.AddBond(id, 0, ligand.bond_type)
            else:
                connection_atom_id = ligand.connection_atom_id
                # If we have CovalentLigand, check if the connection is a halogen.
                if isinstance(ligand, CovalentLigand):
                    # Get neighbors to connection atom
                    neighbours = ligand.mol.GetAtomWithIdx(
                        ligand.connection_atom_id
                    ).GetNeighbors()
                    # Get the anumic nums of the neighbors
                    neighbours_idx = [n.GetIdx() for n in neighbours]
                    neighbours_atomid = [
                        ligand.mol.GetAtomWithIdx(n.GetIdx()).GetAtomicNum() for n in neighbours
                    ]
                    # Check the neighbors. If any halogen we remove it.
                    for atom_id, idx in zip(neighbours_atomid, neighbours_idx):
                        if atom_id in [9, 17, 35]:
                            halogen_idx = atom_ids[i + 1][idx]
                            emol.RemoveAtom(halogen_idx)
                            break
                # Add bond to metal.
                connection_atom_id = atom_ids[i + 1][connection_atom_id]
                emol.AddBond(connection_atom_id, 0, ligand.bond_type)

            # Remove any explicit hydrogens on the atom. Otherwise this hydrogen gives sanitation error.
            emol.GetAtomWithIdx(connection_atom_id).SetNumExplicitHs(0)

        # Commit changes made and get mol
        emol.CommitBatchEdit()
        mol = emol.GetMol()

        # Catch sanitation errors. NB! could lead to error later in workflow.
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _logger.warning("Sanitation error! Molecule: {mol}")
            _logger.warning(f"Traceback : {e}")

        # Set Chiral Tag and Permutation Order
        if chiralTag:
            metal = mol.GetAtomWithIdx(mol.GetSubstructMatch(self.metal.atom)[0])
            self._setChiralTagAndOrder(metal, chiralTag, permutationOrder)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _logger.warning("Sanitation error after chiral tag! Molecule: {mol}")
            _logger.warning(f"Traceback : {e}")
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

    def __init__(
        self,
        mol: Mol,
        connection_atom_id: None = None,
        fixed: bool = False,
        smarts_match: bool = True,
    ) -> None:
        self.mol = mol
        if not connection_atom_id:
            self.find_donor_atom(smarts_match=smarts_match)
        else:
            self.connection_atom_id = connection_atom_id
        self.fixed = fixed

    def __repr__(self):
        return f"{MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)}"

    def __hash__(self) -> int:
        return hash(MolHash(self.mol, HashFunction.CanonicalSmiles))

    def __eq__(self, other):
        # type(self) ensures that different child classes of this base class are not seen as equal
        if isinstance(other, type(self)):
            if self.__hash__() == other.__hash__():
                return True
        return False

    @property
    def smiles(self) -> str:
        return MolHash(Chem.RemoveHs(self.mol), HashFunction.CanonicalSmiles)

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

    def __init__(
        self,
        mol: Mol,
        connection_atom_id: int = None,
        fixed: bool = False,
        smarts_match: bool = True,
    ) -> None:
        super().__init__(mol=mol, connection_atom_id=connection_atom_id, smarts_match=smarts_match)
        self.bond_type = Chem.BondType.SINGLE

    def find_donor_atom(
        self,
        smarts_match: bool = True,
        reference_smiles: str = "[Mo]<-N#N",
        xtb_args=None,
        n_cores: int = 1,
        calc_dir=Path("."),
        numConfs: int = 20,
    ) -> None:
        if smarts_match:
            connection_atom_id = None
            for pattern in DONORS_covalent:
                p = Chem.MolFromSmarts("[" + pattern + "]")
                match = self.mol.GetSubstructMatch(p)

                if len(match) > 0:
                    # If the chosen pattern is halogen, set the connection id to the halogen neighbor.
                    if pattern == HALOGENS:
                        neighbours = self.mol.GetAtomWithIdx(match[0]).GetNeighbors()
                        connection_atom_id = neighbours[0].GetIdx()
                    else:
                        connection_atom_id = match[0]
                    break
            if not isinstance(connection_atom_id, int):
                _logger.warning(
                    f"No donor atom found for CovalentLigand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            # Ensure that the connection atom id is none if something fails.
            connection_atom_id = None

            # Find all possible donor atoms # TODO SOMETHING WITH THE SMARTS PATTERN MAKES THE MATCH FAIL IF DONE LIKE THE DATIVE LIGAND
            matches = ()
            type_match = []
            for elem in DONORS_covalent:
                pattern = Chem.MolFromSmarts("[" + elem + "]")
                if self.mol.GetSubstructMatches(pattern):
                    matches += self.mol.GetSubstructMatches(pattern)
                    type_match += [elem] * len(self.mol.GetSubstructMatches(pattern))

            if len(matches) == 0:
                _logger.warning("No donor atoms found in CovelentLigand")
            elif len(matches) == 1:
                # If the 1 match is a halogen set connection atom as neighbor.
                if HALOGENS in type_match:
                    neighbours = self.mol.GetAtomWithIdx(matches[0][0]).GetNeighbors()
                    connection_atom_id = neighbours[0].GetIdx()
                else:
                    connection_atom_id = matches[0][0]
            else:
                # Make all possible constitutional isomers
                _logger.info(f"Found {len(matches)} possible donor atoms in CovalentLigand.")
                _logger.info(
                    "Generating all possible constitutional isomers and calculating binding energies."
                )
                binding_energies = []
                reference_mol = Chem.AddHs(Chem.MolFromSmiles(reference_smiles))
                central_id = reference_mol.GetSubstructMatch(
                    Chem.MolFromSmarts(TRANSITION_METALS)
                )[0]
                for match, type in zip(matches, type_match):
                    # If halogen we need to find neighbor
                    if type == HALOGENS:
                        neighbours = self.mol.GetAtomWithIdx(match[0]).GetNeighbors()
                        connection_atom_id = neighbours[0].GetIdx()
                        halogen_idx = match[0]
                    else:
                        connection_atom_id = match[0]
                    tmp = Chem.CombineMols(reference_mol, self.mol)

                    # Attach ligands to central atom
                    emol = Chem.RWMol(tmp)
                    emol.BeginBatchEdit()
                    atom_ids = Chem.GetMolFrags(tmp)

                    # Get donor id in combined mol
                    comb_donor_id = atom_ids[1][connection_atom_id]

                    # Remove any explicit hydrogens on the atom. Otherwise this hydrogen gives sanitation error.
                    emol.GetAtomWithIdx(comb_donor_id).SetNumExplicitHs(0)

                    # Add Bond to Central Atom
                    emol.AddBond(comb_donor_id, central_id, Chem.BondType.SINGLE)

                    # Remove halogen atom.
                    if type == HALOGENS:
                        halogen_idx = atom_ids[1][halogen_idx]
                        emol.RemoveAtom(halogen_idx)
                        # TODO NB! IF ATOMS ARE REMOVED THE DONOR IDS THAT ARE SAVED ARE WRONG. SHOULD MAYBE JUST NOT SAVE THE IDS.

                    emol.CommitBatchEdit()
                    mol = emol.GetMol()

                    # Try sanitation
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception as e:
                        _logger.info(
                            f"Sanitation error for {self.mol} for match, type: {match, type}"
                        )
                        _logger.info(f"Traceback : {e}")
                        continue

                    metal = mol.GetAtomWithIdx(central_id)
                    metal.SetChiralTag(Chem.rdchem.ChiralType.CHI_SQUAREPLANAR)
                    metal.SetIntProp("_chiralPermutation", 2)
                    Chem.SanitizeMol(mol)
                    _logger.info(f"Isomer: {Chem.MolToSmiles(Chem.RemoveHs(mol))}")

                    # Embed test molecule
                    mol = Chem.AddHs(mol)
                    _ = rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=numConfs,
                        useRandomCoords=True,
                        pruneRmsThresh=0.5,
                        randomSeed=42,
                    )

                    # Get adjacency matrix
                    adj = Chem.GetAdjacencyMatrix(mol)

                    # Find lowest energy conformer
                    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    results = []

                    workers = np.min([n_cores, numConfs])
                    cpus_per_worker = n_cores // workers

                    # Create separate folders for all conformers
                    calc_dirs = [calc_dir / f"{i}" for i in range(len(mol.GetConformers()))]
                    [x.mkdir(exist_ok=True) for x in calc_dirs]

                    # Construct args
                    args = [
                        (
                            atoms,
                            conf.GetPositions(),
                            {"gfn": "ff", "opt": "tight"},
                            calc_dir,
                            cpus_per_worker,
                        )
                        for conf, calc_dir in zip(mol.GetConformers(), calc_dirs)
                    ]
                    # Submit to paralell
                    result = optimize(args, workers)

                    opt_coords_list = []
                    ff_energies = []
                    for res in result:
                        opt_coords = res[1]
                        # Check adjacency matrix after optimization
                        opt_adj = Chem.GetAdjacencyMatrix(ac2mol(atoms, opt_coords))
                        if not np.array_equal(adj, opt_adj):
                            _logger.warning(
                                f"\tChange in adjacency matrix after GFN-FF optimization. Skipping conformer"
                            )
                            continue
                        else:
                            opt_coords_list.append(opt_coords)
                            ff_energies.append(res[2])

                    # Construct args
                    args = [
                        (
                            atoms,
                            coords,
                            {
                                "gfn": 2,
                                "charge": xtb_args["charge"],
                                "uhf": xtb_args["uhf"],
                            },
                            calc_dir,
                            cpus_per_worker,
                        )
                        for coords, calc_dir in zip(opt_coords_list, calc_dirs)
                    ]

                    # Submit to paralell
                    result_sp = optimize(args, workers)

                    sp_energies = [res[2] for res in result_sp]

                    final_results = [(connection_atom_id, energy) for energy in sp_energies]

                    if len(final_results) == 0:
                        binding_energies.append((connection_atom_id, np.nan))
                    else:
                        final_results.sort(
                            key=lambda x: float("inf") if math.isnan(x[1]) else x[1]
                        )
                        binding_energies.append(final_results[0])
                binding_energies.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])

                _logger.info("Binding energies:")
                _logger.info(
                    ("{:>12}{:>12}{:>27}").format(
                        "Donor ID", "Atom Type", " Binding Energy [Hartree] - (GFN2-SP)"
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

        self.connection_atom_id = connection_atom_id


class DativeLigand(Ligand):
    """Dative bound ligands."""

    def __init__(
        self,
        mol: Mol,
        connection_atom_id: None = None,
        fixed: bool = False,
        smarts_match: bool = True,
    ) -> None:
        super().__init__(mol=mol, connection_atom_id=connection_atom_id, smarts_match=smarts_match)
        self.bond_type = Chem.BondType.DATIVE

    def find_donor_atom(
        self,
        smarts_match: bool = True,
        reference_smiles: str = "[Pd]<-P",
        xtb_args=None,
        n_cores: int = 1,
        calc_dir: str = ".",
        numConfs: int = 20,
    ) -> None:
        if smarts_match:
            connection_atom_id = None
            for p in priority_dative:
                match = self.mol.GetSubstructMatch(p)
                if len(match) > 0:
                    connection_atom_id = match[0]
                    break
            if not isinstance(connection_atom_id, int):
                _logger.warning(
                    f"No donor atom found for DativeLigand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            # Ensure that the connection atom id is none if something fails.
            connection_atom_id = None

            # Find all possible donor atoms
            pattern = Chem.MolFromSmarts("[" + ",".join(DONORS_dative) + "]")
            matches = self.mol.GetSubstructMatches(pattern)

            if len(matches) == 0:
                _logger.warning(
                    f"No donor atom found for DativeLigand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
            elif len(matches) == 1:
                # Make all possible constitutional isomers
                _logger.info(f"Found 1 possible donor atoms in DativeLigand.")
                connection_atom_id = matches[0][0]
            else:
                # Make all possible constitutional isomers
                _logger.info(f"Found {len(matches)} possible donor atoms for DativeLigand.")
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
                    _logger.info(f"Isomer: {Chem.MolToSmiles(Chem.RemoveHs(mol))}")

                    # Embed test molecule
                    mol = Chem.AddHs(mol)
                    _ = rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=numConfs,
                        useRandomCoords=True,
                        pruneRmsThresh=0.5,
                        randomSeed=42,
                    )

                    # Get adjacency matrix
                    adj = Chem.GetAdjacencyMatrix(mol)

                    # Find lowest energy conformer
                    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    results = []

                    workers = np.min([n_cores, numConfs])
                    cpus_per_worker = n_cores // workers

                    # Create separate folders for all conformers
                    calc_dirs = [calc_dir / f"{i}" for i in range(len(mol.GetConformers()))]
                    [x.mkdir(exist_ok=True) for x in calc_dirs]

                    # Construct args
                    args = [
                        (
                            atoms,
                            conf.GetPositions(),
                            {"gfn": "ff", "opt": "tight"},
                            calc_dir,
                            cpus_per_worker,
                        )
                        for conf, calc_dir in zip(mol.GetConformers(), calc_dirs)
                    ]

                    # Submit to paralell
                    result = optimize(args, workers)

                    opt_coords_list = []
                    ff_energies = []
                    for res in result:
                        opt_coords = res[1]
                        # Check adjacency matrix after optimization
                        opt_adj = Chem.GetAdjacencyMatrix(ac2mol(atoms, opt_coords))
                        if not np.array_equal(adj, opt_adj):
                            _logger.warning(
                                f"\tChange in adjacency matrix after GFN-FF optimization. Skipping conformer"
                            )
                            continue
                        else:
                            opt_coords_list.append(opt_coords)
                            ff_energies.append(res[2])

                    # Construct args
                    args = [
                        (
                            atoms,
                            coords,
                            {
                                "gfn": 2,
                                "charge": xtb_args["charge"],
                                "uhf": xtb_args["uhf"],
                            },
                            calc_dir,
                            cpus_per_worker,
                        )
                        for coords, calc_dir in zip(opt_coords_list, calc_dirs)
                    ]

                    # Submit to paralell
                    result_sp = optimize(args, workers)

                    sp_energies = [res[2] for res in result_sp]

                    final_results = [(connection_atom_id, energy) for energy in sp_energies]

                    if len(final_results) == 0:
                        binding_energies.append((connection_atom_id, np.nan))
                    else:
                        final_results.sort(
                            key=lambda x: float("inf") if math.isnan(x[1]) else x[1]
                        )
                        binding_energies.append(final_results[0])

                binding_energies.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])

                _logger.info("Binding energies:")
                _logger.info(
                    ("{:>12}{:>12}{:>27}").format(
                        "Donor ID", "Atom Type", " Binding Energy [Hartree] - GFN2-SP"
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

        self.connection_atom_id = connection_atom_id


class BidentateLigand(Ligand):
    """Bidentate ligands."""

    def __init__(self, mol, connection_atom_id=None, fixed=False, smarts_match=False):
        super().__init__(mol=mol, connection_atom_id=connection_atom_id, smarts_match=smarts_match)
        self.bond_type = Chem.BondType.DATIVE

    def find_donor_atom(
        self, smarts_match=True, reference_smiles="[Pd]<-P", n_cores=1, calc_dir="."
    ):
        """For this ligand, there are two connection atom ids.

        These are stored in a list
        """

        # Alays true for now
        smarts_match = True

        connection_atom_id = []
        matches = []

        if smarts_match:
            # Prioritize amines for bidentates
            for elem in [AMINE, CARBENE, PHOSPHINE, OXYGEN, CO]:
                pattern = Chem.MolFromSmarts("[" + elem + "]")
                if self.mol.GetSubstructMatches(pattern):
                    matches += self.mol.GetSubstructMatches(pattern)
            # Crude way of only handling ligands with 2 clear attachment point.
            # Only if 2 matches exists the ligand is accepted.
            if len(matches) == 2:
                connection_atom_id.append(matches[0][0])
                connection_atom_id.append(matches[1][0])
            if not connection_atom_id:
                _logger.warning(
                    f"No donor atoms found for BidentateLigand( {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )
        else:
            raise NotImplementedError("Bonding site selection not implemented yet")

        # Set donor id on ligand
        self.connection_atom_id = connection_atom_id


class Metal:
    """Transition Metal."""

    def __init__(self, atom: str, coordination_number: None = None) -> None:
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
