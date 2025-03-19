import concurrent.futures
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDistGeom
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolHash import HashFunction, MolHash

from catalystGA.xtb import ac2mol, xtb_calculate

TRANSITION_METALS = (
    "[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg]"
)

#  Dative bond patterns  ###

CARBENE = "#6&v2H0,#6&v3H0"
PHOSPHINE = "#15&v3"
AMINE = "#7&v3"
OXYGEN = "#8&v2,#8&v1"
CO = "C-,v5"

# Extra patterns based on tmQMg
SULPHUR = "#16&v2,#16&v3,#16&v1"  # sulphurs
ARSENIC = "#33&v2,#33&v3"  # Triple or double bound Arsenic
SELENIUM = "#34&v1,#34&v2"  # Single or double bound Selenium
SILICON = "#14&v2"  # Double bound Silicon
tmQMg_patterns = [SULPHUR, ARSENIC, SELENIUM, SILICON]


DONORS_dative = [CARBENE, PHOSPHINE, AMINE, OXYGEN, CO] + tmQMg_patterns
priority_dative = [Chem.MolFromSmarts("[" + pattern + "]") for pattern in DONORS_dative]


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
        self.fitness = math.nan
        self.error = ""
        self.idx = (-1, -1)
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

    @hide_warnings  # Suppress MetalDisconnector output
    def from_smiles(cls, smiles: str):
        """Generate Catalyst from SMILES string.

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

        # label donor atoms
        for atom in mol.GetAtomWithIdx(metal_id).GetNeighbors():
            atom.SetBoolProp("donor_atom", True)

            # fragment complex
            fragments = rdMolStandardize.DisconnectOrganometallics(mol)
            ligands = []
            for ligand_mol in Chem.GetMolFrags(fragments, asMols=True):
                Chem.SanitizeMol(ligand_mol)
                if ligand_mol.HasSubstructMatch(Chem.MolFromSmarts(TRANSITION_METALS)):
                    metal = Metal(ligand_mol)
                else:
                    # find donor atom
                    for atom in ligand_mol.GetAtoms():
                        if atom.HasProp("donor_atom"):
                            donor_id = atom.GetIdx()
                            break
                    ligand_mol = Chem.AddHs(ligand_mol)
                    Chem.SanitizeMol(ligand_mol)
                    ligands.append(Ligand(ligand_mol, donor_id=donor_id))

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
            # emol.GetAtomWithIdx(connection_atom_id).SetNumExplicitHs(0)

        # Commit changes made and get mol
        emol.CommitBatchEdit()
        mol = emol.GetMol()

        # Catch sanitation errors. NB! could lead to error later in workflow.
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _logger.warning(f"Sanitation error! Molecule: {mol}")
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
        smarts_match: bool = True,
    ) -> None:
        self.mol = mol
        if not connection_atom_id:
            self.find_donor_atom(smarts_match=smarts_match)
        else:
            self.connection_atom_id = connection_atom_id

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


class CovalentLigand(Ligand):
    """Covalently bound ligands."""

    def __init__(
        self,
        mol: Mol,
        connection_atom_id: int = None,
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
                                "\tChange in adjacency matrix after GFN-FF optimization. Skipping conformer"
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
        calc_dir: Path = Path("."),
        numConfs: int = 2,
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
                _logger.info("Found 1 possible donor atoms in DativeLigand.")
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
                                "Change in adjacency matrix after gfn-ff optimization. skipping conformer"
                            )
                            continue
                        else:
                            opt_coords_list.append(opt_coords)
                            ff_energies.append(res[2])
                            _logger.info(f"Found {len(ff_energies)} valid ff energies")

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


class Ligand:
    """Class representing a ligand."""

    def __init__(
        self,
        smiles,
        connection_atom_ids=None,
    ) -> None:
        self.mol = Chem.MolFromSmiles(smiles)
        self.smiles = smiles
        if connection_atom_ids:
            self.connection_atom_ids = connection_atom_ids
        else:
            # Define a monodentate binding site
            connection_atom_ids = None
            for p in priority_dative:
                match = self.mol.GetSubstructMatch(p)
                if len(match) > 0:
                    self.connection_atom_ids = [match[0]]
                    break
            if not self.connection_atom_ids:
                _logger.warning(
                    f"No donor atom found for Ligand {Chem.MolToSmiles(Chem.RemoveHs(self.mol))}"
                )

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


class TMC:
    """Class to represent TMC."""

    save_attributes = {}

    def __init__(self, metal: Chem.Mol, ligands: List[Ligand]):
        self.score = math.nan
        self.fitness = math.nan
        self.error = ""
        self.idx = (-1, -1)
        self.metal = metal
        self.ligands = ligands

        # Caches ligands to not rerun assmble code all the time
        self._cached_mol = self.assemble()
        self._cached_smiles = Chem.MolToSmiles(self._cached_mol)
        self._ligands_snapshot = None

        self.tm_idx = self._cached_mol.GetSubstructMatch(Chem.MolFromSmarts(TRANSITION_METALS))[0]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metal},{self.ligands})"

    def __hash__(self) -> int:
        return hash(self.smiles)

    def __eq__(self, other) -> bool:
        if isinstance(other, TMC):
            if self.__hash__() == other.__hash__():
                return True
        return False

    @property
    def mol(self):
        if self._ligands_changed():
            mol = self.assemble()
            self._cached_smiles = MolHash(Chem.RemoveHs(mol), HashFunction.CanonicalSmiles)
            self._cached_mol = Chem.MolFromSmiles(self._cached_smiles)
        return self._cached_mol

    @property
    def smiles(self) -> str:
        if self._ligands_changed():
            mol = self.assemble()
            self._cached_smiles = MolHash(Chem.RemoveHs(mol), HashFunction.CanonicalSmiles)
            self._cached_mol = Chem.MolFromSmiles(self._cached_smiles)
        return self._cached_smiles

    def _ligands_changed(self) -> bool:
        """Check if the ligands list has changed since the last assemble."""
        ligands_snapshot = [(lig.smiles, lig.connection_atom_ids) for lig in self.ligands]
        if ligands_snapshot != self._ligands_snapshot:
            self._ligands_snapshot = ligands_snapshot
            return True
        return False

    @property
    def dispatcher(self):
        "Utility function used to dispatch the scoring"
        return {
            "calculate_score": self.calculate_score,
            "toy": self.toy,
        }

    def toy(self, args):
        "Scoring function to use for debugging"
        _logger.info("Getting logp")
        self.score = Descriptors.MolLogP(self.mol)
        _logger.info("Got logP")

    def save(self, directory=".") -> None:
        """Dump TMC object into file."""
        filename = os.path.join(directory, "../ind.pkl")
        with open(filename, "wb+") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def get_props(self):
        return vars(self)

    @classmethod
    def from_smiles(cls, smiles: str):
        """Generate TMC from SMILES string.

        Args:
            smiles (str): TMC SMILES string

        Returns:
            Instance of TMC Class
        """
        mol = Chem.MolFromSmiles(smiles)
        test_smiles = smiles
        assert mol, "Could not parse SMILES string"

        # get transition metal
        metal_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(TRANSITION_METALS))
        assert len(metal_matches) > 0, "No transition metal found in molecule"
        assert len(metal_matches) < 2, "More than one transition metal found in molecule"

        tmc_idx = None
        for a in mol.GetAtoms():
            a.SetIntProp("__origIdx", a.GetIdx())
            if a.GetAtomicNum() in TRANSITION_METALS_NUM:
                # tm_atom = a.GetSymbol()
                tmc_idx = a.GetIdx()

        # Get TM neighbors to get ligand coonnection atom ids.
        coordinating_atoms = np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol)[tmc_idx, :])[0]

        mdis = rdMolStandardize.MetalDisconnector(params)
        mdis.SetMetalNon(Chem.MolFromSmarts(MetalNon_Hg))
        frags = mdis.Disconnect(mol)
        frag_mols = rdmolops.GetMolFrags(frags, asMols=True)

        # Get ligand list
        ligands = []
        for i, f in enumerate(frag_mols):
            if f.GetSubstructMatch(Chem.MolFromSmarts(TRANSITION_METALS)):
                metal = Metal(f)
                continue
            # print(lig_charge)
            lig_coordinating_atoms = [
                a.GetIdx() for a in f.GetAtoms() if a.GetIntProp("__origIdx") in coordinating_atoms
            ]

            # Get mapped ids
            smiles, mapper = get_smiles_atomidx_mapping(f)
            lig_coordinating_atoms = [mapper[x] for x in lig_coordinating_atoms]

            ligands.append(
                Ligand(
                    Chem.MolToSmiles(f),
                    connection_atom_ids=lig_coordinating_atoms,
                )
            )

        # Instantiate based on ligand list.
        cat = cls(metal, ligands)
        assert (
            cat.smiles == test_smiles
        ), f"SMILES string does not match input SMILES: {cat.smiles} != {test_smiles}"
        return cat

    def assemble(
        self,
        extraLigands=None,
        chiralTag=None,
        permutationOrder=None,
    ) -> Mol:
        """Forms bonds from Ligands to Metal Center, adds any extra Ligands
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
            connection_atom_ids = ligand.connection_atom_ids

            # Add bond to metal.
            mapped_coords = atom_ids[i + 1]
            for elem in connection_atom_ids:
                connection_atom_id = mapped_coords[elem]
                emol.AddBond(connection_atom_id, 0, Chem.BondType.DATIVE)

        # Commit changes made and get mol
        emol.CommitBatchEdit()
        mol = emol.GetMol()

        # Catch sanitation errors. NB! could lead to error later in workflow.
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _logger.warning(f"Sanitation error! Molecule: {mol}")
            _logger.warning(f"Traceback : {e}")

        # Set Chiral Tag and Permutation Order
        if chiralTag:
            metal = mol.GetAtomWithIdx(self.tm_idx)
            self._setChiralTagAndOrder(metal, chiralTag, permutationOrder)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            _logger.warning("Sanitation error after applying chiral tag! Molecule: {mol}")
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
        """Embed the TMC Molecule using ETKDG.

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

    def calculate_score(self, args) -> None:
        """Calculate score for the catalyst."""

        scratch = args["output_dir"] / args["scratch"]
        scratch.mkdir(parents=True, exist_ok=True)
        start = time.time()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.StreamHandler(),  # For debugging. Can be removed on remote
            ],
        )
        _logger.info(socket.gethostname())
        _logger.info(f"Calculating score for {self}\nSMILES: {self.smiles}\n")

        args["charge"] = Chem.GetFormalCharge(self.mol)
        args["uhf"] = 2
        args["multiplicity"] = 3

        # Create moles with different permutations of the ligands. We only use the trans configuration
        permutation_mols = []
        for permutationOrder in [2]:
            permutation_mols.append(
                self.embed(
                    chiralTag=Chem.CHI_SQUAREPLANAR,
                    permutationOrder=permutationOrder,
                    numConfs=args["n_confs"],
                    useRandomCoords=True,
                    pruneRmsThresh=args["rms_prune"],
                    numThreads=args["cpus_per_mol"],
                )
            )
        args["timeout_min"] = args["timeout_min"] // len(permutation_mols) - 1

        results_dict = defaultdict(dict)

        # XTB calulation optikons
        options = {"pop": True, "opt": True}

        _logger.info("Running xtb with args: \n")
        _logger.info(json.dumps(options, indent=4), extra={"simple": True})
        _logger.info(json.dumps(make_json_serializable(args), indent=4), extra={"simple": True})

        for i, permutation in enumerate(permutation_mols):
            start_connectivity = Chem.GetAdjacencyMatrix(permutation)
            _logger.info(f"Checking permutation {Chem.MolToSmiles(permutation)}")

            # Instantiate optimizer class
            tempdir = scratch / "permutation" / f"{self.idx[0]:03d}_{self.idx[1]:03d}_{i}"
            tempdir.mkdir(parents=True, exist_ok=True)
            args["name"] = tempdir

            xyz_block = Chem.MolToXYZBlock(permutation)
            atoms, coords = xyz2ac(xyz_block)

            results = xtb_calculate(
                atoms=atoms,
                coords=coords,
                charge=args["charge"],
                multiplicity=args["multiplicity"],
                n_cores=args["cpus_per_mol"],
                timeout=args["timeout_min"],
                calc_dir=tempdir,
                options=options,
            )
            if not results["normal_termination"]:
                _logger.info(f"Permtation {i} did not terminate normally")
                continue
            atoms = results["atoms"]
            coords = results["coords"]

            # Check for bonds breaking during calcultion
            is_good = adjacency_check(start_connectivity, atoms=atoms, coords=coords)
            if is_good:
                results_dict[i] = results

        _logger.info(f"Finished all permutations for {repr(self)}, {self.idx}")
        # Save results
        if not results_dict:
            _logger.warning(f"No valid calculations for {self.idx}. Returning from calculation")
            return

        # Get the best charge results
        if args["maximize_score"]:
            largest_dict = max(
                results_dict, key=lambda d: results_dict[d]["mulliken"][self.tm_idx]
            )
        else:
            largest_dict = min(
                results_dict, key=lambda d: results_dict[d]["mulliken"][self.tm_idx]
            )

        self.results = results_dict[largest_dict]

        try:
            # Get the normalized score
            score = self.results["mulliken"][self.tm_idx]
            _logger.debug(args["maximize_score"])
            norm = self.get_normalized_score(score, maximize_score=args["maximize_score"])
            self.score = norm

        except Exception as e:
            _logger.info(f"Score calculation failed for {self.idx}. Traceback : {e}")
        self.timing = time.time() - start

    def get_normalized_score(self, value, maximize_score=True, newRange=(0, 1)):
        "Normalize score to the range 0-1"

        # These xmin and xmax values are hardcoded based on expected good and bad values given the scoring function.
        if maximize_score:
            xmin, xmax = -1, 1
        else:
            xmin, xmax = 1, -1

        if math.isnan(value):
            norm = 0
        else:
            norm = (value - xmin) / (xmax - xmin)  # scale between zero and one
        if newRange == (0, 1):
            pass
        elif newRange != (0, 1):
            norm = norm * (newRange[1] - newRange[0]) + newRange[0]  # scale to a different range.

        if norm > 1:
            norm = 1
        elif norm < 0:
            norm = 0

        return norm


def optimize(args, workers):
    """Do paralell optimization of all the entries in args."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            xtb_calculate,
            [arg[0] for arg in args],
            [arg[1] for arg in args],
            [arg[2] for arg in args],
            [arg[3] for arg in args],
            [arg[4] for arg in args],
        )
    return results
