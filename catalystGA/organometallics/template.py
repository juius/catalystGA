import itertools

import networkx as nx
import numpy as np
import xyz2mol
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from catalystGA.organometallics.components import Ligand

non_metals = [
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    32,
    33,
    34,
    35,
    36,
    51,
    52,
    53,
    54,
]


class StructureTemplate:
    # needs to be able to handle charge, store that somewhere
    # also how to deal with stereoisomers
    def __init__(self, central_atom, dummy_ligands, fixed_ligands=None):
        self.central_atom = central_atom
        self.dummy_ligands = dummy_ligands
        self.fixed_ligands = fixed_ligands

    @classmethod
    def from_sdf(sdf_file):
        pass

    @classmethod
    def from_axyzc(
        cls,
        atomnums,
        coords,
        charge,
        use_huckel=True,
        central_atom_id=None,
        dummy_ligand_ids=None,
        fixed_ligand_donor_ids=None,
        not_bound_to_metal=None,
    ):
        """Generate StructureTemplate from atomnums, charge, coords"""
        ac = xyz2mol.xyz2AC(atomnums, coords, charge, use_huckel=use_huckel)[0]

        # remove wrong bonds from AC
        if fixed_ligand_donor_ids:
            not_bound = list(itertools.combinations(fixed_ligand_donor_ids, 2))
            for id1, id2 in not_bound:
                ac[id1, id2] = 0
                ac[id2, id1] = 0

        # determine centeral atom id
        if not central_atom_id:
            metals = [(i, an) for i, an in enumerate(atomnums) if an not in non_metals]
            assert len(metals) == 1, f"Found more than one metal in xyz file: {metals}"
            central_atom_id, metal = metals[0]

        # remove wrong bonds to metal
        if not_bound_to_metal:
            for idx in not_bound:
                ac[idx, central_atom_id] = 0
                ac[central_atom_id, idx] = 0

        # determine ligand ids
        if not dummy_ligand_ids:
            mask = ac[central_atom_id]
            dummy_ligand_ids = list(np.where(mask)[0])

        # add fixed ligands
        if fixed_ligand_donor_ids:
            for idx in fixed_ligand_donor_ids:
                if idx in dummy_ligand_ids:
                    dummy_ligand_ids.remove(idx)
            # remove connections to central atom in ac and find subgraphs
            zeros = np.zeros_like(ac[central_atom_id])
            seperate = list(itertools.permutations(fixed_ligand_donor_ids, 2))
            for pair in seperate:
                ac[pair[0], pair[1]] = 0
            ac[central_atom_id, :] = zeros
            ac[:, central_atom_id] = zeros
            g = nx.from_numpy_matrix(ac)
            fixed_ligands = []
            for didx in fixed_ligand_donor_ids:
                frag = nx.node_connected_component(g, didx)
                mol = Chem.RWMol()
                num_dict = {}
                for idx in frag:
                    num = mol.AddAtom(Chem.Atom(atomnums[idx]))
                    # Set Positions of atoms as atomprop
                    mol.GetAtomWithIdx(num).SetProp("position", f"{coords[idx]}")
                    num_dict[idx] = num
                    edges = g.edges(idx)
                    for i, j in edges:
                        # add bond if both atoms have been initialized
                        if i > j:
                            _ = mol.AddBond(num_dict[i], num_dict[j], Chem.BondType.SINGLE)
                mol = mol.GetMol()
                fixed_ligand = Ligand(mol=mol, donor_id=0, fixed=True)
                fixed_ligand.set_positions(np.take(coords, list(frag), axis=0))
                fixed_ligands.append(fixed_ligand)
        else:
            fixed_ligands = None

        central_atom = Point3D(*coords[central_atom_id])
        dummy_ligands = [Point3D(*coords[id]) for id in dummy_ligand_ids]

        return cls(central_atom, dummy_ligands, fixed_ligands)

    @classmethod
    def from_xyz(
        cls,
        xyz_file,
        **kwargs,
    ):
        """Generate StructureTemplate from xyz file"""
        atomnums, charge, coords = xyz2mol.read_xyz_file(xyz_file)
        return cls.from_axyzc(atomnums, coords, charge, **kwargs)

    # @classmethod
    # def from_mol(
    #     cls,
    #     mol,
    #     **kwargs,
    # ):
    #     """Generate StructureTemplate from xyz file"""
    #     # atomnums, charge, coords = xyz2mol.read_xyz_file(xyz_file)
    #     return cls.from_axyzc(atomnums, coords, charge, **kwargs,)
