import numpy as np
from rdkit import Chem

from catalystGA.utils import MoleculeOptions


def check_n_heavy_atoms(mol: Chem.Mol, mol_options: MoleculeOptions) -> bool:
    """Check if molecule has number of heavy atoms within range specified in mol_options

    Args:
        mol (Chem.Mol): Molecule to check
        mol_options (MoleculeOptions): Average number of heavy atoms and standard deviation

    Returns:
        bool: is within range
    """
    target_size = mol_options.size_std * np.random.randn() + mol_options.average_size
    if mol.GetNumHeavyAtoms() > 5 and mol.GetNumHeavyAtoms() < target_size:
        return True
    else:
        return False
