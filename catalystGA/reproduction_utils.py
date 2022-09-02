import copy
import random
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def list_crossover(genome1: List, genome2: List, n_cutpoints: int = 1) -> List:
    """
    Performs crossover of two lists
    """
    assert len(genome1) == len(genome2)
    n_genes = len(genome1)

    if n_cutpoints == 1:
        cut_point = random.randint(0, n_genes - 1)
        new_genome = genome1[:cut_point] + genome2[cut_point:]
    elif n_cutpoints == 2:
        raise NotImplementedError()
    else:
        raise ValueError("n_cutpoints can be either 1 or 2")
    return new_genome


def list_mutate(genome, possible_genes) -> List:
    """
    Inserts random gene from possible_genes at random position in genome
    """
    n_genes = len(genome)
    new_gene = random.choice(possible_genes)
    insert_point = random.randint(0, n_genes - 1)
    new_genome = copy.deepcopy(genome)
    new_genome[insert_point] = new_gene
    return new_genome


def cut(mol: Chem.Mol) -> List[Chem.Mol] or None:
    """Cut a molecule into fragments at non ring position

    Args:
        mol (Chem.Mol): Input molecule

    Returns:
        List[Chem.Mol]: List of fragments
    """
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None
    bis = random.choice(
        mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]"))
    )  # single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        return fragments
    except:
        return None


def cut_ring(mol: Chem.Mol) -> List[Chem.Mol] or None:
    """Cut a molecule into two fragments at ring position

    Args:
        mol (Chem.Mol): Input molecule

    Returns:
        List[Chem.Mol]: List of fragments
    """
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[2], bis[3]),
            )
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[1], bis[2]),
            )
        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(
            mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)]
        )

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            return None

        if len(fragments) == 2:
            return fragments

    return None


def ring_OK(mol: Chem.Mol) -> bool:
    """Check if ring system is OK

    Args:
        mol (Chem.Mol): Input molecule to check

    Returns:
        bool: is ring system OK
    """
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_OK(mol: Chem.Mol) -> bool:
    """Check if molecule passes sanitization

    Args:
        mol (Chem.Mol): Input molecule to check

    Returns:
        bool: is mol OK
    """
    try:
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    except:
        return False
    if test_mol != None:
        return True
    else:
        return False


def crossover_ring(parent_A: Chem.Mol, parent_B: Chem.Mol) -> Chem.Mol or None:
    """Performs crossover of two molecules with ring system

    Args:
        parent_A (Chem.Mol): Input molecule 1
        parent_B (Chem.Mol): Input molecule 2

    Returns:
        Chem.Mol or None
    """
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    rxn_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]
    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)
        if fragments_A == None or fragments_B == None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_OK(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_OK(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def crossover_non_ring(parent_A: Chem.Mol, parent_B: Chem.Mol) -> Chem.Mol or None:
    """Performs crossover of two molecules on non-ring systems

    Args:
        parent_A (Chem.Mol): Input molecule 1
        parent_B (Chem.Mol): Input molecule 2

    Returns:
        Chem.Mol or None
    """
    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A == None or fragments_B == None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_OK(mol):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None


def graph_crossover(parent_A: Chem.Mol, parent_B: Chem.Mol) -> Chem.Mol or None:
    """Performs crossover between two molecules, either via ring system or non-ring system

    Args:
        parent_A (Chem.Mol): Input molecule 1
        parent_B (Chem.Mol): Input molecule 2

    Returns:
        Chem.Mol or None: Output molecule or None if molecule did not pass checks
    """
    parent_A = copy.deepcopy(parent_A)
    parent_B = copy.deepcopy(parent_B)
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)
    except:
        pass
    for i in range(10):
        if random.random() <= 0.5:
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol != None:
                new_smiles = Chem.MolToSmiles(new_mol)
            if new_mol != None and new_smiles not in parent_smiles:
                Chem.SanitizeMol(new_mol)
                return new_mol
        else:
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol != None:
                new_smiles = Chem.MolToSmiles(new_mol)
            if new_mol != None and new_smiles not in parent_smiles:
                Chem.SanitizeMol(new_mol)
                return new_mol

    return None


def delete_atom() -> str:
    """Reaction SMARTS to delete an atom from a molecule"""
    choices = [
        "[*:1]~[D1]>>[*:1]",
        "[*:1]~[D2]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]",
        "[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]",
        "[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom() -> str:
    """Reaction SMARTS to append an atom to a molecule"""
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    if BO == "triple":
        rxn_smarts = "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)

    return rxn_smarts


def insert_atom() -> str:
    """Reaction SMARTS to insert an atom into a molecule"""
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    if BO == "triple":
        rxn_smarts = "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)

    return rxn_smarts


def change_bond_order() -> str:
    """Reaction SMARTS to change bond order in a molecule"""
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond() -> str:
    """Reaction SMARTS to delete an atom into a molecule"""
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def add_ring() -> str:
    """Reaction SMARTS to add a 3-6 membered ring to a molecule"""
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol: Chem.Mol) -> str:
    """Reaction SMARTS to replace a single atom in a molecule"""
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)


def graph_mutate(mol: Chem.Mol) -> Chem.Mol or None:
    """Performs mutation on molecule (add, remove or replace bond or atom)

    Args:
        mol (Chem.Mol): Input molecule

    Returns:
        Chem.Mol or None: Output molecule or None if molecule did not pass checks
    """
    mol = copy.deepcopy(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * [""]
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            if mol_OK(m) and ring_OK(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            new_mol = random.choice(new_mols)
            Chem.SanitizeMol(new_mol)
            return new_mol

    return None
