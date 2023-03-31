from unittest import TestCase

from rdkit import Chem

from catalystGA import BaseCatalyst, Ligand, Metal


class TestBaseCatalyst(TestCase):

    n_ligands = 2

    def setUp(self):
        # Create a metal object
        metal = Metal("Pd")

        # Create a list of ligands
        ligands = [
            Ligand.from_smiles("CCN(C)C"),
            Ligand.from_smiles("CC(C)(C)P"),
        ]

        # Create a dummy BaseCatalyst object
        self.base_catalyst = BaseCatalyst(metal, ligands)

    def test_smiles(self):
        # Check that the smiles string is correct
        self.assertEqual(self.base_catalyst.smiles, "CCN(C)(C)->[Pd]<-PC(C)(C)C")

    def test_assemble_mol(self):
        # Assemble the catalyst molecule
        catalyst_mol = self.base_catalyst.assemble()

        # Check that the catalyst molecule is a Chem.Mol object
        self.assertIsInstance(catalyst_mol, Chem.Mol)

        # Check that the correct number of atoms are present in the catalyst molecule
        self.assertEqual(catalyst_mol.GetNumAtoms(), 33)

        # Check that the donor atoms in the ligands have been correctly bonded to the metal atom
        self.assertEqual(
            catalyst_mol.GetBondBetweenAtoms(self.base_catalyst.donor_ids[0], 0).GetBondType(),
            Chem.BondType.DATIVE,
        )
        self.assertEqual(
            catalyst_mol.GetBondBetweenAtoms(self.base_catalyst.donor_ids[1], 0).GetBondType(),
            Chem.BondType.DATIVE,
        )

    def test_assemble_setting_donor_ids(self):

        # Assemble the catalyst molecule
        _ = self.base_catalyst.assemble()

        # Check that the `donor_ids` of `self` have been correctly set
        self.assertEqual(self.base_catalyst.donor_ids, [3, 21])

        # Assemble the catalyst molecule with extra ligands
        _ = self.base_catalyst.assemble(extraLigands="[Pd:1]>>[Pd:1](-Br)(-C=C)")

        # Check that the `donor_ids` of `self` have been correctly set
        self.assertEqual(self.base_catalyst.donor_ids, [9, 27])

    def test_assemble_extra_ligands(self):
        # Assemble the catalyst molecule
        catalyst_mol = self.base_catalyst.assemble(extraLigands="[Pd:1]>>[Pd:1](-Br)(-C=C)")

        # Check that the correct number of atoms are present in the catalyst molecule
        self.assertEqual(catalyst_mol.GetNumAtoms(), 39)

        # Check that the donor atoms in the extra ligands have been correctly bonded to the metal atom
        self.assertEqual(
            catalyst_mol.GetBondBetweenAtoms(1, 0).GetBondType(), Chem.BondType.SINGLE
        )
        self.assertEqual(
            catalyst_mol.GetBondBetweenAtoms(2, 0).GetBondType(), Chem.BondType.SINGLE
        )

    def test_assemble_chiral_tag_and_permutation_order(self):
        # Assemble the catalyst molecule
        catalyst_mol = self.base_catalyst.assemble(
            chiralTag=Chem.CHI_SQUAREPLANAR, permutationOrder=2
        )

        # Check that this right chiral tag is set on the metal atom
        self.assertEqual(catalyst_mol.GetAtomWithIdx(0).GetChiralTag(), Chem.CHI_SQUAREPLANAR)

        # Check that the permutation order is set correctly
        self.assertEqual(catalyst_mol.GetAtomWithIdx(0).GetIntProp("_chiralPermutation"), 2)

    def test_embed(self):
        # Embed the catalyst molecule
        catalyst_mol = self.base_catalyst.embed()

        # Check that the embedded molecule is a Chem.Mol object
        self.assertIsInstance(catalyst_mol, Chem.Mol)

        # Check that the correct number of conformers have been generated
        self.assertEqual(catalyst_mol.GetNumConformers(), 10)
