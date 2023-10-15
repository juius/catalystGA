from unittest import TestCase, mock

from rdkit import Chem

from catalystGA.components import DativeLigand, Ligand

MOCK_ATOMS = [
    "Pd",
    "P",
    "H",
    "H",
    "H",
    "C",
    "N",
    "C",
    "P",
    "C",
    "C",
    "C",
    "C",
    "C",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
]
MOCK_COORDS1 = [
    [-1.57888339, -0.55562389, 0.87361627],
    [-2.47564105, -1.10751468, 3.06938604],
    [-3.89214016, -0.74625043, 3.08824222],
    [-1.78022103, -0.28384969, 4.04944753],
    [-2.28346651, -2.52439073, 3.32218052],
    [-1.33346772, 0.93294386, -1.51592473],
    [-0.646054, -0.18770339, -0.8942155],
    [-0.90781606, -1.34730331, -1.73336651],
    [1.07192823, 0.02058541, -0.71065083],
    [2.10711304, -0.30538399, -2.15229587],
    [1.65526195, 1.26086791, 0.45278333],
    [1.44650073, 2.69279757, 0.03124446],
    [1.05301962, 1.10832537, 1.83485514],
    [3.16991047, 1.08306313, 0.63668105],
    [-1.49670924, 1.71155629, -0.74911545],
    [-0.81337479, 1.28504286, -2.41455019],
    [-2.34926227, 0.56698758, -1.84687625],
    [-1.96256929, -1.38810146, -2.06688641],
    [-0.28059972, -1.30747932, -2.64719528],
    [-0.64100228, -2.28949878, -1.21372017],
    [2.80566947, -1.16037671, -1.92353857],
    [2.78067766, 0.54856964, -2.41204113],
    [1.58056008, -0.58610183, -3.07432604],
    [1.27327475, 2.81424441, -1.04247621],
    [0.55388917, 3.15537519, 0.54229583],
    [2.3077809, 3.32751912, 0.35379362],
    [1.66144605, 1.78847328, 2.50081019],
    [1.10725076, 0.07378815, 2.2161122],
    [0.02013672, 1.50397135, 1.89661761],
    [3.63251752, 1.94698894, 1.11448512],
    [3.55464155, 0.89775635, -0.40536722],
    [3.30724445, 0.12639414, 1.18309913],
]

MOCK_COORDS2 = [
    [-2.20986041, -0.68027484, -0.61771658],
    [-3.13400754, -2.30384767, -2.19303737],
    [-4.01273415, -3.23749283, -1.53595008],
    [-2.04429639, -3.09813183, -2.78366378],
    [-3.75845144, -1.59330793, -3.28347437],
    [0.46559226, 1.6575085, -0.98189656],
    [0.64293697, 1.05304526, 0.34883137],
    [1.27414717, -0.24734633, 0.09011505],
    [-1.0045736, 0.72435409, 0.97348589],
    [-1.92377657, 2.32768246, 1.10518902],
    [-1.00069857, -0.10191261, 2.6087055],
    [-0.04336559, 0.51251565, 3.59527473],
    [-0.7702887, -1.58492934, 2.42293508],
    [-2.40217017, 0.0546039, 3.2139536],
    [-0.551205, 1.42704326, -1.38789604],
    [1.17721318, 1.21852809, -1.71715439],
    [0.60523349, 2.74152379, -0.95562971],
    [1.75428592, -0.64180415, 1.00268456],
    [0.59574292, -0.95067032, -0.39835612],
    [2.0962558, -0.0418759, -0.64806479],
    [-1.79044645, 2.94334274, 0.18635825],
    [-2.98378358, 2.18640884, 1.37283476],
    [-1.4458022, 2.89926856, 1.95104785],
    [-0.47394756, 1.46853534, 3.93435009],
    [-0.04006995, -0.1564976, 4.49223747],
    [0.9889458, 0.59463675, 3.21721776],
    [0.31166175, -1.81358003, 2.55558313],
    [-1.37089666, -2.1542545, 3.15337256],
    [-1.06268507, -1.90318138, 1.39288969],
    [-2.61251906, -0.76761796, 3.91099265],
    [-3.10609754, -0.03697666, 2.35984666],
    [-2.50736254, 1.04726758, 3.67335045],
]


class TestLigand(TestCase):
    def setUp(self):
        smiles = "CN(C)P(C)C(C)(C)C"
        # self.ligand = Ligand.from_smiles(smiles)
        self.ligand = DativeLigand(Chem.MolFromSmiles(smiles), smarts_match=True)

    def test_ligand_mol(self):
        # Check that the catalyst molecule is a Chem.Mol object
        self.assertIsInstance(self.ligand.mol, Chem.Mol)
        # Check that a donor id is set
        self.assertIsInstance(self.ligand.donor_id, int)

    def test_eq(self):
        # Check that `__eq__` works correctly
        smiles = "CN(C)P(C)C(C)(C)C"
        ligand2 = Ligand.from_smiles(smiles)
        self.assertEqual(self.ligand, ligand2)

    def test_not_eq(self):
        # Check that `__eq__` works correctly
        smiles = "C(C)P(C)C(C)(C)C"
        ligand3 = Ligand.from_smiles(smiles)
        self.assertNotEqual(self.ligand, ligand3)

    def test_find_donor_atom_smarts(self):
        # Check that right donor atom is found with smarts matching
        self.ligand.find_donor_atom(smarts_match=True)
        self.assertEqual(self.ligand.donor_id, 3)

    @mock.patch(
        "catalystGA.components.xtb_calculate",
        side_effect=[
            *44 * [(MOCK_ATOMS, MOCK_COORDS1, -38.22)],
            *44 * [(MOCK_ATOMS, MOCK_COORDS2, -38.24)],
        ],
    )
    def test_find_donor_atom_xtb(self, mock_xtb):
        # Check that right donor atom is found from xtb calculations
        # xTB calculation in mocked and conformer generation is fixed with random seed
        self.ligand.donor_id = None
        self.ligand.find_donor_atom(smarts_match=False, xtb_args={"charge": 0, "uhf": 0})
        self.assertEqual(self.ligand.donor_id, 3)
