import math

from rdkit import Chem
from rdkit.Chem import Descriptors

from catalystGA import GA
from catalystGA.reproduction_utils import graph_crossover, graph_mutate
from catalystGA.utils import MoleculeOptions


class OrganicCatalyst:
    save_attributes = {}  # any other attributes to save to the database

    def __init__(self, mol: Chem.Mol) -> None:
        self.mol = mol
        self.score = math.nan
        self.fitness = math.nan
        self.timing = math.nan
        self.error = ""
        self.idx = (-1, -1)

    @property
    def smiles(self) -> str:
        """Yields SMILES string of molecule, needed for Database.

        Returns:
            str: SMILES string
        """
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol))

    def calculate_score(
        self, n_cores: int = 1, envvar_scratch: str = "SCRATCH", scoring_kwargs: dict = {}
    ):
        """Calculate score of molecule, store in self.score.

        Args:
            n_cores (int, optional): Number of cores to use when run on cluster. Defaults to 1.
            envvar_scratch (str, optional): Name of environmental variable pointing to scratch directory. Defaults to 'SCRATCH'.
            scoring_kwargs (dict, optional): Additional keyword agruments parsed to scoring function. Defaults to {}.
        """
        # TODO: implement scoring function
        # this is just a placeholder
        logP = Descriptors.MolLogP(self.mol)
        self.score = logP


class GraphGA(GA):
    def __init__(
        self,
        mol_options,
        population_size,
        n_generations,
        mutation_rate,
        scoring_kwargs,
        db_location,
    ):
        super().__init__(
            mol_options=mol_options,
            population_size=population_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
            db_location=db_location,
            scoring_kwargs=scoring_kwargs,
        )

    def make_initial_population(self):
        with open("data/ligands.smi", "r") as f:
            lines = f.readlines()
        mols = [Chem.MolFromSmiles(line.strip()) for line in lines]
        population = [OrganicCatalyst(mol) for mol in mols[: self.population_size]]
        return population

    def crossover(self, ind1, ind2):
        mol1 = ind1.mol
        mol2 = ind2.mol
        new_mol = None
        while not new_mol:
            new_mol = graph_crossover(mol1, mol2)
        try:
            Chem.SanitizeMol(new_mol)
            ind = OrganicCatalyst(new_mol)
            return ind
        except Exception:
            return None

    def mutate(self, ind):
        mol = ind.mol
        new_mol = None
        while not new_mol:
            new_mol = graph_mutate(mol)
        try:
            Chem.SanitizeMol(new_mol)
            ind = OrganicCatalyst(new_mol)
            return ind
        except Exception:
            return None

    def run(self):
        results = []  # here the best individuals of each generation will be stored
        self.print_parameters()
        self.population = self.make_initial_population()
        self.population = self.calculate_scores(self.population, gen_id=0)
        self.db.add_individuals(0, self.population)
        self.print_population(self.population, 0)
        for n in range(0, self.n_generations):
            self.calculate_fitness(self.population)
            self.db.add_generation(n, self.population)
            self.append_results(results, gennum=n, detailed=True)
            children = self.reproduce(self.population, n + 1)
            children = self.calculate_scores(children, gen_id=n + 1)
            self.db.add_individuals(n + 1, children)
            self.population = self.prune(self.population + children)
            self.print_population(self.population, n + 1)
        self.calculate_fitness(self.population)
        self.db.add_generation(n + 1, self.population)
        self.append_results(results, gennum=n + 1, detailed=True)
        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ga = GraphGA(
        mol_options=MoleculeOptions(OrganicCatalyst),
        population_size=10,
        n_generations=15,
        mutation_rate=0.5,
        db_location="organic.sqlite",
        scoring_kwargs={},
    )

    results = ga.run()

    generations = [r[0] for r in results]
    best_scores = [max([ind.score for ind in res[1]]) for res in results]

    fig, ax = plt.subplots()
    ax.plot(generations, best_scores)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Max Score")

    plt.savefig("organic.png")
