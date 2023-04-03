import random
from typing import List

from rdkit import Chem
from suzuki import SuzukiCatalyst

from catalystGA import GA, L_Ligand, Metal
from catalystGA.reproduction_utils import list_crossover, list_mutate
from catalystGA.utils import MoleculeOptions, ScoringOptions


class LigandGA(GA):
    def __init__(
        self,
        mol_options: MoleculeOptions,
        scoring_options: ScoringOptions = ScoringOptions(),
        population_size=5,
        n_generations=10,
        maximize_score=True,
        selection_pressure=1.5,
        mutation_rate=0.5,
    ):
        super().__init__(
            mol_options=mol_options,
            scoring_options=scoring_options,
            population_size=population_size,
            n_generations=n_generations,
            maximize_score=maximize_score,
            selection_pressure=selection_pressure,
            mutation_rate=mutation_rate,
        )

    def make_initial_population(self) -> List[SuzukiCatalyst]:
        """Make initial population as a list of SuzukiCatalysts."""
        population = []
        while len(population) < self.population_size:
            metal = random.choice(metals_list)
            ligands = random.choices(ligands_list, k=self.mol_options.individual_type.n_ligands)
            cat = self.mol_options.individual_type(metal, ligands)
            # remove duplicates
            if cat not in population:
                population.append(cat)
        return population

    @staticmethod
    def crossover(ind1: SuzukiCatalyst, ind2: SuzukiCatalyst) -> SuzukiCatalyst:
        """Crossover the ligands of two SuzukiCatalysts."""
        ind_type = type(ind1)
        new_ligands = list_crossover(ind1.ligands, ind2.ligands, n_cutpoints=1)
        child = ind_type(ind1.metal, new_ligands)
        child.assemble()
        return child

    @staticmethod
    def mutate(ind: SuzukiCatalyst) -> SuzukiCatalyst:
        """Mutate one ligand of a SuzukiCatalyst."""
        new_ligands = list_mutate(ind.ligands, ligands_list)
        ind.ligands = new_ligands
        ind.assemble()
        return ind


if __name__ == "__main__":

    # read ligands
    ligands_list = []
    with open("data/ligands.smi", "r") as f:
        for line in f:
            ligands_list.append(L_Ligand(Chem.MolFromSmiles(line.rstrip())))

    metals_list = [Metal("Pd")]

    # Set Options for Molecule
    mol_options = MoleculeOptions(
        individual_type=SuzukiCatalyst,
        average_size=10,
        size_std=5,
    )

    # Set Options for Scoring
    scoring_options = ScoringOptions(n_cores=1000, parallel=True, cpus_per_task=4)

    # Initialize GA
    ga = LigandGA(
        mol_options=mol_options,
        scoring_options=scoring_options,
        population_size=5,
        n_generations=5,
    )

    # Run the GA
    results = ga.run()
