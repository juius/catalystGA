import copy
import random
from typing import List

from rdkit import Chem
from suzuki import SuzukiCatalyst

from catalystGA import GA, DativeLigand, Metal, X_Ligand
from catalystGA.reproduction_utils import graph_crossover, graph_mutate
from catalystGA.utils import MoleculeOptions


class GraphGA(GA):
    def __init__(
        self,
        mol_options: MoleculeOptions,
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
    def crossover(ind1: SuzukiCatalyst, ind2: SuzukiCatalyst) -> SuzukiCatalyst or None:
        """Crossover the graphs of two ligands of SuzukiCatalysts."""
        ind_type = type(ind1)
        # choose one ligand at random from ind1 and crossover with random ligand from ind2, then replace this ligand in ind1 with new ligand
        ind1_ligands = copy.deepcopy(ind1.ligands)
        new_mol = None
        counter = 0
        while not new_mol:
            idx1 = random.randint(0, len(ind1_ligands) - 1)
            idx2 = random.randint(0, len(ind2.ligands) - 1)
            new_mol = graph_crossover(ind1.ligands[idx1].mol, ind2.ligands[idx2].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            # this will catch if new_mol has no donor atom
            new_ligand = DativeLigand(new_mol)
            ind1_ligands[idx1] = new_ligand
            child = ind_type(ind1.metal, ind1_ligands)
            child.assemble()
            return child
        except Exception:
            return None

    @staticmethod
    def mutate(ind: SuzukiCatalyst) -> SuzukiCatalyst or None:
        """Mutate the graph of one ligand of a SuzukiCatalyst."""
        # pick one ligand at random, mutate and replace in ligand list
        idx = random.randint(0, len(ind.ligands) - 1)
        new_mol = None
        counter = 0
        while not new_mol:
            new_mol = graph_mutate(ind.ligands[idx].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            ind.ligands[idx] = DativeLigand(new_mol)
            ind.assemble()
            return ind
        except Exception:
            return None


if __name__ == "__main__":

    # read ligands
    ligands_list = []
    with open("/home/magstr/Documents/genetic_algorithm/data/ligands.smi", "r") as f:
        for line in f:
            ligands_list.append(X_Ligand(Chem.MolFromSmiles(line.rstrip())))

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
    ga = GraphGA(
        mol_options=mol_options,
        scoring_options=scoring_options,
        population_size=5,
        n_generations=2,
    )

    # Run the GA
    results = ga.run()
