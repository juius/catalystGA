import datetime
import inspect
import math
import random
import shutil
import time
from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np
import submitit
from tabulate import tabulate

from catalystGA.filters.size import check_n_heavy_atoms
from catalystGA.utils import GADatabase, MoleculeOptions, ScoringOptions, param_types


def rank(list):
    # return [sorted([x if not math.isnan(x) else math.inf for x in list ]).index(x) for x in list]
    clean_list = [x if not math.isnan(x) else math.inf for x in list]
    return [index for element, index in sorted(zip(clean_list, range(len(list))))]


DB_LOCATION = f"ga_{time.strftime('%Y-%m-%d_%H-%M')}.sqlite"


class GA(ABC):
    def __init__(
        self,
        mol_options: MoleculeOptions,
        scoring_options: ScoringOptions = ScoringOptions(),
        population_size=5,
        n_generations=10,
        maximize_score=True,
        selection_pressure=1.5,
        mutation_rate=0.5,
        db_location=DB_LOCATION,
    ):
        self.mol_options = mol_options
        self.scoring_options = scoring_options
        self.population_size = population_size
        self.n_generations = n_generations
        self.maximize_score = maximize_score
        self.selection_pressure = selection_pressure

        self.mutation_rate = mutation_rate
        self.db = GADatabase(db_location)
        self.db.create_tables()
        self.health_check()

    @abstractmethod
    def make_initial_population(self):
        pass

    @abstractstaticmethod
    def crossover(ind1, ind2):
        pass

    @abstractstaticmethod
    def mutate(ind):
        pass

    def health_check(self):
        """Checks if methods 'calculate_score' and '__eq__' are implemented in
        the individual class.

        Raises:
            NotImplementedError: If methods are not implemented
        """
        for fct in ["calculate_score", "__eq__"]:
            if not hasattr(self.mol_options.individual_type, fct):
                raise NotImplementedError(
                    f"{fct} is not implemented for {self.mol_options.individual_type.__name__}"
                )

    @staticmethod
    def wrap_scoring(individual, results_dir):
        print(f"Scoring individual {individual.idx}")
        individual.calculate_score(results_dir=results_dir)
        print(individual.score)
        return individual

    def calculate_scores(self, population: list) -> list:

        """Calculates scores for all individuals in the population.

        Args:
            population (List): List of individuals

        Returns:
            population: List of individuals with scores
        """
        if not self.scoring_options.parallel:
            for ind in population:
                ind.calculate_score()
        else:
            # WORK IN PROGRESS
            executor = submitit.AutoExecutor(
                folder="_scoring_tmp",
            )
            executor.update_parameters(
                tasks_per_node=self.scoring_options.cpus_per_task,
                cpus_per_task=self.scoring_options.n_cores,
                timeout_min=self.scoring_options.timeout_min,
                slurm_partition=self.scoring_options.slurm_partition,
                slurm_array_parallelism=self.scoring_options.slurm_array_parallelism,
            )
            jobs = executor.map_array(
                self.wrap_scoring,
                population,
                [self.scoring_options.results_dir for _ in population],
            )
            # read results, if job terminated with error then return individual without score
            # population = [
            #     catch(job.result, handle=lambda e: population[i]) for i, job in enumerate(jobs)
            # ]
            for job in jobs:
                print(job.results())
        # sort population based on score, if score is NaN then it is always last
        population.sort(
            key=lambda x: (self.maximize_score - 0.5) * float("-inf")
            if math.isnan(x.score)
            else x.score,
            reverse=self.maximize_score,
        )

        try:
            shutil.rmtree("_scoring_tmp")
        except FileNotFoundError:
            pass
        return population

    def calculate_fitness(self, population: list) -> None:
        """Calculates fitness of all individuals in the population.

        Args:
            population (list): List of individuals
        """
        scores = [ind.score for ind in population]
        ranks = rank(scores)
        fitness = [
            2
            - self.selection_pressure
            + (2 * (self.selection_pressure - 1) * (rank - 1) / (len(ranks) - 1))
            for rank in ranks
        ]
        normalized_fitness = [float(i) / sum(fitness) for i in fitness]
        for ind, fitness in zip(population, normalized_fitness):
            ind.fitness = fitness

    def reproduce(self, population: list, genid: int) -> list:
        """Creates new offspring from the population.

        Args:
            population (list): List of individuals (parents)

        Returns:
            list: List of new individuals (children)
        """
        children = []
        fitness = [ind.fitness for ind in population]
        ind_idx = 0
        while len(children) < self.population_size:
            parent1, parent2 = np.random.choice(population, p=fitness, size=2, replace=False)
            child = self.crossover(parent1, parent2)
            if child and check_n_heavy_atoms(child.mol, self.mol_options):
                child.parents = (parent1.idx, parent2.idx)
                if random.random() <= self.mutation_rate:
                    child = self.mutate(child)
                    if child:
                        child.mutated = True
                if child and child not in children and not self.db.exists(child.smiles):
                    child.idx = (genid, ind_idx)
                    children.append(child)
                    ind_idx += 1
        return children

    def prune(self, population: list) -> list:
        """Keep the best individuals in the population, cut down to
        'population_size'.

        Args:
            population (list): List of all individuals

        Returns:
            list: List of kept individuals
        """
        tmp = list(set(population))
        tmp.sort(
            key=lambda x: (self.maximize_score - 0.5) * float("-inf")
            if math.isnan(x.score)
            else x.score,
            reverse=self.maximize_score,
        )
        return tmp[: self.population_size]

    def append_results(self, results, gennum, detailed=False):
        if detailed:
            results.append((gennum, self.population))
        else:
            results.append((gennum, self.population[0]))
        return results

    @staticmethod
    def print_population(population, genid):
        print(
            tabulate(
                [[str(ind), ind.score] for ind in population],
                headers=[f"Generation {genid}", "Score"],
                tablefmt="simple",
                floatfmt=".05f",
                maxcolwidths=[64, None],
            )
            + "\n"
        )

    def print_parameters(self):
        params = []
        for param in inspect.getfullargspec(self.__init__)[0][1:]:
            if param == "mol_options":
                mol_params = [[key, val] for key, val in self.mol_options.__dict__.items()]
                print(f"###    Molecule Options    ###\n{tabulate(mol_params)}\n")
            elif param == "scoring_options":
                scoring_params = [[key, val] for key, val in self.scoring_options.__dict__.items()]
                print(f"###     Scoring Options    ###\n{tabulate(scoring_params)}\n")
            else:
                params.append([param, getattr(self, param)])
        print(f"###      GA Parameters     ###\n{tabulate(params)}\n")

    def params2dict(self):
        # create params dict
        blacklist = set(["db", "scoring_options"])
        params = {}
        for key in self.__dict__.keys():
            if key not in blacklist:
                value = self.__dict__[key]
                if isinstance(value, MoleculeOptions):
                    for k, v in value.__dict__.items():
                        params[k] = str(v)
                else:
                    params[key] = str(param_types(value))
        return params

    @staticmethod
    def print_timing(start, end, time_per_gen, population):
        if hasattr(population[0], "timing"):
            scoring_timing = [ind.timing for ind in population]
        else:
            scoring_timing = None
        runtime = np.round(end - start)
        mean_spgen = np.round(np.mean(time_per_gen))
        std_spgen = np.round(np.std(time_per_gen))
        if scoring_timing:
            mean_spsco = np.round(np.mean(scoring_timing))
            std_spsco = np.round(np.std(scoring_timing))
        times = [
            ["Overall Runtime", str(datetime.timedelta(seconds=runtime))],
            [
                "Mean Time per Generation",
                f"{str(datetime.timedelta(seconds=mean_spgen))}+/-{str(datetime.timedelta(seconds=std_spgen))}",
            ],
        ]
        if scoring_timing:
            times.append(
                [
                    "Mean Time per Scoring",
                    f"{str(datetime.timedelta(seconds=mean_spsco))}+/-{str(datetime.timedelta(seconds=std_spsco))}",
                ]
            )

        print(f"###         Timing       ###\n{tabulate(times)}\n")

    def run(self):
        # print parameters for GA and scoring
        # start_time = time.time()
        self.print_parameters()
        # self.db.write_parameters(self.params2dict())
        if hasattr(self.mol_options.individual_type, "print_scoring_params"):
            self.mol_options.individual_type.print_scoring_params()
        results = []
        time_per_gen = []
        tmp_time = time.time()
        self.population = self.make_initial_population()
        self.population = self.calculate_scores(self.population)
        # self.db.add_individuals(0, self.population)
        self.print_population(self.population, 0)
        for n in range(0, self.n_generations):
            self.calculate_fitness(self.population)
            # self.db.add_generation(n, self.population)
            self.append_results(results, gennum=n, detailed=True)
            children = self.reproduce(self.population, n + 1)
            children = self.calculate_scores(children)
            # self.db.add_individuals(n + 1, children)
            self.population = self.prune(self.population + children)
            self.print_population(self.population, n + 1)
            time_per_gen.append(time.time() - tmp_time)
            # tmp_time = time.time()
        self.calculate_fitness(self.population)
        # self.db.add_generation(n + 1, self.population)
        self.append_results(results, gennum=n + 1, detailed=True)
        # self.print_timing(start_time, time.time(), time_per_gen, self.population)

        return results
