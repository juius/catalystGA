import datetime
import inspect
import random
import shutil
import time

import numpy as np
import submitit
from tabulate import tabulate

from catalystGA.utils import MoleculeOptions, ScoringOptions, catch


def rank(list):
    return [sorted(list).index(x) for x in list]


class GA:
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
        self.mol_options = mol_options
        self.scoring_options = scoring_options
        self.population_size = population_size
        self.n_generations = n_generations
        self.maximize_score = maximize_score
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.health_check()

    def make_initial_population(self):
        raise NotImplementedError("make_initial_population not implemented for Base GA Class")

    @staticmethod
    def crossover(ind1, ind2):
        raise NotImplementedError("crossover not implemented for Base GA Class")

    @staticmethod
    def mutate(ind):
        raise NotImplementedError("mutate not implemented for Base GA Class")

    def health_check(self):
        for fct in ["calculate_score", "__eq__"]:
            if not hasattr(self.mol_options.individual_type, fct):
                raise NotImplementedError(
                    f"{fct} is not implemented for {self.mol_options.individual_type.__name__}"
                )

    @staticmethod
    def wrap_scoring(individual):
        individual.calculate_score()
        print(individual.score)
        return individual

    def calculate_scores(self, population):
        if not self.scoring_options.parallel:
            for ind in population:
                ind.calculate_score()
        else:
            executor = submitit.AutoExecutor(
                folder="_scoring_tmp",
            )
            executor.update_parameters(
                cpus_per_task=self.scoring_options.cpus_per_task,
                slurm_mem_per_cpu="1GB",
                timeout_min=self.scoring_options.timeout_min,
                slurm_partition=self.scoring_options.slurm_partition,
                slurm_array_parallelism=self.scoring_options.slurm_array_parallelism,
            )
            jobs = executor.map_array(self.wrap_scoring, population)
            population = [catch(job.result) for job in jobs]

        population.sort(key=lambda x: x.score, reverse=self.maximize_score)
        try:
            shutil.rmtree("_scoring_tmp")
        except FileNotFoundError:
            pass
        return population

    def calculate_fitness(self, population):
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

    def reproduce(self, population):
        children = []
        fitness = [ind.fitness for ind in population]
        while len(children) < self.population_size:
            parent1, parent2 = np.random.choice(population, p=fitness, size=2, replace=False)
            child = self.crossover(parent1, parent2, self.mol_options)
            if child:
                if random.random() <= self.mutation_rate:
                    child = self.mutate(child, self.mol_options)
                if child not in children:
                    if child:
                        children.append(child)
        return children

    def prune(self, population):
        tmp = list(set(population))
        tmp.sort(key=lambda x: x.score, reverse=self.maximize_score)
        return tmp[: self.population_size]

    def write_results(self, results, gennum, detailed=False):
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
            )
            + "\n"
        )

    def print_parameters(self):
        params = []
        for param in inspect.getfullargspec(self.__init__)[0][1:]:
            if param == "mol_options":
                mol_params = [[key, val] for key, val in self.mol_options.__dict__.items()]
            elif param == "scoring_options":
                scoring_params = [[key, val] for key, val in self.scoring_options.__dict__.items()]
            else:
                params.append([param, getattr(self, param)])
        print(f"###      GA Parameters     ###\n{tabulate(params)}\n")
        print(f"###    Molecule Options    ###\n{tabulate(mol_params)}\n")
        print(f"###     Scoring Options    ###\n{tabulate(scoring_params)}\n")

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
        start_time = time.time()
        self.print_parameters()
        if hasattr(self.mol_options.individual_type, "print_scoring_params"):
            self.mol_options.individual_type.print_scoring_params()
        results = []
        time_per_gen = []
        tmp_time = time.time()
        self.population = self.make_initial_population()
        self.population = self.calculate_scores(self.population)
        for n in range(1, self.n_generations + 1):
            self.calculate_fitness(self.population)
            self.write_results(results, gennum=n, detailed=True)
            self.print_population(self.population, n)
            children = self.reproduce(self.population)
            children = self.calculate_scores(children)
            self.population = self.prune(self.population + children)
            time_per_gen.append(time.time() - tmp_time)
            tmp_time = time.time()
        self.write_results(results, gennum=n + 1, detailed=True)
        self.print_population(self.population, n + 1)
        self.print_timing(start_time, time.time(), time_per_gen, self.population)
        return results
