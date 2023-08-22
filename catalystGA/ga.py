import datetime
import math
import os
import random
import shutil
import time
import uuid
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path

import numpy as np
import submitit
import tomli

from catalystGA.utils import GADatabase, MoleculeOptions, str_table


class GA(ABC):

    DB_LOCATION = f"ga_{time.strftime('%Y-%m-%d_%H-%M')}.sqlite"

    def __init__(
        self,
        mol_options: MoleculeOptions,
        population_size=25,
        n_generations=50,
        maximize_score=True,
        selection_pressure=1.5,
        mutation_rate=0.5,
        donor_atoms_smarts_match=False,
        db_location=DB_LOCATION,
        config_file="./config.toml",
        scoring_args={},
    ):
        self.mol_options = mol_options
        self.population_size = population_size
        self.n_generations = n_generations
        self.maximize_score = maximize_score
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.donor_atoms_smarts_match = donor_atoms_smarts_match
        self.config_file = config_file
        self.scoring_args = scoring_args
        self.db = GADatabase(db_location, cat_type=mol_options.individual_type)
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
        """Checks if methods `calculate_score` and `__eq__` are implemented in
        the individual class.

        Raises:
            NotImplementedError: If methods are not implemented
        """
        for fct in ["calculate_score", "__eq__"]:
            if not hasattr(self.mol_options.individual_type, fct):
                raise NotImplementedError(
                    f"{fct} is not implemented for {self.mol_options.individual_type.__name__}"
                )

        with open(self.config_file, mode="rb") as fp:
            self.config = tomli.load(fp)

    def calculate_scores(self, population: list, gen_id: int) -> list:

        """Calculates scores for all individuals in the population.

        Args:
            population (List): List of individuals

        Returns:
            population: List of individuals with scores
        """

        def _wrap_scoring(individual, n_cores, envvar_scratch, scoring_args):
            print(f"Scoring individual {individual.idx}")
            print(f"Scoring args: {scoring_args}")
            individual.calculate_score(n_cores, envvar_scratch, **scoring_args)
            print(individual.score)
            return individual

        scoring_temp_dir = self.config["slurm"]["tmp_dir"] + "_" + str(uuid.uuid4())
        executor = submitit.AutoExecutor(
            folder=scoring_temp_dir,
        )
        executor.update_parameters(
            name=f"scoring_{gen_id}",
            cpus_per_task=self.config["slurm"]["cpus_per_task"],
            timeout_min=self.config["slurm"]["timeout_min"],
            slurm_partition=self.config["slurm"]["queue"],
            slurm_array_parallelism=self.config["slurm"]["array_parallelism"],
        )
        jobs = executor.map_array(
            _wrap_scoring,
            population,
            [self.config["slurm"]["cpus_per_task"]] * len(population),
            [self.config["slurm"]["envvar_scratch"]] * len(population),
            [self.scoring_args] * len(population),
        )
        # read results, if job terminated with error then return individual without score
        new_population = []
        for i, job in enumerate(jobs):
            error = "Normal termination"
            try:
                cat = job.result()
            except Exception as e:
                error = f"Exception: {e}\n"
                error += f"{job.stderr()}"
                cat = population[i]
            finally:
                cat.error = error
            new_population.append(cat)
        population = new_population

        self.sort_population(population, self.maximize_score)

        try:
            shutil.rmtree(scoring_temp_dir)
        except FileNotFoundError:
            pass

        return population

    def calculate_fitness(self, population: list) -> None:
        """Calculates fitness of all individuals in the population.

        Args:
            population (list): List of individuals
        """

        # Sort by score
        self.sort_population(population, self.maximize_score)

        ranks = list(reversed(range(len(population))))
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
            if child and self.mol_options.check(child.mol):
                child.parents = str((parent1.idx, parent2.idx))
                if random.random() <= self.mutation_rate:
                    child = self.mutate(child)
                    if child:
                        child.mutated = int(True)
                if (
                    child
                    and self.mol_options.check(child.mol)
                    and child not in children
                    and not self.db.exists(child.smiles)
                ):
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
        self.sort_population(tmp, self.maximize_score)

        return tmp[: self.population_size]

    def find_all_donor_atoms(
        self, population: list, smarts_match=False, reference_smiles="[Pd]<-P", n_cores=1
    ):
        """Find all donor atoms in the population.

        Args:
            population (list): List of all individuals

        Returns:
            list: List of donor atoms
        """

        # skip all of this if donor atoms are already known from SMARTS match
        if self.donor_atoms_smarts_match:
            return None

        def _wrap_find_donor_atoms(
            individual, smarts_match, reference_smiles, n_cores, envvar_scratch
        ):
            # Setup scrach directory
            scratch = os.environ.get(envvar_scratch, ".")
            calc_dir = Path(scratch)
            jobid = os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
            calc_dir = calc_dir / jobid
            calc_dir.mkdir(exist_ok=True)
            for ligand in individual.ligands:
                ligand.find_donor_atom(smarts_match, reference_smiles, n_cores, calc_dir)
            return [ligand.donor_id for ligand in individual.ligands]

        temp_dir = self.config["slurm"]["tmp_dir"] + "_" + str(uuid.uuid4())
        executor = submitit.AutoExecutor(
            folder=temp_dir,
        )
        executor.update_parameters(
            name="find_donor_atoms",
            cpus_per_task=min([4, self.config["slurm"]["cpus_per_task"]]),
            timeout_min=self.config["slurm"]["timeout_min"],
            slurm_partition=self.config["slurm"]["queue"],
            slurm_array_parallelism=self.config["slurm"]["array_parallelism"],
        )
        jobs = executor.map_array(
            _wrap_find_donor_atoms,
            population,
            [smarts_match] * len(population),
            [reference_smiles] * len(population),
            [min([4, self.config["slurm"]["cpus_per_task"]])] * len(population),
            [self.config["slurm"]["envvar_scratch"]] * len(population),
        )
        # read results, if job terminated with error then return the donor_id from smarts matching
        for i, job in enumerate(jobs):
            try:
                donor_ids = job.result()
                # update donor id
                for ligand, donor_id in zip(population[i].ligands, donor_ids):
                    ligand.donor_id = donor_id
            except Exception as e:
                print(f"Coulnd't find donor atoms for {population[i].smiles} with error {e}")

        try:
            shutil.rmtree(temp_dir)
        except FileNotFoundError:
            pass

    @staticmethod
    def sort_population(population: list, maximize_score: bool) -> list:
        """Sorts the population by score, if score is NaN then it is always
        last.

        Args:
            population (list): List of all individuals

        Returns:
            list: Sorted list of all individuals
        """
        population.sort(
            key=lambda x: (maximize_score - 0.5) * float("-inf")
            if math.isnan(x.score)
            else x.score,
            reverse=maximize_score,
        )

    def append_results(self, results, gennum, detailed=False):
        if detailed:
            results.append((gennum, self.population))
        else:
            results.append((gennum, self.population[0]))
        return results

    @staticmethod
    def print_population(population, genid):
        smiles = [ind.smiles for ind in population]
        scores = [ind.score for ind in population]
        print(
            str_table(
                title=f"Generation {genid}",
                headers=["SMILES", "Score"],
                data=[smiles, scores],
                percision=4,
            )
        )

    def print_parameters(self):
        print(89 * "=")
        print(
            str_table(
                title="GA Parameters",
                headers=["Parameter", "Value"],
                data=[
                    [
                        "Population Size",
                        "Number of Generations",
                        "Mutation Rate",
                        "Selection Pressure",
                    ],
                    [
                        self.population_size,
                        self.n_generations,
                        self.mutation_rate,
                        self.selection_pressure,
                    ],
                ],
                frame=False,
            )
        )
        print(89 * "-")
        print(
            str_table(
                title="Molecule Options",
                headers=["Parameter", "Value"],
                data=[
                    ["Minimum Size", "Minimum Size"],
                    [self.mol_options.min_size, self.mol_options.max_size],
                ],
                frame=False,
            )
        )
        print(89 * "=" + "\n\n")

    @staticmethod
    def print_timing(start, end, time_per_gen, population):
        runtime = np.round(end - start)
        mean_spgen = np.round(np.mean(time_per_gen))
        std_spgen = np.round(np.std(time_per_gen))
        print(
            str_table(
                title="Timings",
                headers=[],
                data=[
                    ["Total Wall Time", "Mean Time per Generation"],
                    [
                        str(datetime.timedelta(seconds=runtime)),
                        f"{str(datetime.timedelta(seconds=mean_spgen))}+/-{str(datetime.timedelta(seconds=std_spgen))}",
                    ],
                ],
                frame=False,
            )
        )

    def run(self):
        # print parameters for GA and scoring
        start_time = time.time()
        self.print_parameters()
        results = []
        time_per_gen = []
        tmp_time = time.time()
        self.population = self.make_initial_population()
        # self.find_all_donor_atoms(self.population)
        self.population = self.calculate_scores(self.population, gen_id=0)
        self.db.add_individuals(0, self.population)
        self.print_population(self.population, 0)
        for n in range(0, self.n_generations):
            self.calculate_fitness(self.population)
            self.db.add_generation(n, self.population)
            self.append_results(results, gennum=n, detailed=True)
            children = self.reproduce(self.population, n + 1)
            self.find_all_donor_atoms(children)
            children = self.calculate_scores(children, gen_id=n + 1)
            self.db.add_individuals(n + 1, children)
            self.population = self.prune(self.population + children)
            self.print_population(self.population, n + 1)
            time_per_gen.append(time.time() - tmp_time)
            tmp_time = time.time()
        self.calculate_fitness(self.population)
        self.db.add_generation(n + 1, self.population)
        self.append_results(results, gennum=n + 1, detailed=True)
        self.print_timing(start_time, time.time(), time_per_gen, self.population)

        return results

    @classmethod
    def load(cls, db_location, mol_options, scoring_args={}):
        return cls(
            mol_options=mol_options,
            population_size=25,
            n_generations=50,
            maximize_score=True,
            selection_pressure=1.5,
            mutation_rate=0.5,
            donor_atoms_smarts_match=False,
            db_location=db_location,
            config_file="./config.toml",
            scoring_args=scoring_args,
        )

    def rerun(self, n_extra_generations=25):
        self.population
