import math
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from rdkit import Chem

from catalystGA.default_params import DEFAULTS
from catalystGA.organometallics.components import BaseCatalyst


@dataclass
class MoleculeOptions:
    """Set average size and standard deviation of molecule"""

    individual_type: BaseCatalyst
    average_size: int = 5
    size_std: int = 5


@dataclass
class ScoringOptions:
    """Set parameters used during scoring"""

    n_cores: int = DEFAULTS["N_CORES"]
    parallel: bool = True
    timeout_min: int = DEFAULTS["TIMEOUT_MIN"]
    # needs to be same as in calculate_score of individual
    cpus_per_task: int = DEFAULTS["CPUS_PER_TASK"]
    slurm_partition: str = DEFAULTS["SLURM_PARTITION"]
    slurm_array_parallelism: int = field(init=False)

    def __post_init__(self):
        self.slurm_array_parallelism = self.n_cores // self.cpus_per_task


def catch(func: Callable, *args, handle=lambda e: e, **kwargs):
    """Catches an exception handels it

    Args:
        func (Callable): Function to evaluate
        handle (Callable, optional): What to return if Exception.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


class GADatabase(object):
    def __init__(self, location: str) -> None:
        """Initialize db class variables"""
        self.connection = sqlite3.connect(location)
        self.cur = self.connection.cursor()

    def commit(self):
        """commit changes to database"""
        self.connection.commit()

    def exists(self, smiles: str) -> bool:
        """check if a smiles exists in the database"""
        with self.connection:
            self.cur.execute(
                f"""
                    SELECT
                    EXISTS(
                        SELECT 1
                        FROM individuals
                        WHERE smiles="{smiles}"
                        LIMIT 1
                        );
                    """
            )
            return bool(self.cur.fetchone()[0])

    def create_tables(self) -> None:
        """create a database table if it does not exist already"""
        self.cur.execute(
            """
          CREATE TABLE IF NOT EXISTS individuals (
              idx INTEGER,
              smiles TEXT,
              score REAL,
              error TEXT,
              timing REAL,
              structure1 TEXT,
              structure2 TEXT
            )
          """
        )

        self.cur.execute(
            """
          CREATE TABLE IF NOT EXISTS generations (
              idx INTEGER,
              smiles TEXT,
              fitness REAL
            )
          """
        )
        self.commit()

    def add_generation(self, genid, population):
        """add a generation to the database"""
        ids = [genid for i in population]
        smiles = [i.smiles for i in population]
        try:
            fitness = [i.fitness for i in population]
        except AttributeError:
            fitness = [math.nan for _ in population]
        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO generations (idx, smiles, fitness)
            VALUES (?, ?, ?)
            """,
                [(i, smi, f) for i, smi, f in zip(ids, smiles, fitness)],
            )

    def add_individuals(self, genid, population):
        """add individuals to the database (smiles, score, error, timing and structures as sdf string)"""
        tmp = []
        for ind in population:
            try:
                struc1 = Chem.MolToMolBlock(ind.structure1)
                struc2 = Chem.MolToMolBlock(ind.structure2)
            except AttributeError:
                struc1 = ""
                struc2 = ""
            tmp.append(
                (
                    genid,
                    ind.smiles,
                    ind.score,
                    ind.error,
                    ind.timing,
                    struc1,
                    struc2,
                )
            )

        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO individuals (idx, smiles, score, error, timing, structure1, structure2)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                tmp,
            )
