import math
import sqlite3
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem

from catalystGA.organometallics.components import BaseCatalyst


class LoggingFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


@dataclass
class MoleculeOptions:
    individual_type: BaseCatalyst
    average_size: int = 5
    size_std: int = 5


@dataclass
class ScoringOptions:
    n_cores: int = 8
    parallel: bool = True
    timeout_min: int = 30
    # needs to be same as in calcualte_score of individual
    cpus_per_task: int = 1
    slurm_partition: str = "kemi1"
    slurm_array_parallelism: int = field(init=False)

    def __post_init__(self):
        self.slurm_array_parallelism = self.n_cores // self.cpus_per_task


def read_ga_results(csv_file):
    df = pd.read_csv(csv_file)
    df.set_index(["generation", "idx"], inplace=True)
    return df


class GADatabase(object):
    def __init__(self, location):
        """Initialize db class variables"""
        self.connection = sqlite3.connect(location)
        self.cur = self.connection.cursor()

    def close(self):
        """close sqlite3 connection"""
        self.connection.close()

    def execute(self, new_data):
        """execute a row of data to current cursor"""
        self.cur.execute(new_data)

    def executemany(self, many_new_data):
        """add many new data to database in one go"""
        self.cur.executemany("REPLACE INTO jobs VALUES(?, ?, ?, ?)", many_new_data)

    def commit(self):
        """commit changes to database"""
        self.connection.commit()

    def create_tables(self):
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
        ids = [genid for i in population]
        smiles = [i.smiles for i in population]
        try:
            fitness = [i.fitness for i in population]
        except AttributeError:
            fitness = [math.nan for i in population]
        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO generations (idx, smiles, fitness)
            VALUES (?, ?, ?)
            """,
                [(i, smi, f) for i, smi, f in zip(ids, smiles, fitness)],
            )

    def add_individuals(self, genid, population):
        tmp = []
        for ind in population:
            try:
                struc1 = Chem.MolToMolBlock(ind.structures[0])
                struc2 = Chem.MolToMolBlock(ind.structures[1])
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
