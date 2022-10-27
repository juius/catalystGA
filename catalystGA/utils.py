import pickle
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

import rdkit

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


def sqltype(value):
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, str):
        return "TEXT"
    else:
        return "BLOB"


def param_types(value):
    if isinstance(value, list):
        value = str(value)
    elif isinstance(value, dict):
        value = str(value)
    elif isinstance(value, float):
        value = float(value)
    elif isinstance(value, int):
        value = int(value)
    else:
        value = str(value)
    return value


def adapt_mol(mol):
    return pickle.dumps(mol)


sqlite3.register_adapter(
    rdkit.Chem.rdchem.Mol, adapt_mol
)  # cannot use pickle.dumps directly because of inadequate argument signature
sqlite3.register_converter("rdkit.Chem.rdchem.Mol", pickle.loads)


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
              energy_diff REAL,
              sa_score REAL,
              error TEXT,
              timing REAL,
              structure1 BLOB,
              structure2 BLOB
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

    def add_generation(self, genid, population):
        """add a generation to the database"""
        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO generations (idx, smiles, fitness)
            VALUES (?, ?, ?)
            """,
                [(genid, ind.smiles, ind.fitness) for ind in population],
            )

    def add_individuals(self, genid, population):
        """add individuals to the database (smiles, score, error, timing and structures as pickled blobs)"""

        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO individuals (idx, smiles, score, error, timing, structure1, structure2)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        genid,
                        ind.smiles,
                        ind.score,
                        ind.error,
                        ind.timing,
                        ind.structure1,
                        ind.structure2,
                    )
                    for ind in population
                ],
            )

    def write_parameters(self, params):
        """write parameters to database"""
        with self.connection:
            self.cur.execute(
                f"""CREATE TABLE IF NOT EXISTS parameters(
                    {', '.join([f'{key} {sqltype(value)}' for key, value in params.items()])}
                    )"""
            )
            insert_params = "INSERT INTO parameters ({}) VALUES ({})".format(
                ",".join(params), ",".join(["?"] * len(params))
            )
            self.cur.execute(insert_params, tuple(params.values()))
