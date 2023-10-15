import concurrent.futures
import sqlite3
import textwrap
from dataclasses import dataclass
from typing import Optional

from rdkit.Chem import rdMolDescriptors

from catalystGA.components import BaseCatalyst
from catalystGA.xtb import xtb_calculate


@dataclass
class MoleculeOptions:

    individual_type: BaseCatalyst
    min_size: int = 1
    max_size: int = 30
    num_rotatable_bonds: Optional[int] = 5

    def check(self, mol):
        if not mol:
            return False
        elif mol.GetNumHeavyAtoms() < self.min_size or mol.GetNumHeavyAtoms() > self.max_size:
            return False
        elif (
            self.num_rotatable_bonds
            and rdMolDescriptors.CalcNumRotatableBonds(mol) > self.num_rotatable_bonds
        ):
            return False
        else:
            return True


class GADatabase(object):
    def __init__(self, location: str, cat_type) -> None:
        """Initialize db class variables."""
        self.connection = sqlite3.connect(location)
        self.cur = self.connection.cursor()
        self.cat_type = cat_type

    def commit(self):
        """commit changes to database."""
        self.connection.commit()

    def exists(self, smiles: str) -> bool:
        """check if a smiles exists in the database."""
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
        """create a database table if it does not exist already."""
        table = """
            idx TEXT,
            smiles TEXT,
            score REAL,
            timing REAL,
            status TEXT
            """
        for key, value in self.cat_type.save_attributes.items():
            table += f",\n{key} {value}"
        self.cur.execute(
            f"""
          CREATE TABLE IF NOT EXISTS individuals (
              {table}
            )
          """
        )

        self.cur.execute(
            """
          CREATE TABLE IF NOT EXISTS generations (
              generation INTEGER,
              idx TEXT,
              smiles TEXT,
              fitness REAL
            )
          """
        )

    def add_generation(self, genid, population):
        """add a generation to the database."""
        with self.connection:
            self.cur.executemany(
                """
            INSERT INTO generations (generation, idx, smiles, fitness)
            VALUES (?, ?, ?, ?)
            """,
                [
                    (genid, f"{ind.idx[0]:03d}-{ind.idx[1]:03d}", ind.smiles, ind.fitness)
                    for ind in population
                ],
            )

    def add_individuals(self, genid, population):
        """add individuals to the database (idx, smiles, score, timing, status
        and 'save_attributes')"""
        columns = ["idx", "smiles", "score", "timing", "status"]
        columns += list(self.cat_type.save_attributes.keys())
        with self.connection:
            self.cur.executemany(
                f"""
            INSERT INTO individuals ({', '.join(columns)})
            VALUES ({', '.join(['?'] * len(columns))})
            """,
                [
                    (
                        f"{ind.idx[0]:03d}-{ind.idx[1]:03d}",
                        ind.smiles,
                        ind.score,
                        ind.timing,
                        ind.error,
                    )
                    + tuple(
                        [
                            ind.__getattribute__(key)
                            for key in list(self.cat_type.save_attributes.keys())
                        ]
                    )
                    for ind in population
                ],
            )


def str_table(title=None, headers=[], data=[], column_widths=[75, 14], percision=4, frame=True):
    table = sum(column_widths) * "=" + "\n" if frame else ""
    if title:
        table += textwrap.fill(title, width=len(title)).center(sum(column_widths)) + "\n"

    for i, column in enumerate(headers):
        table += f"{column:<{column_widths[i]}}"
    table += "\n" + sum(column_widths) * "-" + "\n"

    for row in zip(*data):
        for i, d in enumerate(row):
            if isinstance(d, float):
                d = f"{d:.0{percision}f}"
            table += f"{str(d)[:column_widths[i]-2]:<{column_widths[i]}}"
        table += "\n"
    table += sum(column_widths) * "=" + "\n" if frame else ""
    return table


def optimize(args, workers):
    """Do paralell optimization of all the entries in args."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            xtb_calculate,
            [arg[0] for arg in args],
            [arg[1] for arg in args],
            [arg[2] for arg in args],
            [arg[3] for arg in args],
            [arg[4] for arg in args],
        )
    return results
