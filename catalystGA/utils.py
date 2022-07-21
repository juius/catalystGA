from dataclasses import dataclass, field

import pandas as pd

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
