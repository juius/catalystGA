from dataclasses import dataclass

from catalystGA.organometallics.components import BaseCatalyst


class LoggingFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


@dataclass
class MoleculeOptions:
    individual_type: BaseCatalyst
    average_size: int = 5
    size_std: int = 5
