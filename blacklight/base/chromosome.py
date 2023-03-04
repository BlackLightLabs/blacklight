from collections import OrderedDict
from abc import ABC, abstractmethod


class Chromosome(ABC):
    """
    Abstract class for a chromosome in an individual. Implements common traits between all chromosomes.
    """

    def __init__(self):
        self.genes = None

    @abstractmethod
    def _random_genes(self):
        """
        Randomly initialize genes for each type of chromosome if there are no parents.
        """
        pass

    @abstractmethod
    def _crossover(self, other_chromosome):
        """
        Crossover genes from parents to create new genes.
        """
        pass

    @abstractmethod
    def _mutate(self):
        """
        Mutate genes of this chromosome.
        """
        pass


