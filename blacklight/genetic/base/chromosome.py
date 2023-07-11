from abc import ABC, abstractmethod
from typing import Tuple


class Chromosome(ABC):
    """
    Abstract class for a chromosome in an individual. Implements common traits between all chromosomes.
    """

    def __init__(self):
        self.genes = None
        self.length = None

    @abstractmethod
    def _random_genes(self):
        """
        Randomly initialize genes for each type of chromosome if there are no parents.
        """
        pass

    @abstractmethod
    def cross_over(self, *args):
        """
        Handle cross over between two chromosomes.
        """
        pass

    @abstractmethod
    def _mutate(self):
        """
        Mutate genes of this chromosome.
        """
        pass

    @staticmethod
    def get_shortest_chromosome(
            chromosome_a: object, chromosome_b: object) -> Tuple[object, object]:
        """
        Determine the shortest and longest chromosomes.

        Args:
            chromosome_a (BaseChromosome): The first chromosome to compare.
            chromosome_b (BaseChromosome): The second chromosome to compare.

        Returns:
            Tuple[BaseChromosome, BaseChromosome]: A tuple containing the shortest chromosome and the longest chromosome, in that order.
        """
        if chromosome_a.length > chromosome_b.length:
            return chromosome_b, chromosome_a
        else:
            return chromosome_a, chromosome_b
