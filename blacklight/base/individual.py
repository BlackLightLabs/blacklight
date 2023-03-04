from abc import ABC, abstractmethod


def _check_individual_inputs(parents, population) -> bool:
    """
    1. Population can not be None
    2. If parents are not None, they must be a list of individuals.
    """
    if population is None:
        raise ValueError("Population can not be None.")

    if parents is not None:
        if not isinstance(parents, list):
            raise ValueError("Parents must be a list of individuals.")
        if not all(isinstance(genes, dict) for genes in parents):
            raise ValueError("Parents must be a list of individuals.")

    if parents is None:
        return True
    return False


class Individual(ABC):
    """
    Abstract class for an individual in a population. Implements common traits between all individuals.
    An individual is a KERAS model.
    Genes are passed down from parents, or instantiated randomly.
    Individual is a base class that implements the basic functionality of an individual.
    Child classes hold specific information for different types of models.
    """

    fitness = 0

    def __init__(self, parents_genes=None, population=None):
        self.need_new_genes = _check_individual_inputs(
            parents_genes, population)
        self.population = population
        self.parents_genes = parents_genes
        self.genes = None
        # Check inputs
        self.NUM_PARENTS = None

    @abstractmethod
    def _random_genes(self):
        """
        Randomly initialize genes for each type of individual if there are no parents.
        """
        pass

    @abstractmethod
    def _crossover(self):
        """
        Crossover genes from parents to create new genes.
        """
        pass

    @abstractmethod
    def _mutate(self):
        """
        Mutate genes of this individual.
        """
        pass

    @abstractmethod
    def _make_model(self):
        """
        Make a Keras model from this individual's genes.
        """
        pass

    @abstractmethod
    def get_fitness(self):
        """
        Get fitness of this individual.
        """
        pass
