from blacklight.genetic.base import Individual
from blacklight.genetic.base import Population
from blacklight.genetic.chromosomes import FeedForwardChromosome
from blacklight.engine import ModelConfig


class FeedForwardIndividual(Individual):
    """
    Individual for a feed forward neural network.

    Attributes:
        model_options (ModelConfig): ModelConfig Object containing options for the model (e.g. number of layers, number of neurons, etc.).
        problem_type (str): The problem type that the population is solving.
        num_classes (int): The number of classes for classification problems.
        model (Optional[Keras Model]): Keras model representing the individual's neural network.
        model_history (Optional[History]): Keras model training history.
        train_data (Optional[BlacklightDataset]): Training dataset.
        test_data (Optional[BlacklightDataset]): Testing dataset.
        chromosome (FeedForwardChromosome): Chromosome representing the individual's neural network topology.

    Args:
        options (ModelConfig): ModelConfig Object containing options for the model (e.g. number of layers, number of neurons, etc.).
        population (Population): The population object that this individual belongs to.
        parents_genes (Optional[Tuple[FeedForwardChromosome, FeedForwardChromosome]]): List of genes from parents.
    Examples:
        >>> options = ModelConfig(...)
        >>> population = Population(...)
        >>> individual1 = FeedForwardIndividual(options, population)
        >>> individual2 = FeedForwardIndividual(options, population)

        >>> fitness1 = individual1.get_fitness()
        >>> fitness2 = individual2.get_fitness()

        >>> child = individual1.mate(individual2)
        >>> child_fitness = child.get_fitness()
    """

    def __init__(
            self,
            options: ModelConfig,
            population: Population,
            parents_genes=None,
    ):
        """
        Initializes a FeedForwardIndividual.
        """
        super().__init__(parents_genes, population)
        self.model_options = options
        self.problem_type = population.problem_type
        self.num_classes = population.num_classes
        # Class parameters

        self.model = None
        self.model_history = None

        # data
        self.train_data = population.get_training_data()
        self.test_data = population.get_testing_data()

        # Create chromosome if parents genes are not provided
        if self.need_new_genes:
            self.chromosome = FeedForwardChromosome(
                self.model_options,
                genes=None,
                mutation_prob=0.1,
            )
        # Otherwise, create chromosome from parents genes
        else:
            self.chromosome = self._crossover(self.parents_genes)

    def _crossover(self, parents_chromosomes):
        """
        Perform crossover between the chromosomes of two parents to create a new chromosome.

        Args:
            parents_chromosomes (Tuple[FeedForwardChromosome, FeedForwardChromosome]): A tuple of parent chromosomes.

        Returns:
            FeedForwardChromosome: A new chromosome created by crossing over the parent chromosomes.

        """
        parent_one_chrome, parent_two_chrome = parents_chromosomes
        return FeedForwardChromosome.cross_over(
            parent_one_chrome, parent_two_chrome)

    def get_fitness(self):
        """
        Get the fitness of the individual.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the model
        self.model = self.chromosome.get_model()
        fitness = self.chromosome.evaluate_model(
            self.train_data, self.test_data)
        self.fitness = fitness
        self.model_history = self.chromosome.get_model_history()

        return fitness

    def mate(self, other_individual):
        """
        Mate with another individual to produce a child.

        Args:
            other_individual (FeedForwardIndividual): The individual to mate with.

        Returns:
            FeedForwardIndividual: A new child individual created from the parents.
        """
        # Get genes from parents
        parents_genes = (self.chromosome, other_individual.chromosome)
        # Create child
        child = FeedForwardIndividual(
            parents_genes=parents_genes,
            population=self.population,
            options=self.model_options)
        # Return child
        return child
