import tensorflow as tf
from tensorflow import keras

from src.individuals.individual import Individual
from src.individuals.individualutils import *
from collections import OrderedDict
import random


class FeedForwardIndividual(Individual):
    """
    Individual for a feed forward neural network.
    Description of genes: 
     - Dictionary of layers. 
        {number of layers: int, layer_activation: str}
    """

    def __init__(self, parents_genes=None, population=None, MAX_NEURONS=64, MAX_LAYERS=10, NUM_PARENTS=2, **kwargs):
        super().__init__(parents_genes, population)
        # Feed Forward Neural Network parameters
        self.MAX_LAYERS = MAX_LAYERS
        self.MAX_NEURONS = MAX_NEURONS
        self.NUM_PARENTS = NUM_PARENTS
        # Keras parameters -> Maybe these need to be in the population, not the individual?
        self.EPOCHS = kwargs.get("EPOCHS", 1000)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 2048)
        self.VERBOSE = kwargs.get("VERBOSE", 0)
        self.OPTIMIZER = kwargs.get("OPTIMIZER", "adam")
        self.VALIDATION_SPLIT = kwargs.get("VALIDATION_SPLIT", None)
        self.OUTPUT_BIAS = kwargs.get("OUTPUT_BIAS", True)
        self.LOSS = kwargs.get("LOSS", keras.metrics.BinaryCrossentropy())
        self.METRICS = kwargs.get("METRICS", FeedForwardConstants.default_metrics)
        self.LEARNING_RATE = kwargs.get("LEARNING_RATE", .0001)
        # Training parameters
        self.EARLY_STOPPING = kwargs.get("EARLY_STOPPING", True)
        self.CALLBACKS = kwargs.get("CALLBACKS", FeedForwardConstants.default_early_stopping)
        # Class parameters
        self.dominant_gene = None
        self.genes = self._random_genes() if self.need_new_genes else self._crossover()
        self.model = None
        self.fitness_metric = kwargs.get("fitness_metric")
        # data
        self.X_train = population.get_X_train()
        self.y_train = population.get_y_train()
        self.X_test = population.get_X_test()
        self.y_test = population.get_y_test()

    def _random_genes(self):
        """
        Randomly initialize genes for each type of individual if there are no parents.
        Feed Forward Neural Networks are made up of layers. Each layer has a number of neurons and an activation function.
        Initialize genes as a dictionary of dictionaries. Each sub_dictionary entry represents a layer.
        """
        genes = {
            f"gene_{i}": OrderedDict({random.randint(1, self.MAX_NEURONS): random.choice(["relu", "sigmoid", "tanh"])
                                      for _ in range(self.MAX_LAYERS)})
            for i in range(2)}

        self.dominant_gene = random.choice(list(genes.keys()))
        return genes

    def _crossover(self):
        """
        Takes in a list of genes and returns a new set of genes
        for this individual.
        Models real crossover, see merge_genes for details.
        :param parents_genes: list of genes
        :return:
        """

        # Get crossover points
        length_of_smallest_parental_chromosome = get_min_length_chromosome(self.parents_genes)
        crossover_points = get_crossover_points_from_num_parents(self.NUM_PARENTS,
                                                                 length_of_smallest_parental_chromosome)

        # Define recombined genes
        recombinant_genes = merge_genes(self.parents_genes[0], self.parents_genes[1], crossover_points[0])

        # Get two genes from the recombinant genes
        selection_process = random.sample(list(recombinant_genes.keys()), 2)

        # Reassign for naming convention
        real_genes = {f"gene_{i}": recombinant_genes[selection_process[i]] for i in range(len(selection_process))}

        # Randomly select dominant gene
        self.dominant_gene = real_genes[random.choice(list(real_genes.keys()))]

        # After we cross over, we mutate.
        self._mutate()
        return real_genes

    def _mutate(self):
        """
        Mutate the genes of the individual.
        :return:
        """
        self.dominant_gene, _ = mutate_dominant_gene(self.dominant_gene, self.MAX_NEURONS)

    def get_fitness(self):
        """
        Get the fitness of the individual.
        """
        # Evaluate the model
        self._evaluate_model()
        return self.fitness

    def mate(self, other_individual):
        """
        Mate with another individual.
        :param other_individual: Individual to mate with.
        :return:
        """
        # Get genes from parents
        parents_genes = [self.genes, other_individual.genes]
        # Create child
        child = FeedForwardIndividual(parents_genes=parents_genes, population=self.population)
        # Return child
        return child

    def _make_model(self):
        """
        Make a keras model from the genes.
        Implementing Forward-Forward Neural Networks to save on computation time
        during the architecture search.
        :return:
        """

        feed_forward_model = keras.Sequential(
            [tf.keras.layers.Dense(self.X_train.shape[-1], activation='relu',
                                   input_shape=(self.X_train.shape[-1],))] +
            [tf.keras.layers.Dense(i, activation=self.dominant_gene.get(i)) for i in self.dominant_gene.keys()] +
            [keras.layers.Dense(1, activation='sigmoid')]
        )
        feed_forward_model.compile(
            optimizer=self.OPTIMIZER(lr=self.LEARNING_RATE),
            loss=self.LOSS,
            metrics=self.METRICS
        )
        self.model = feed_forward_model

    def _train_model(self):
        """
        Train the model.
        :return:
        """
        # Train the model
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            verbose=self.VERBOSE,
            validation_split=self.VALIDATION_SPLIT,
            callbacks=self.CALLBACKS if self.EARLY_STOPPING else None
        )

    def _evaluate_model(self):
        """
        Evaluate the model.
        """
        # First make the model
        self._make_model()
        # Then train the model
        self._train_model()
        # Then get the fitness
        results = self.model.evaluate(
            self.X_test,
            self.y_test,
            batch_size=self.BATCH_SIZE,
            verbose=self.VERBOSE,
            return_dict=True
        )
        # the fitness metric is extracted.
        fitness = results[self.fitness_metric]
        self.fitness = fitness
