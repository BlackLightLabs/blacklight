import tensorflow as tf
from blacklight.base.individual import Individual
from blacklight.autoML.individuals.individualutils import FeedForwardConstants
from collections import OrderedDict


class ConvolutionalIndividual(Individual):
    """
    Individual for a convolutional neural network.

    Parameters:
            parents_genes: list of genes from parents
            population: population object that this individual belongs to
            MAX_NEURONS: maximum number of neurons allowed
            MAX_LAYERS: maximum number of layers allowed
            MAX_
            NUM_PARENTS: number of parents used for crossover
            **kwargs: arguments to be passed to Keras

    Returns:
        None
    """

    def __init__(
            self,
            parents_genes=None,
            population=None,
            MAX_NEURONS=64,
            MAX_LAYERS=10,
            NUM_PARENTS=2,
            **kwargs):
        """
        Initializes a FeedForwardIndividual.
        """
        super().__init__(parents_genes, population)
        # Feed Forward Neural Network parameters
        self.MAX_LAYERS = MAX_LAYERS
        self.MAX_NEURONS = MAX_NEURONS
        self.NUM_PARENTS = NUM_PARENTS
        # Keras parameters -> Maybe these need to be in the population, not the
        # individual?
        self.EPOCHS = kwargs.get("EPOCHS", 1000)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 2048)
        self.VERBOSE = kwargs.get("VERBOSE", 0)
        self.OPTIMIZER = kwargs.get("OPTIMIZER", tf.keras.optimizers.Adam)
        self.VALIDATION_SPLIT = kwargs.get("VALIDATION_SPLIT", None)
        self.OUTPUT_BIAS = kwargs.get("OUTPUT_BIAS", True)
        self.LOSS = kwargs.get("LOSS", 'categorical_crossentropy')
        self.METRICS = kwargs.get(
            "METRICS", FeedForwardConstants.default_metrics)
        self.LEARNING_RATE = kwargs.get("LEARNING_RATE", .0001)
        # Training parameters
        self.EARLY_STOPPING = kwargs.get("EARLY_STOPPING", True)
        self.CALLBACKS = kwargs.get(
            "CALLBACKS", FeedForwardConstants.default_early_stopping)
        # Class parameters
        self.dominant_gene = None
        self.genes = self._random_genes() if self.need_new_genes else self._crossover()

    def _random_genes(self):
        """
        Randomly generate genes for this individual.
        """
        genes = OrderedDict()
        for layer in range(self.MAX_LAYERS):
            genes[layer] = OrderedDict()
