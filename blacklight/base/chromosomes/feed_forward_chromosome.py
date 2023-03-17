from blacklight.base.chromosome import BaseChromosome
import tensorflow as tf
from tensorflow import keras
import random


def handle_feed_forward_chromosome_cross_over(chromosomeA, chromosomeB):
    """
    Handle feed forward cross over.
    """
    if chromosomeA.length > chromosomeB.length:
        longer_chromosome = chromosomeA
        shorter_chromosome = chromosomeB
    else:
        longer_chromosome = chromosomeB
        shorter_chromosome = chromosomeA

    points = random.randint(1, len(shorter_chromosome.genes))
    base_one = shorter_chromosome.genes[:points]
    link_one = longer_chromosome.genes[points:]

    base_two = longer_chromosome.genes[:points]
    link_two = shorter_chromosome.genes[points:]

    recombinant_one = base_one + link_one
    recombinant_two = base_two + link_two

    genes = random.choice([recombinant_one, recombinant_two])

    new_chromosome = FeedForwardChromosome(
        input_shape=chromosomeA.input_shape,
        mutation_prob=chromosomeA.mutation_prob,
        genes=genes,
        model_params=chromosomeA.model_params)

    return new_chromosome


class FeedForwardChromosome(BaseChromosome):
    """
    Feed forward chromosome class.

    Parameters:
            parents_genes: list of genes from parents

    """

    def __init__(self,
                 input_shape=None,
                 max_neurons=64,
                 max_layers=10,
                 min_neurons=1,
                 genes=None,
                 mutation_prob=None,
                 model_params=None):
        super().__init__()
        self.model_params = model_params if model_params else {}
        has_new_genes = genes is not None

        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.input_shape = input_shape

        self.length = None
        self.mutation_prob = mutation_prob

        # Keras model parameters
        self.OPTIMIZER = self.model_params.get(
            "optimizer", tf.keras.optimizers.Adam)

        self.LOSS = self.model_params.get("loss", 'categorical_crossentropy')
        self.METRICS = self.model_params.get(
            "metrics")
        self.LEARNING_RATE = self.model_params.get("learning_rate", .0001)

        self.genes = genes if genes else self._random_genes()
        self.length = len(self.genes)
        if has_new_genes:
            self._mutate()

        self.model = self._make_model()

    def _random_genes(self):
        """
        Generate random feed forward genes for the chromosome.
        """
        layer_type_activation_types = ['relu', 'sigmoid', 'tanh', 'selu']
        layer_size = range(self.min_neurons, self.max_neurons)

        genes = [
            (
                random.choice(layer_size),
                layer_type_activation_types[random.randint(0, len(layer_type_activation_types) - 1)]

            ) for _ in range(random.choice(range(2, self.max_layers)))
        ]
        return genes

    def _make_model(self):
        feed_forward_model = keras.Sequential()
        # Add input layer
        feed_forward_model.add(
            tf.keras.layers.Dense(self.genes[0][0], activation=self.genes[0][1], input_shape=(self.input_shape,)))
        # Add hidden layers
        for allele in self.genes[1:]:
            feed_forward_model.add(tf.keras.layers.Dense(allele[0], activation=allele[1]))
        # Add output layer
        target_layer = self.model_params.get("target_layer")
        feed_forward_model.add(tf.keras.layers.Dense(target_layer[0], activation=target_layer[1]))

        feed_forward_model.compile(
            optimizer=self.OPTIMIZER(learning_rate=self.LEARNING_RATE),
            loss=self.LOSS,
            metrics=self.METRICS
        )
        return feed_forward_model

    def _mutate(self):
        """
        Mutate the chromosome.
        """
        mutate = random.choices([True, False], weights=[
            self.mutation_prob, 10 - self.mutation_prob], k=1)[0]

        if mutate:
            layer_idx = random.choice(range(self.length))
            new_allele = random.randint(1, self.max_neurons)
            new_activation = random.choice(['relu', 'sigmoid', 'tanh', 'selu'])
            self.genes[layer_idx] = (new_allele, new_activation)

    def get_model(self):
        return self.model

    def __repr__(self):
        return f"FeedForwardChromosome with genes: {self.genes}"

    def __str__(self):
        return f"FeedForwardChromosome with genes: {self.genes} \n and model: {self.model.summary()}"
