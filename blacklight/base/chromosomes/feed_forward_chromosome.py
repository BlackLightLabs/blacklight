from blacklight.base.chromosome import BaseChromosome
from typing import List, Optional, Union, Tuple
import tensorflow as tf
from tensorflow import keras
import random




class FeedForwardChromosome(BaseChromosome):
    """
    Feed forward chromosome class representing a feed forward neural network.

    Args:
        input_shape (Optional[int], optional): The input shape of the neural network. Defaults to None.
        max_neurons (int, optional): The maximum number of neurons in a layer. Defaults to 64.
        max_layers (int, optional): The maximum number of layers in the neural network. Defaults to 10.
        min_neurons (int, optional): The minimum number of neurons in a layer. Defaults to 1.
        genes (Optional[List[Tuple[int, str]]], optional): The list of genes representing the neural network structure. Defaults to None.
        mutation_prob (Optional[int], optional): The probability of mutation for the chromosome. Defaults to None.
        model_params (Optional[dict], optional): The parameters for the Keras model. Defaults to None.

    Examples:
        >>> chromosome = FeedForwardChromosome(input_shape=10)
    """

    def __init__(self,
                 input_shape: Optional[int] = None,
                 max_neurons: int = 64,
                 max_layers: int = 10,
                 min_neurons: int = 1,
                 genes: Optional[List[Tuple[int, str]]] = None,
                 mutation_prob: Optional[int] = None,
                 model_params: Optional[dict] = None):
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

    def _random_genes(self) -> List[Tuple[int, str]]:
        """
        Generate random feed forward genes for the chromosome.

        Returns:
            List[Tuple[int, str]]: A list of genes representing the neural network structure.
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

    @staticmethod
    def cross_over(chromosome_a, chromosome_b):
        """
        Handle feed forward cross over between two chromosomes.

        Args:
            chromosome_a (FeedForwardChromosome): The first chromosome to perform cross over.
            chromosome_b (FeedForwardChromosome): The second chromosome to perform cross over.

        Returns:
            FeedForwardChromosome: A new chromosome created by cross over between the input chromosomes.

        Examples:
            >>> chromosomeA = FeedForwardChromosome(input_shape=10)
            >>> chromosomeB = FeedForwardChromosome(input_shape=10)
            >>> new_chromosome = handle_feed_forward_chromosome_cross_over(chromosome_a, chromosome_b)
        """
        shorter_chromosome, longer_chromosome = BaseChromosome.get_shortest_chromosome(chromosome_a, chromosome_b)

        points = random.randint(1, len(shorter_chromosome.genes))
        base_one = shorter_chromosome.genes[:points]
        link_one = longer_chromosome.genes[points:]

        base_two = longer_chromosome.genes[:points]
        link_two = shorter_chromosome.genes[points:]

        recombinant_one = base_one + link_one
        recombinant_two = base_two + link_two

        genes = random.choice([recombinant_one, recombinant_two])

        new_chromosome = FeedForwardChromosome(
            input_shape=chromosome_a.input_shape,
            mutation_prob=chromosome_a.mutation_prob,
            genes=genes,
            model_params=chromosome_a.model_params)

        return new_chromosome

    def _make_model(self) -> tf.keras.Sequential:
        """
        Create a Keras Sequential model based on the chromosome's genes.

        Returns:
            tf.keras.Sequential: A feed forward neural network model.
        """
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

    def _mutate(self) -> None:
        """
        Mutate the chromosome by randomly changing one of its genes.
        """
        mutate = random.choices([True, False], weights=[
            self.mutation_prob, 10 - self.mutation_prob], k=1)[0]

        if mutate:
            layer_idx = random.choice(range(self.length))
            new_allele = random.randint(1, self.max_neurons)
            new_activation = random.choice(['relu', 'sigmoid', 'tanh', 'selu'])
            self.genes[layer_idx] = (new_allele, new_activation)

    def get_model(self) -> tf.keras.Sequential:
        """
        Get the Keras Sequential model of the chromosome.

        Returns:
            tf.keras.Sequential: The feed forward neural network model.

        Examples:
            >>> chromosome = FeedForwardChromosome(input_shape=10)
            >>> model = chromosome.get_model()
        """
        return self.model

    def __repr__(self):
        return f"FeedForwardChromosome with genes: {self.genes}"

    def __str__(self):
        return f"FeedForwardChromosome with genes: {self.genes} \n and model: {self.model.summary()}"
