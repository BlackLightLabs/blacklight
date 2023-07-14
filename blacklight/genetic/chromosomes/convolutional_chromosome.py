from __future__ import annotations
import random
from abc import ABC
import tensorflow as tf
from blacklight.genetic.base.chromosome import Chromosome
from blacklight.engine import BlacklightModel


class ConvolutionalChromosome(Chromosome, ABC):
    """
    Convolutional chromosome class, which represents a convolutional neural network architecture.

    The class inherits from BaseChromosome and implements methods to generate random genes, mutate
    the chromosome, and create a Keras model based on the genes. The genes represent both the
    convolutional and dense layers of the neural network.

    Parameters:
        input_shape (tuple): The input shape of the model (width, height, channels).
        max_conv_layers (int): The maximum number of convolutional layers in the model.
        max_dense_layers (int): The maximum number of dense layers in the model.
        genes (list): List of genes from parents (default: None).
        mutation_prob (float): Probability of mutation (default: None).
        model_params (dict): Dictionary of model parameters such as optimizer, loss, metrics, and learning rate.

    Attributes:
        model (tf.keras.Model): The Keras model represented by the chromosome.
        genes (list): The list of genes that describe the architecture of the neural network.
        length (int): The number of genes in the chromosome.

    Raises:
        ValueError: If the input_shape is not a tuple or if the mutation probability is not between 0 and 1.

    Methods:
        _random_genes: Generate random genes for the chromosome.
        _make_model: Create a Keras model based on the chromosome's genes.
        _mutate: Mutate the chromosome's genes.
        get_model: Return the Keras model.

    Examples:
        >>> input_shape = (32, 32, 3)
        >>> max_conv_layers = 5
        >>> max_dense_layers = 3
        >>> model_params = {
            "optimizer": tf.keras.optimizers.Adam,
            "loss": 'categorical_crossentropy',
            "metrics": ['accuracy'],
            "learning_rate": 0.001
        }
        >>> chromosome = ConvolutionalChromosome(input_shape, max_conv_layers, max_dense_layers, model_params=model_params)
        >>> model = chromosome.get_model()
    """

    def __init__(
            self,
            input_shape=None,
            max_conv_layers=10,
            max_dense_layers=5,
            min_conv_layers=1,
            min_dense_layers=1,
            genes=None,
            mutation_prob=None,
            model_params=None):
        super().__init__()
        self.model_params = model_params if model_params else {}
        has_new_genes = genes is not None

        self.length = None
        self.mutation_prob = mutation_prob

        self.genes = genes if genes else self._random_genes()
        self.length = len(self.genes)
        if has_new_genes:
            self._mutate()

        model = BlacklightModel(self.model_params, self.genes)

        self.model = model.create_model()

    @staticmethod
    def split_convolutional_dense(genes: list) -> tuple[list, list]:
        """
        Split convolutional and dense layers from the genes.

        Parameters:
            genes: list of alleles for a convolutional chromosome

        Returns:
            - conv_part: the convolutional part of the genes
            - dense_part: the dense part of the genes
        """
        try:
            flatten_index = genes.index(("Flatten",))
        except ValueError:
            raise ValueError("Flatten layer not found in the genes.")

        conv_part = genes[:flatten_index]
        dense_part = genes[flatten_index + 1:]
        return conv_part, dense_part

    @staticmethod
    def cross_over(
            chromosome_a: ConvolutionalChromosome,
            chromosome_b: ConvolutionalChromosome) -> ConvolutionalChromosome:
        """
            Perform a crossover operation between two ConvolutionalChromosome instances.

            This function creates two new offspring by crossing over the genes of the input chromosomes.
            The crossover operation is applied separately to the convolutional part and the dense part
            of the chromosomes, respecting their structure.

            Args:
                chromosome_a (ConvolutionalChromosome): The first parent chromosome.
                chromosome_b (ConvolutionalChromosome): The second parent chromosome.

            Returns:
                ConvolutionalChromosome: A new ConvolutionalChromosome instances created through crossover.

            Raises:
                ValueError: If the "flatten" layer is not found in the genes or if the input chromosomes have different input shapes.
            """
        shorter_chromosome, longer_chromosome = Chromosome.get_shortest_chromosome(
            chromosome_a, chromosome_b)

        shorter_conv_part, shorter_dense_part = ConvolutionalChromosome.split_convolutional_dense(
            shorter_chromosome.genes)
        longer_conv_part, longer_dense_part = ConvolutionalChromosome.split_convolutional_dense(
            longer_chromosome.genes)

        crossover_point_conv = random.randint(1, len(shorter_conv_part))
        crossover_point_dense = random.randint(1, len(shorter_dense_part))

        recombinant_conv_part = shorter_conv_part[:crossover_point_conv] + \
            longer_conv_part[crossover_point_conv:]
        recombinant_dense_part = shorter_dense_part[:crossover_point_dense] + \
            longer_dense_part[crossover_point_dense:]

        recombinant_genes = recombinant_conv_part + \
            [("Flatten",)] + recombinant_dense_part

        new_chromosome = ConvolutionalChromosome(
            input_shape=chromosome_a.input_shape,
            mutation_prob=chromosome_a.mutation_prob,
            genes=recombinant_genes,
            model_params=chromosome_a.model_params)

        return new_chromosome

    def _mutate(self) -> None:
        """
        Mutate the chromosome's genes while preserving the structure.

        This method applies random mutations to the genes of the chromosome while ensuring that
        convolutional layers remain in the convolutional section and dense layers remain in the
        dense section. It selects a random layer for mutation and changes its parameters based on
        the layer type.

        Raises:
            ValueError: If an unexpected layer type is encountered in the genes.
        """
        mutate = random.choices([True, False], weights=[
            self.mutation_prob, 10 - self.mutation_prob], k=1)[0]

        if mutate:
            conv_indices = [
                i for i, gene in enumerate(
                    self.genes) if gene[0] == "Conv2D"]
            dense_indices = [
                i for i, gene in enumerate(
                    self.genes) if gene[0] == "Dense"]

            mutation_type = random.choice(["conv", "dense"])
            if mutation_type == "conv":
                layer_idx = random.choice(conv_indices)
                new_filters = random.randint(32, 256)
                self.genes[layer_idx] = (
                    "Conv2D",
                    new_filters,
                    self.genes[layer_idx][2],
                    self.genes[layer_idx][3])
            elif mutation_type == "dense":
                layer_idx = random.choice(dense_indices)
                new_neurons = random.randint(32, 256)
                self.genes[layer_idx] = (
                    "Dense", new_neurons, self.genes[layer_idx][2])

    def get_model(self) -> tf.keras.Model:
        return self.model

    def __repr__(self):
        return f"ConvolutionalChromosome with genes: {self.genes}"

    def __str__(self):
        return f"ConvolutionalChromosome with genes: {self.genes} \n and model: {self.model.summary()}"
