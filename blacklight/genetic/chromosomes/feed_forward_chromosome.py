from __future__ import annotations
from blacklight.genetic.base import Chromosome
from blacklight.engine import BlacklightModel
from blacklight.engine import ModelConfig
from typing import List, Optional, Tuple
from keras import Sequential

import random


class FeedForwardChromosome(Chromosome):
    """
    Feed forward chromosome class representing a feed forward neural network.

    Args:
        model_params ModelConfig: The parameters for the Keras model. Defaults to None.
        genes (Optional[List[Tuple[int, str]]], optional): The list of genes representing the neural network structure. Defaults to None.
        mutation_prob (Optional[int], optional): The probability of mutation for the chromosome. Defaults to None.

    Examples:
        >>> chromosome = FeedForwardChromosome(input_shape=10)
    """

    def __init__(self,
                 model_params: ModelConfig,
                 genes: Optional[List[Tuple[int, str]]] = None,
                 mutation_prob: Optional[float] = None,
                 ):
        super().__init__()

        # Get the model params to build chromosome
        self.model_params = model_params

        # Check if the chromosome has new genes from parents
        has_new_genes = genes is not None

        # Get the mutation probability
        self.mutation_prob = mutation_prob

        # Get random genes if no genes are provided from parents.
        self.genes = genes if genes else self._random_genes()

        # Set the length of the chromosome
        self.length = len(self.genes)

        # If we got genes from parents, we need to mutate them.
        if has_new_genes:
            self._mutate()

        # Create the Keras model based on the genes, and the model params.
        self.model = BlacklightModel(
            self.model_params, self.genes)
        self.model.create_model()

    def _random_genes(self) -> List[Tuple[str, int, str]]:
        """
        Generate random feed forward genes for the chromosome.

        Returns:
            List[Tuple[int, str]]: A list of genes representing the neural network structure.
        """
        layer_size = range(
            self.model_params.get("min_dense_neurons"),
            self.model_params.get("max_dense_neurons"))

        genes = [
            ("Dense",
             random.choice(layer_size),
             random.choice(
                 self.model_params.get("dense_activation_types"))) for _ in range(
                random.choice(
                    range(
                        self.model_params.get("min_dense_layers"),
                        self.model_params.get("max_dense_layers"))))]
        return genes

    @staticmethod
    def cross_over(chromosome_a: FeedForwardChromosome,
                   chromosome_b: FeedForwardChromosome) -> FeedForwardChromosome:
        """
        Handle feed forward cross over between two chromosomes.

        Args:
            chromosome_a (FeedForwardChromosome): The first chromosome to perform cross over.
            chromosome_b (FeedForwardChromosome): The second chromosome to perform cross over.

        Returns:
            FeedForwardChromosome: A new chromosome created by cross over between the input chromosomes.

        Examples:
            >>> chromosome_a = FeedForwardChromosome(input_shape=10)
            >>> chromosome_b = FeedForwardChromosome(input_shape=10)
            >>> new_chromosome = FeedForwardChromosome.cross_over(chromosome_a,chromosome_b)
        """
        shorter_chromosome, longer_chromosome = Chromosome.get_shortest_chromosome(
            chromosome_a, chromosome_b)

        points = random.randint(1, len(shorter_chromosome.genes))
        base_one = shorter_chromosome.genes[:points]
        link_one = longer_chromosome.genes[points:]

        base_two = longer_chromosome.genes[:points]
        link_two = shorter_chromosome.genes[points:]

        recombinant_one = base_one + link_one
        recombinant_two = base_two + link_two

        genes = random.choice([recombinant_one, recombinant_two])

        new_chromosome = FeedForwardChromosome(
            model_params=chromosome_a.model_params,
            genes=genes,
            mutation_prob=chromosome_a.mutation_prob,
        )

        return new_chromosome

    def _mutate(self) -> None:
        """
        Mutate the chromosome by randomly changing one of its genes.
        """
        mutate = random.choices([True, False], weights=[
            self.mutation_prob, 1 - self.mutation_prob], k=1)[0]

        if mutate:
            layer_idx = random.choice(range(self.length))
            new_allele = random.randint(
                1, self.model_params.get("max_dense_neurons"))
            new_activation = random.choice(
                self.model_params.get("dense_activation_types"))
            self.genes[layer_idx] = ("Dense", new_allele, new_activation)

    def evaluate_model(self, train_data, test_data):
        """
        Evaluate the model and return the loss and accuracy.

        Returns:
            Tuple[float, float]: The loss and accuracy of the model.
        """
        fitness = self.model.evaluate_model(train_data, test_data)
        return fitness

    def get_model(self) -> Sequential:
        """
        Get the Keras Sequential model of the chromosome.

        Returns:
            tf.keras.Sequential: The feed forward neural network model.

        Examples:
            >>> chromosome = FeedForwardChromosome(input_shape=10)
            >>> model = chromosome.get_model()
        """
        return self.model.get_model()

    def get_model_history(self):
        return self.model.get_model_history()

    def __repr__(self):
        return f"FeedForwardChromosome with genes: {self.genes}"

    def __str__(self):
        return f"FeedForwardChromosome with genes: {self.genes} \n and model: {self.model.summary()}"
