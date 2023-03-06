import unittest
import numpy as np
from blacklight.base.individuals.feedforwardindividual import FeedForwardIndividual
from blacklight.base.population import Population
import random
from unittest import mock
from blacklight.dataLoaders.dataLoader import Dataset


class TestIndividual(unittest.TestCase):

    def setUp(self):
        X = np.array([[1, 5.1, 3.5, 1.4, 0.2], [
                     2, 4.9, 3.0, 1.4, 0.2], [3, 4.7, 3.2, 1.3, 0.2]])
        y = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        self.dataSet = Dataset(X, y, None)
        random.seed(69)

    def test_individual(self):
        with self.assertRaises(ValueError):
            _ = FeedForwardIndividual(None, None)
        return

    @mock.patch('blacklight.base.population.Population.get_training_data')
    def test_individual_random_genes(self, mock_get_training_data):
        NewPopulation = Population(2, 2, 0.2, 5)
        NewPopulation.data = self.dataSet
        mock_get_training_data.return_value = self.dataSet

        this_individual = FeedForwardIndividual(None, population=NewPopulation)
        self.assertIsNotNone(this_individual.chromosome)
        np.testing.assert_array_equal(
            this_individual.chromosome.genes, [
                (7, 'sigmoid'), (5, 'tanh')])
        return

    @mock.patch('blacklight.base.population.Population.get_training_data')
    def test_individual_inheritance(self, mock_get_training_data):
        NewPopulation = Population(2, 2, 0.2, 5)
        mock_get_training_data.return_value = self.dataSet

        first_individual = FeedForwardIndividual(
            None, population=NewPopulation)
        second_individual = FeedForwardIndividual(
            None, population=NewPopulation)

        child = first_individual.mate(second_individual)
        self.assertIsNotNone(child.chromosome)
