import unittest
from blacklight.autoML._feed_forward import FeedForward
from unittest import mock
import numpy as np
from blacklight.blacklightDataLoader import BlacklightDataset
import pandas as pd
import random


class TestFeedForward(unittest.TestCase):
    """
    Basic test to assure that CI/CD is functioning properly
    """

    def setUp(self):
        X = np.array([[1, 5.1, 3.5, 1.4, 0.2], [2, 4.9, 3.0, 1.4, 0.2], [3, 4.7, 3.2, 1.3, 0.2], [4, 4.6, 3.1, 1.5, 0.2], [3, 4.7, 3.2, 1.3, 0.2], [3, 4.7, 3.2, 1.3, 0.2], [3, 4.7, 3.2, 1.3, 0.2]])
        y = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa'])
        self.X, self.y = X, y
        self.dataSet = BlacklightDataset(X, y, None)
        self.dataDF = pd.DataFrame(
            data=X,
            columns=[
                'ID',
                'SepalLengthCm',
                'SepalWidthCm',
                'PetalLengthCm',
                'PetalWidthCm'])
        self.dataDF['label'] = y

    def test_correct_loading(self):
        individual = FeedForward(2, 2, 0.2, 10, VERBOSE=2)
        self.assertEqual(individual.num_individuals, 2)
        self.assertEqual(individual.num_parents_mating, 2)
        self.assertEqual(individual.death_percentage, 0.2)
        self.assertEqual(individual.num_generations, 10)

    @mock.patch('blacklight.base.individuals.feedForwardIndividuals.ClassifierFeedForwardIndividual.get_fitness')
    def test_iris_data_is_loaded(
            self,
            mock_get_fitness
    ):

        mock_get_fitness.return_value = 0.9
        pop = FeedForward(4, 4, 0.2, 10)
        pop.fit(self.dataDF)
        self.assertIsNotNone(pop.individuals)
        self.assertIsNotNone(pop.data)

    @mock.patch('blacklight.base.individuals.feedForwardIndividuals.ClassifierFeedForwardIndividual.get_fitness')
    def test_kill_off_individual_works_as_expected(self, mock_get_fitness):
        mock_get_fitness.return_value = random.uniform(0, 1)
        pop = FeedForward(4, 4, 0.2, 3)
        pop.fit(self.dataDF)
        self.assertGreater(len(pop.individuals), 0)