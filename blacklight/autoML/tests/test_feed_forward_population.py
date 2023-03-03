import unittest
from unittest import mock
from blacklight.autoML._feed_forward import FeedForward
from blacklight.autoML.individuals.feedforwardindividual import FeedForwardIndividual
from collections import OrderedDict


class TestFeedForward(unittest.TestCase):
    """
    Basic test to assure that CI/CD is functioning properly
    """

    @classmethod
    def setUpClass(cls):
        cls.data_loc = "dataLoaders/tests/data/iris.csv"

    @classmethod
    def tearDownClass(cls):
        cls.data_loc = None

    def test_correct_loading(self):
        individual = FeedForward(2, 2, 0.2, 10, VERBOSE=2)
        self.assertEqual(individual.num_individuals, 2)
        self.assertEqual(individual.num_parents_mating, 2)
        self.assertEqual(individual.death_percentage, 0.2)
        self.assertEqual(individual.num_generations, 10)

    @mock.patch.object(FeedForwardIndividual, '_evaluate_model', return_value=0.5)
    def test_iris_data_is_loaded(self):
        pop = FeedForward(4, 4, 0.2, 10)
        pop._evaluate = mock.Mock().method.return_value = OrderedDict({FeedForwardIndividual(None, pop): 0.5, FeedForwardIndividual(None, pop): 0.6})
        pop.fit(self.data_loc)
        self.assertIsNotNone(pop.individuals)
        self.assertIsNotNone(pop.data)

