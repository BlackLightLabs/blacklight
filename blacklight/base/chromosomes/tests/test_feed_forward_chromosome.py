import unittest
from blacklight.base.chromosomes.feed_forward_chromosome import FeedForwardChromosome, \
    handle_feed_forward_chromosome_cross_over
import random
import numpy as np
from tensorflow import keras


class TestFeedForwardChromosome(unittest.TestCase):

    def test_chromosome_initialize(self):
        random.seed(69)
        chrome = FeedForwardChromosome(
            input_shape=5, mutation_prob=0.1, model_params={'target_layer': (3, 'softmax')})
        self.assertEqual(chrome.length, 2)
        np.testing.assert_array_equal(
            chrome.genes, [(7, 'sigmoid'), (5, 'tanh')])

    def test_handle_chromosome_crossover(self):
        random.seed(69)
        chromeA = FeedForwardChromosome(
            input_shape=5, mutation_prob=0.1, model_params={'target_layer': (3, 'softmax')})
        chromeB = FeedForwardChromosome(
            input_shape=5, mutation_prob=0.1, model_params={'target_layer': (3, 'softmax')})

        new_chrome = handle_feed_forward_chromosome_cross_over(
            chromeA, chromeB)
        assert (isinstance(new_chrome, FeedForwardChromosome))

    def test_get_model(self):
        random.seed(69)
        chrome = FeedForwardChromosome(
            input_shape=5, mutation_prob=0.1, model_params={'target_layer': (3, 'softmax')})
        model = chrome.get_model()
        assert (model is not None)
        self.assertIsInstance(model, keras.models.Sequential)
        return
