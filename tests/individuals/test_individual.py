import unittest
from src.individuals.feedforwardindividual import FeedForwardIndividual

class TestIndividual(unittest.TestCase):

    def test_individual(self):
        with self.assertRaises(ValueError):
            this_individual = FeedForwardIndividual(None, None)
        return

    def test_individual_random_genes(self): 
        this_individual = FeedForwardIndividual(None, "a random population")
        self.assertNotEqual(this_individual.genes, None)
        return

    def test_individual_random_genes_right_length(self):
        this_individual = FeedForwardIndividual(None, "a random population")
        self.assertEqual(len(this_individual.genes.keys()), 2)
        three_parent = FeedForwardIndividual(None, "a random population", NUM_PARENTS=3)
        self.assertEqual(len(three_parent.genes.keys()), 3)

        