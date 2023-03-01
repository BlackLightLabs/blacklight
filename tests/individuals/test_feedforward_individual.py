import unittest
import numpy as np
from src.individuals.feedforwardindividual import FeedForwardIndividual
from src.individuals.individualutils import get_crossover_points_from_num_parents
from src.populations.population import Population
import random
from collections import OrderedDict


class TestIndividual(unittest.TestCase):

    def test_individual(self):
        with self.assertRaises(ValueError):
            _ = FeedForwardIndividual(None, None)
        return

    def test_individual_random_genes(self):
        this_individual = FeedForwardIndividual(None, Population(2, 2, 0.2, 10, data=None, data_type=None))
        # Genes should not be None
        self.assertNotEqual(this_individual.genes, None)
        # Genes should be a dictionary, with both keys "gene_0" and "gene_1"
        gene_names = np.array(list(this_individual.genes.keys()))
        np.testing.assert_array_equal(gene_names, np.array(["gene_0", "gene_1"]))
        return

    def test_individual_random_genes_right_length(self):
        """
        Test that the random genes are the right length. An individual should only ever have 2 genes.
        :return:
        """
        this_individual = FeedForwardIndividual(None, Population(2, 2, 0.2, 10, data=None, data_type=None))
        self.assertEqual(len(this_individual.genes.keys()), 2)

        three_parent = FeedForwardIndividual(None, Population(2, 2, 0.2, 10, data=None, data_type=None), NUM_PARENTS=3)
        self.assertEqual(len(three_parent.genes.keys()), 2)

    def test_individual_crossover(self):
        """
        Test that the crossover function works.
        :return:
        """
        random.seed(69)
        parents_genes = [
            {"gene_0": OrderedDict({2: "relu", 3: "relu", 4: "relu"}),
             "gene_1": OrderedDict({5: "sigmoid", 6: "sigmoid", 7: "sigmoid"})},
            {"gene_0": OrderedDict({8: "relu", 9: "relu", 10: "relu"}),
             "gene_1": OrderedDict({11: "sigmoid", 12: "sigmoid", 13: "sigmoid"})}
        ]
        population = Population(2, 2, 0.2, 10, data=None, data_type=None)
        child = FeedForwardIndividual(parents_genes=parents_genes, population=population)
        # Genes should not be None
        self.assertNotEqual(child.genes, None)
        # Genes should be a dictionary, with both keys "gene_0" and "gene_1"
        gene_names = np.array(list(child.genes.keys()))
        np.testing.assert_array_equal(gene_names, np.array(["gene_0", "gene_1"]))
        # Recombinates
        crossover_points = get_crossover_points_from_num_parents(2, 3)
        print(crossover_points)
        child_genes = {"gene_0": OrderedDict({2: "relu", 9: "relu", 10: "relu"}),
                       "gene_1": OrderedDict({8: "relu", 3: "relu", 4: "relu"})
                       }

        for gene in child.genes.keys():
            # Test the values
            np.testing.assert_array_equal(list(child.genes[gene].values()), list(child_genes[gene].values()))
            # Test the keys
            np.testing.assert_array_equal((list(child.genes[gene].keys())), (list(child_genes[gene].keys())))
