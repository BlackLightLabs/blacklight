import unittest
from src.individuals.individualutils import *
import numpy as np

class TestMinLengthChromosome(unittest.TestCase):
    # Test get_min_length_chromosome
    def test_min_length_chromosome(self):
        parents_chromosomes = [
            {
                # gene_1 should be the min len chromosome, returning a value of 3.
                "gene_1": {1: "relu", 2: "relu", 3: "relu"},
                "gene_2": {1: "sigmoid", 2: "sigmoid", 3: "sigmoid", 4: "sigmoid"}
            },
            {
                "gene_1": {1: "relu", 2: "relu", 3: "relu", 4: "relu"},
                "gene_2": {1: "sigmoid", 2: "sigmoid", 3: "sigmoid", 4: "sigmoid", 5: "sigmoid"}
            }
        ]
        self.assertEqual(get_min_length_chromosome(parents_chromosomes), 3)
        return

    # Test get_min_length_chromosome with empty chromosomes
    def test_min_length_chromosome_empty(self):
        chromosomes = {}
        with self.assertRaises(ValueError):
            get_min_length_chromosome(chromosomes)
        return


class TestCrossoverPoints(unittest.TestCase):
    # Test get_crossover_points_from_num_parents
    def test_crossover_points(self):
        num_parents = 3
        chromosome_length = 10
        crossover_points = get_crossover_points_from_num_parents(
            num_parents, chromosome_length)
        self.assertEqual(len(crossover_points), num_parents - 1)
        self.assertTrue(all(point <= chromosome_length -
                            1 for point in crossover_points))
        return


class TestMergeGenes(unittest.TestCase):

    def test_merge_genes(self):
        parents_genes = [
            {"gene_0": OrderedDict({2: "relu", 3: "relu", 4: "relu"}),
             "gene_1": OrderedDict({5: "sigmoid", 6: "sigmoid", 7: "sigmoid"})},
            {"gene_0": OrderedDict({8: "relu", 9: "relu", 10: "relu"}),
             "gene_1": OrderedDict({11: "sigmoid", 12: "sigmoid", 13: "sigmoid"})}
        ]

        crossover_point = 1
        correct_child_genes = {
            "gene_0_recombinant_one": OrderedDict({2: "relu", 9: "relu", 10: "relu"}),
            "gene_0_recombinant_two": OrderedDict({8: "relu", 3: "relu", 4: "relu"}),
            "gene_1_recombinant_one": OrderedDict({5: "sigmoid", 12: "sigmoid", 13: "sigmoid"}),
            "gene_1_recombinant_two": OrderedDict({11: "sigmoid", 6: "sigmoid", 7: "sigmoid"})
        }
        child_genes = merge_genes(parents_genes[0], parents_genes[1], crossover_point)

        for recombinant in correct_child_genes.keys():
            np.testing.assert_array_equal(child_genes[recombinant].keys(), correct_child_genes[recombinant].keys())

class TestMutateGene(unittest.TestCase):

    def test_mutate(self):
        random.seed(889)
        gene = OrderedDict({1: "relu", 2: "relu", 3: "relu", 4: "relu", 5: "relu"})
        mutated = False
        dom_gene = None
        while not mutated:
            dom_gene, mutated = mutate_dominant_gene(gene, 30)
        self.assertEqual(len(gene), 5)
        self.assertEqual(5 in list(dom_gene.keys()), True)