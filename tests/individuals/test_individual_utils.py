import unittest
from src.individuals.individualutils import *


class TestMinLengthChromosome(unittest.TestCase):
    # Test get_min_length_chromosome
    def test_min_length_chromosome(self):
        chromosomes = {
            "chromosome1": {
                1: 1, 2: 2, 3: 3}, "chromosome2": {
                1: 1, 2: 2, 3: 3, 4: 4}}
        self.assertEqual(get_min_length_chromosome(chromosomes), "chromosome1")
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
        self.assertEqual(len(crossover_points), num_parents)
        self.assertTrue(all(point < chromosome_length -
                        1 for point in crossover_points))
        return
