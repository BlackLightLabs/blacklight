"""
Utils for individuals.
"""

import random


def get_min_length_chromosome(chromosomes):
    """
    Get the chromosome with the minimum length.
    """
    if len(chromosomes) == 0:
        raise ValueError("Chromosomes can not be empty.")
    return min(chromosomes, key=lambda chromosome: len(
        chromosomes[chromosome].keys()))


def get_crossover_points_from_num_parents(num_parents, chromosome_length):
    """
    Get a list of crossover points for a given number of parents and chromosome length.
    """
    num_crossover_points = 0
    crossover_points = []
    index_chromosome_list = 0

    while num_crossover_points <= num_parents:
        index = random.randint(index_chromosome_list, chromosome_length - 2)
        crossover_points.append(index)
        index_chromosome_list = index + 1
        num_crossover_points += 1

    return crossover_points
