from collections import OrderedDict
import numpy as np


class Population:
    """
    An abstract class representing a population of Neural Network topologies that can be evaluated. Their GOAL is the current dataset that this population wants to work on.
    """

    def __init__(
            self,
            number_of_individuals: int,
            num_parents_mating: int,
            death_percentage: float,
            number_of_generations: int,
            options):
        self.test_data = None
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage
        self.problem_type = None
        self.num_classes = 3
        self.individuals = None
        self.train_data = None
        self.test_data = None

    def simulate(self):
        """
        Simulate the population for the number of generations.
        Callable method.
        :return:
        """
        pass

    def _evaluate(self):
        """
        Evaluate the fitness of all individuals in the population.
        :return:
        """
        evaluated_individuals = OrderedDict(
            (individual.get_fitness,
             individual) for individual in self.individuals.values())
        self.individuals = OrderedDict(sorted(
            evaluated_individuals.items(),
            key=lambda x: x[0],
            reverse=True))

    def _kill_off_worst(self):
        for _ in range(
                int((self.death_percentage * self.num_individuals))):
            self.individuals.popitem()
            self.num_individuals -= 1

    def _reproduce(self):
        self._kill_off_worst()
        for i in range(self.num_parents_mating):
            parents = np.random.choice(
                list(self.individuals.keys()), size=2, replace=False)
            child = parents[0].mate(parents[1])
            # Add individual from Bayesian Optimization.
            self.individuals[child] = f"new_child_{i}"
            self.num_individuals += 1

    def get_training_data(self):
        return self.train_data if self.train_data else None

    def get_testing_data(self):
        return self.test_data if self.test_data else None

    def _get_fitness(self):
        pass
