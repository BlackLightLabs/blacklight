from collections import OrderedDict
import numpy as np


class Population:
    """
    A population of DNN topologies that can be evaluated. Their GOAL is the current dataset that this population wants to work on.
    """

    def __init__(
            self,
            number_of_individuals,
            num_parents_mating,
            death_percentage,
            number_of_generations,
            **kwargs):
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage
        self.individuals = None
        self.data = None

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

    def _reproduce(self):
        self._kill_off_worst()
        for _ in range(self.num_individuals - len(self.individuals)):
            parents = np.random.choice(
                list(self.individuals.values()), replace=False)
            child = parents[0].mate(parents[1])
            self.individuals[" "] = child

    def _kill_off_worst(self):
        for _ in range(
                int((self.death_percentage * self.num_individuals))):
            self.individuals.popitem()

    def get_training_data(self):
        return self.data if self.data else None

    def get_testing_data(self):
        return self.data if self.data else None

    def _get_fitness(self):
        pass
