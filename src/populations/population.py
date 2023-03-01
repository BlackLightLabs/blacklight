from src.individuals.feedforwardindividual import FeedForwardIndividual
from src.dataLoaders.dataLoader import *
from collections import OrderedDict
import numpy as np


class Population:
    """
    A population of DNN topologies that can be evaluated. Their GOAL is the current dataset that this population wants to work on.
    """

    def __init__(self, number_of_individuals, num_parents_mating, death_percentage, number_of_generations, data,
                 data_type, **kwargs):
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage
        self.data = choose_data_loader(data, data_type) if data else None
        self.data_type = data_type
        # Initialize individuals
        self._initialize_individuals(**kwargs)

    def _initialize_individuals(self, **kwargs):
        self.individuals = OrderedDict(
            {" ": FeedForwardIndividual(None, self, **kwargs) for i in range(self.num_individuals)})

    def simulate(self):
        """
        Simulate the population for the number of generations.
        Callable method.
        :return:
        """
        for _ in range(self.num_generations):
            self._evaluate()
            self._reproduce()

    def _evaluate(self):
        """
        Evaluate the fitness of all individuals in the population.
        :return:
        """
        evaluated_individuals = OrderedDict(
            (individual.get_fitness, individual) for individual in self.individuals.values())
        self.individuals = sorted(evaluated_individuals.items(), key=lambda x: x[0], reverse=True)

    def _reproduce(self):
        self._kill_off_worst()
        for _ in range(self.num_individuals - len(self.individuals)):
            parents = np.random.choice(list(self.individuals.values()), replace=False)
            child = parents[0].mate(parents[1])
            self.individuals[" "] = child

    def _kill_off_worst(self):
        for _ in range(self.num_individuals - self.death_percentage * self.num_individuals):
            self.individuals.popitem()

    def get_X_train(self):
        return self.data.X if self.data else None

    def get_y_train(self):
        return self.data.y if self.data else None

    def get_X_test(self):
        return self.data.X_test if self.data else None

    def get_y_test(self):
        return self.data.y_test if self.data else None

    def _get_fitness(self):
        pass
