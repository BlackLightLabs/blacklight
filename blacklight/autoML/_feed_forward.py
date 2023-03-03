from blacklight.autoML.individuals.feedforwardindividual import FeedForwardIndividual
from blacklight.dataLoaders.dataLoader import choose_data_loader
from blacklight.autoML.populations.population import Population
from collections import OrderedDict
import numpy as np


class FeedForward(Population):
    """
    A population of DNN topologies that can be evaluated. Their GOAL is the current dataset that this population wants to work on.
    """

    def __init__(self, number_of_individuals, num_parents_mating, death_percentage, number_of_generations,  **kwargs):
        super().__init__(number_of_individuals, num_parents_mating, death_percentage, number_of_generations, **kwargs)
        self.model = None
        self.data = None
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage

    def _initialize_individuals(self, **kwargs):
        self.individuals = OrderedDict({FeedForwardIndividual(
            None, self, **kwargs): f"{i}" for i in range(self.num_individuals)})

    def fit(self, data, **kwargs):
        self.data = choose_data_loader(data).get_dataset(kwargs.get("BATCH_SIZE", None)) if data else None
        # Initialize individuals
        self._initialize_individuals(**kwargs)
        self._simulate()
        self.model = list(self.individuals.keys())[0].model

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, data, **kwargs):
        self.fit(data, **kwargs)
        return self.predict(self.get_testing_data())

    def _simulate(self):
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
            {individual:
             individual.get_fitness() for individual in self.individuals.keys()})
        self.individuals = OrderedDict(sorted(
            evaluated_individuals.items(),
            key=lambda x: x[1],
            reverse=True))

    def _reproduce(self):
        self._kill_off_worst()
        for _ in range(self.num_individuals - len(self.individuals)):
            parents = np.random.choice(
                list(self.individuals.keys()), size=2, replace=False)
            child = parents[0].mate(parents[1])
            self.individuals["new_child"] = child

    def _get_fitness(self):
        pass
