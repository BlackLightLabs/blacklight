from blacklight.autoML.individuals.feedforwardindividual import FeedForwardIndividual
from blacklight.dataLoaders.dataLoader import choose_data_loader
from blacklight.autoML.populations.population import Population
from collections import OrderedDict
import numpy as np


class FeedForward(Population):
    """
    A population of Feed Forward DNN topologies that will be evaluated utilizing a genetic search.

    Parameters:
            number_of_individuals (int): The number of individuals that the user wants in the population.
            num_parents_mating (int): The number of parents that will be used to mate and create the next generation.
            death_percentage (float): The percentage of individuals that will die off each generation.
            number_of_generations (int): The number of generations that the population will be simulated for.
    """

    def __init__(
            self,
            number_of_individuals,
            num_parents_mating,
            death_percentage,
            number_of_generations,
            **kwargs):
        super().__init__(
            number_of_individuals,
            num_parents_mating,
            death_percentage,
            number_of_generations,
            **kwargs)
        self.model = None
        self.data = None
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage
        self.kwargs = kwargs

    def _initialize_individuals(self, **kwargs):
        self.individuals = OrderedDict({FeedForwardIndividual(
            None, self, **kwargs): f"{i}" for i in range(self.num_individuals)})

    def fit(self, data, **kwargs):
        """
        Fit this population of FeedForward Individuals to the given data, and return the best model.

        Parameters:
                data: The data to fit this population to. Has to be either a pandas dataframe with columns: [feature1, feature2, ..., featureN, label] or a file path to a file of formats specified in `ref: dataLoaders.dataLoader.choose_data_loader` that have the same layout.
                **kwargs: Any additional arguments to pass to each :class:`~blacklight.autoML.individuals.FeedForwardIndividual` which is a Keras model.
        """
        self.data = choose_data_loader(data).get_dataset(
            kwargs.get("BATCH_SIZE", None)) if data else None
        # Initialize individuals
        self._initialize_individuals(**kwargs)
        self._simulate()
        self.model = list(self.individuals.keys())[0].model

    def predict(self, X):
        """
        Predict the labels of the given data, utilizing the best found model.

        Parameters:
                X: The data to predict the labels of. Has to be either a pandas dataframe with columns: [feature1, feature2, ..., featureN] or a file path to a file of formats specified in `ref: dataLoaders.dataLoader.choose_data_loader` that have the same layout.
        """
        return self.model.predict(X)

    def fit_predict(self, data, **kwargs):
        """
        Fit this population of FeedForward Individuals to the given data, and return the predictions of the best model.

        Parameters:
                data: The data to fit this population to. Has to be either a pandas dataframe with columns: [feature1, feature2, ..., featureN, label] or a file path to a file of formats specified in `ref: dataLoaders.dataLoader.choose_data_loader` that have the same layout.
        """
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
