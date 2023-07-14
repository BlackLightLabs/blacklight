from blacklight.genetic.individuals import FeedForwardIndividual
from blacklight.engine.data import BlacklightDataset
from blacklight.genetic.base import Population
from blacklight.engine import ModelConfig
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging


class FeedForward(Population):
    """
    A population of Feed Forward DNN topologies that will be evaluated utilizing a genetic search.

    Parameters:
            number_of_individuals (int): The number of individuals that the user wants in the population.
            num_parents_mating (int): The number of parents that will be used to mate and create the next generation.
            death_percentage (float): The percentage of individuals that will die off each generation.
            number_of_generations (int): The number of generations that the population will be simulated for.
            options (ModelConfig): ModelConfig Object containing options for the model (e.g. number of layers, number of neurons, etc.).
    """

    def __init__(
        self,
        number_of_individuals,
        num_parents_mating,
        death_percentage,
        number_of_generations,
        options,
    ):
        super().__init__(
            number_of_individuals,
            num_parents_mating,
            death_percentage,
            number_of_generations,
            options,
        )
        self.num_classes = None
        self.test_data = None
        self.problem_type_individual = None
        self.model_history = None
        self.model = None
        self.data = None
        self.num_individuals = number_of_individuals
        self.num_parents_mating = num_parents_mating
        self.num_generations = number_of_generations
        self.death_percentage = death_percentage
        self.problem_type = None
        self.options = ModelConfig.parse_options_to_model_options(options)

    def _initialize_individuals(self):
        """
        Initialize the individuals in this population.
        Returns:

        """
        self.individuals = OrderedDict(
            {
                FeedForwardIndividual(self.options, self, None): f"{i}"
                for i in range(self.num_individuals)
            }
        )

    def fit(self, X_train, y_train=None, X_test=None, y_test=None, **kwargs):
        """
        Fit this population of FeedForward Individuals to the given data, and return the best model.
        You can pass ONLY x if and only if you have a column in your dataset (if pandas) called labels or label.
        You can also do this if you have a numpy array with the label column at position -1.
        Otherwise, you must pass X_train and y_train. If you pass X_test and y_test, then the model will be evaluated
        on the test data after each generation. If not, then test train split will be applied with an 80/20 split.

        Parameters:
                X: The data to fit this population to. Has to be either a pandas dataframe with columns: [feature1, feature2, ..., featureN, label] or a file path to a file of formats specified in `ref: dataLoaders.dataLoader.choose_data_loader` that have the same layout.
                **kwargs: Any additional arguments to pass to each :class:`~blacklight.autoML.individuals.FeedForwardIndividual` which is a Keras model.
                y: The labels of the data. If None, then the labels are assumed to be the last column of the data.
        """
        # Disable tensorflow logging
        import os

        logging.getLogger("tensorflow").disabled = True
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if X_test is not None and y_test is not None:
            self.test_data = BlacklightDataset(
                X_test, y_test, kwargs.get("batch_size", None)
            )
            self.train_data = BlacklightDataset(
                X_train, y_train, kwargs.get("batch_size", None)
            )
        else:
            data = BlacklightDataset(X_train, y_train, kwargs.get("batch_size", None))
            X_train, X_test, y_train, y_test = train_test_split(
                data.X, data.y, test_size=0.2
            )
            self.test_data = BlacklightDataset(
                X_test, y_test, kwargs.get("batch_size", None)
            )
            self.train_data = BlacklightDataset(
                X_train, y_train, kwargs.get("batch_size", None)
            )

        # Initialize individuals
        self._initialize_individuals()

        # Simulate the population for the specified number of generations
        self._simulate()

        # Get the best individual
        self.best_individual = list(self.individuals.keys())[0]
        self.model, self.model_history = (
            self.best_individual.model,
            self.best_individual.model_history,
        )

    def print_model_summary(self):
        """
        Print the summary of the best model.
        """
        self.model.summary()

    def print_model_training_history(self):
        plt.plot(self.model_history.history["accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.show()

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
        print(
            f"\nSimulating feed forward neural network population with {self.num_individuals} individuals for {self.num_generations} generations."
        )
        for gen in range(self.num_generations):
            print(f"\nGeneration {gen}")
            print(f"Evaluating Individuals in generation {gen}")
            self._evaluate()
            print(f"\nReproducing Individuals in generation {gen}")
            self._reproduce()

    def _evaluate(self):
        """
        Evaluate the fitness of all individuals in the population.
        :return:
        """
        evaluated_individuals = OrderedDict(
            {
                individual: individual.get_fitness()
                for individual in tqdm(self.individuals.keys())
            }
        )
        self.individuals = OrderedDict(
            sorted(evaluated_individuals.items(), key=lambda x: x[1], reverse=True)
        )

    def _get_fitness(self):
        pass
