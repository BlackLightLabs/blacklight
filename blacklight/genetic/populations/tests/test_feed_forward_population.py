from blacklight.genetic.populations import FeedForward
from blacklight.genetic.individuals import FeedForwardIndividual
from blacklight.engine import ModelConfig

import numpy as np
import pandas as pd

import unittest.mock
import pytest
from unittest.mock import MagicMock
from collections import OrderedDict
import random


@pytest.fixture
def getdata():
    X = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [
        4.6, 3.1, 1.5, 0.2], [4.7, 3.2, 1.3, 0.2], [4.7, 3.2, 1.3, 0.2], [4.7, 3.2, 1.3, 0.2]])
    y = np.array(['Iris-setosa',
                  'Iris-versicolor',
                  'Iris-virginica',
                  'Iris-setosa',
                  'Iris-versicolor',
                  'Iris-virginica',
                  'Iris-setosa'])
    dataDF = pd.DataFrame(
        data=X,
        columns=[
            'SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm'])
    dataDF['label'] = y

    return X, y


@pytest.fixture
def get_test_data():
    X = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [
        4.6, 3.1, 1.5, 0.2], [4.7, 3.2, 1.3, 0.2], [4.7, 3.2, 1.3, 0.2]])
    y = np.array(['Iris-setosa',
                  'Iris-versicolor',
                  'Iris-virginica',
                  'Iris-setosa',
                  'Iris-setosa'])
    dataDF = pd.DataFrame(
        data=X,
        columns=[
            'SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm'])
    dataDF['label'] = y

    return X, y


@pytest.fixture
def model_config():
    model_options = {
        "layer_information": {
            "problem_type": "classification",
            "input_shape": 4,
            "min_dense_layers": 1,
            "max_dense_layers": 8,
            "min_dense_neurons": 2,
            "max_dense_neurons": 8,
            "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
        },
        "problem_type": "classification",
        "num_classes": 3,
    }
    model_config = ModelConfig.parse_options_to_model_options(model_options)
    return model_config


@pytest.fixture
def model_options():
    model_options = {
        "layer_information": {
            "problem_type": "classification",
            "input_shape": 4,
            "min_dense_layers": 1,
            "max_dense_layers": 8,
            "min_dense_neurons": 2,
            "max_dense_neurons": 8,
            "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
        },
        "problem_type": "classification",
        "num_classes": 3,
    }
    return model_options


def test_initialization(model_options):
    population = FeedForward(10, 5, 0.5, 5, model_options)
    assert isinstance(population, FeedForward)
    assert isinstance(population.options, ModelConfig)


def test_evaluate(model_config, model_options):
    population = FeedForward(3, 2, 0.5, 3, model_options)
    population.individuals = OrderedDict()

    for i in range(3):
        individual = FeedForwardIndividual(model_config, population)
        individual.get_fitness = MagicMock(return_value=i + 1)
        population.individuals[individual] = f"{i}"

    population._evaluate()

    sorted_individuals = list(population.individuals.keys())
    assert len(sorted_individuals) == 3
    assert sorted_individuals[0].get_fitness() == 3
    assert sorted_individuals[1].get_fitness() == 2
    assert sorted_individuals[2].get_fitness() == 1


def test_kill_off_worst(model_config, model_options):
    population = FeedForward(4, 2, 0.5, 3, model_options)
    population.individuals = OrderedDict()

    for i in range(4):
        individual = FeedForwardIndividual(model_config, population)
        individual.get_fitness = MagicMock(return_value=i + 1)
        population.individuals[individual] = f"{i}"

    population._evaluate()

    population._kill_off_worst()

    remaining_individuals = list(population.individuals.keys())
    assert len(remaining_individuals) == 2
    assert remaining_individuals[0].get_fitness() == 4
    assert remaining_individuals[1].get_fitness() == 3


def test_reproduce(model_config, model_options):
    population = FeedForward(4, 2, 0.5, 3, model_options)
    population.individuals = OrderedDict()

    for i in range(4):
        individual = FeedForwardIndividual(model_config, population)
        individual.get_fitness = MagicMock(return_value=i + 1)
        population.individuals[individual] = f"{i}"

    population._evaluate()
    population._reproduce()

    new_individuals = list(population.individuals.keys())
    assert len(new_individuals) == 4
    for child_name in population.individuals.values():
        if isinstance(child_name, str):
            # New children have a fitness value of "new_child_{i}", Check that
            # they exist.
            assert "child" in child_name


def test_fit(model_config, model_options, getdata, get_test_data):
    random.seed(69)
    with unittest.mock.patch.object(FeedForward, '_initialize_individuals', lambda x: None):
        population = FeedForward(4, 2, 0.2, 5, model_options)
        population.individuals = OrderedDict()

        for i in range(6):
            individual = FeedForwardIndividual(model_config, population, )
            individual.get_fitness = MagicMock(return_value=i + 1)
            population.individuals[individual] = f"{i}"

        X_train, y_train = getdata
        X_test, y_test = get_test_data
        # Test correct dataset behaviour with test data
        with unittest.mock.patch.object(FeedForwardIndividual, 'get_fitness', lambda x: random.randint(1, 10)):
            population.fit(X_train, y_train, X_test, y_test)
            assert population.get_training_data().X[0][0] == X_train[0][0]
            assert population.get_training_data().y[0][0] == 1
            assert population.best_individual is not None

            # Test correct dataset behaviour with no test data
            population.fit(X_train, y_train)
            assert population.get_testing_data() is not None
            assert population.get_training_data().X is not None
