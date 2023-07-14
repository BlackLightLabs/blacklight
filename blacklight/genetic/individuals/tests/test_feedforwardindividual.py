import pytest
from unittest.mock import MagicMock
from blacklight.genetic.chromosomes import FeedForwardChromosome
from blacklight.genetic.individuals import FeedForwardIndividual
from blacklight.engine import ModelConfig


@pytest.fixture
def model_config():
    model_options = {
        "layer_information": {
            "problem_type": "binary_classification",
            "input_shape": 2,
            "min_dense_layers": 1,
            "max_dense_layers": 8,
            "min_dense_neurons": 1,
            "max_dense_neurons": 4,
            "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
        },
        "problem_type": "classification",
        "num_classes": 3,
    }
    model_config = ModelConfig.parse_options_to_model_options(model_options)
    return model_config


@pytest.fixture
def mock_population(model_config):
    population = MagicMock()
    population.problem_type = "classification"
    population.num_classes = 10
    population.get_training_data.return_value = (None, None)
    population.get_testing_data.return_value = (None, None)
    return population


def test_initialization(model_config, mock_population):
    individual = FeedForwardIndividual(model_config, mock_population)
    assert isinstance(individual, FeedForwardIndividual)
    assert isinstance(individual.chromosome, FeedForwardChromosome)
    assert individual.chromosome.model_params == model_config
    assert individual.chromosome is not None


def test_get_fitness(model_config, mock_population):
    individual = FeedForwardIndividual(model_config, mock_population)
    individual.chromosome.evaluate_model = MagicMock(return_value=0.75)
    fitness = individual.get_fitness()
    assert fitness == 0.75


def test_mate(model_config, mock_population):
    individual1 = FeedForwardIndividual(model_config, mock_population)
    individual2 = FeedForwardIndividual(model_config, mock_population)
    child = individual1.mate(individual2)
    assert isinstance(child, FeedForwardIndividual)
    assert child.chromosome != individual1.chromosome
    assert child.chromosome != individual2.chromosome
