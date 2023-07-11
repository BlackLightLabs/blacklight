import random
import pytest
from blacklight.genetic.chromosomes import FeedForwardChromosome
from blacklight.engine import ModelConfig
from unittest.mock import MagicMock


def get_model_config() -> ModelConfig:
    model_options = {
        "layer_information": {
            "problem_type": "binary_classification",
            "input_shape": 10,
            "min_dense_layers": 1,
            "max_dense_layers": 10,
            "min_dense_neurons": 5,
            "max_dense_neurons": 10,
            "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
        },
        "problem_type": "classification",
        "num_classes": 3,
    }
    model_config = ModelConfig.parse_options_to_model_options(model_options)
    return model_config


@pytest.fixture
def seed():
    random.seed(42)


def test_feed_forward_chromosome_init(seed):
    model_config = get_model_config()
    ff_chromosome = FeedForwardChromosome(model_params=model_config)

    expected_genes = [("Dense", 5, "tanh"), ("Dense", 6, "sigmoid")]
    assert ff_chromosome.genes == expected_genes


def test_feed_forward_chromosome_crossover(seed):
    model_config = get_model_config()

    # [(Dense, 5, tanh), (Dense, 6, sigmoid)]
    chromosome_a = FeedForwardChromosome(
        model_params=model_config, mutation_prob=0.01)

    # [(Dense, 5, relu), (Dense, 9, selu), (Dense, 5, relu)]
    chromosome_b = FeedForwardChromosome(
        model_params=model_config, mutation_prob=0.01)

    new_chromosome = FeedForwardChromosome.cross_over(
        chromosome_a, chromosome_b)

    assert new_chromosome.genes is not None


def test_feed_forward_chromosome_mutate(seed):
    model_config = get_model_config()
    ff_chromosome = FeedForwardChromosome(
        model_params=model_config, mutation_prob=0.99)
    ff_chromosome._mutate()

    assert ff_chromosome.genes is not None


def test_feed_forward_chromosome_evaluate_model(seed):
    model_config = get_model_config()
    ff_chromosome = FeedForwardChromosome(model_params=model_config)

    # Mock the evaluate_model method on the BlacklightModel class
    ff_chromosome.model.evaluate_model = MagicMock(return_value=0.8)

    train_data = [100, 10]
    test_data = [20, 10]

    fitness = ff_chromosome.evaluate_model(train_data, test_data)
    assert fitness == 0.8
