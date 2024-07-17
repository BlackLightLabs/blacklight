from blacklight.genetic.populations import FeedForward
import pandas as pd
import pytest


@pytest.mark.slow
def test_end_to_end_feed_forward_NN():
    data = pd.read_csv('blacklight/genetic/populations/tests/data/Iris.csv', index_col=0)
    model_options = {
        "layer_information": {
            "problem_type": "classification",
            "input_shape": 4,
            "min_dense_layers": 1,
            "max_dense_layers": 8,
            "min_dense_neurons": 10,
            "max_dense_neurons": 50,
            "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
        },
        "problem_type": "classification",
        "num_classes": 3,
    }
    ff_autoML = FeedForward(8, 2, 0.1, 4, model_options)
    # Fit the model
    ff_autoML.fit(data)
    # Get model
    ff_autoML.print_model_summary()
    ff_autoML.print_model_training_history()
