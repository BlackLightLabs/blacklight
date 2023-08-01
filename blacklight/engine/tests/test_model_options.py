import pytest
from blacklight.engine import ModelConfig

# create a new unit test, modeled after these. if we dont pass any configs, it still has all the parameters
# pytest, end with test or start with test


full_config = {

    "problem_type": "classification",  # Specifies the type of problem, which is classification.
    "input_shape": 4,  # Defines the input shape of the data.
    "min_dense_layers": 1,  # Sets the minimum number of dense layers in the neural network
    "max_dense_layers": 8,  # Sets the maximum number of dense layers in the neural network
    "min_dense_neurons": 2,  # Sets the minimum number of neurons in each dense layer
    "max_dense_neurons": 8,  # Sets the maximum number of neurons in each dense layer.
    "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"],
    # Specifies the activation functions that can be used in the dense layers.

    "target_layer": (1, "sigmoid"),
    # Sets the target layer to the first dense layer, and the activation function for this layer to "sigmoid".

    "loss": "binary_crossentropy",
    # Sets the loss function to "binary_crossentropy" which is commonly used for binary classification tasks.

    "optimizer": "adam",
    # Sets the optimizer algorithm to "adam". This is a popular optimizer algorithm for neural networks.

    "metrics": ModelConfig.get_default_metrics(),
    # Specifies the evaluation metrics to be used during training. This line likely references a method that returns
    # default metrics for the classification task.

    "learning_rate": 0.001,
    # Sets the learning rate for the optimizer to 0.001, controlling the step size during gradient descent.

    "epochs": 1000,
    # Specifies the number of training iterations the model will undergo during training.

    "batch_size": 32,
    # Sets the number of samples in each mini-batch used for training.

    "num_classes": 3,
    # Sets the number of classes in the classification task to 3.

    "verbose": 0,
    # Sets the verbosity level during training. Default is 0 (silent).

    "class_weight": None,
    # Sets the class weights for handling imbalanced datasets. Default is None (equal weights).

    "validation_data": None,
    # Sets the validation data to be used during training. Default is None (no validation data).

    "use_multiprocessing": False,
    # Specifies whether to use multiprocessing during training. Default is False.

    "early_stopping": True,
    # Specifies whether to use early stopping during training. Default is True.

    "callbacks": ModelConfig.get_default_callbacks(),
    # Specifies the callbacks to be used during training. Default is a method that returns default callbacks.

    "output_bias": None,
    # Sets the output bias for the model. Default is None (no output bias).

    "fitness_metric": "auc"
    # Sets the fitness metric to be used for evaluating the model's performance. Default is "auc".
}

default_config = {
    "problem_type": "classification",
    "input_shape": 4,
    "min_dense_layers": 1,
    "max_dense_layers": 8,
    "min_dense_neurons": 2,
    "max_dense_neurons": 8,
    "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"],
    "target_layer": (1, "sigmoid"),
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ModelConfig.get_default_metrics(),
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 32,
    "num_classes": 3, }

partial_config = {
    "min_dense_layers": 2,
    "max_dense_layers": 91,
    "min_dense_neurons": 2,
    "max_dense_neurons": 9,
    "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"],
    "target_layer": (1, "sigmoid"),
    "loss": "ammar",
    "problem_type": "regression",
    "num_classes": 9, }


@pytest.mark.parametrize("config", [None, full_config, partial_config])
def test_model_config(config):
    model_config = ModelConfig(config)
    assert isinstance(model_config, ModelConfig)
    for key in default_config.keys():
        assert key in model_config.config


def test_other_problem_types_parse():
    config_one = {
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "problem_type": "binary_classification"
    }
    config_two = {
        "loss": "mse",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "problem_type": "regression"
    }
    mc = ModelConfig(config_one)
    rc = ModelConfig(config_two)
    assert mc.config is not None
    assert rc.config is not None
