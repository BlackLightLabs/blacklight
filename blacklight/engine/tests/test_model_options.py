import pytest
from blacklight.engine import ModelConfig


#create a new unit test, modeled after these. if we dont pass any configs, it still has all the parameters
#pytest, end with test or start with test

full_config = {

    "layer_information": {
        "problem_type": "classification",
        "input_shape": 4,
        "min_dense_layers": 1,
        "max_dense_layers": 8,
        "min_dense_neurons": 2,
        "max_dense_neurons": 8,
        "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
    },
    "target_layer": (1, "sigmoid"),
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ModelConfig.get_default_metrics(),
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 32,
    "problem_type": "classification",
    "num_classes": 3,
}

default_config = {
    "layer_information": {
        "problem_type": "classification",
        "input_shape": 4,
        "min_dense_layers": 1,
        "max_dense_layers": 8,
        "min_dense_neurons": 2,
        "max_dense_neurons": 8,
        "dense_activation_types": ["relu", "sigmoid", "tanh", "selu"]
    },
    "target_layer": (1, "sigmoid"),
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ModelConfig.get_default_metrics(),
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 32,
    "problem_type": "classification",
    "num_classes": 3,}

partial_config = {
    "layer_information": {
        "input_shape": 99,
        "min_dense_layers": 15,
        "dense_activation_types": ["downy", "down", "tanh", "selu"],
        "problem_type": "classification",
    }
}

@pytest.mark.parametrize("config",[None, partial_config])
def test_model_config(config):


    model_config = ModelConfig(config)
    assert isinstance(model_config, ModelConfig)
    for key, value in default_config.items():
        assert key in model_config.config
    print(model_config.config)



def test_model_config_init():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32
    }
    model_config = ModelConfig(config)
    assert model_config.config == config


def test_model_config_get():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32
    }
    model_config = ModelConfig(config)
    assert model_config.get("target_layer") == (1, "sigmoid")
    assert model_config.get("optimizer") == "adam"
    assert model_config.get(
        "nonexistent_key",
        "default_value") == "default_value"


def test_model_config_check_config():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32
    }
    model_config = ModelConfig(config)
    assert model_config.config == config


def test_model_config_check_config_missing_key():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32
    }
    for key in config.keys():
        invalid_config = config.copy()
        del invalid_config[key]
        with pytest.raises(ValueError, match=f"ModelConfig has no {key} set."):
            ModelConfig(invalid_config)


def test_model_config_check_config_invalid_loss_classification():
    config = {
        "target_layer": (3, "softmax"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "problem_type": "classification",
        "num_classes": 3
    }
    with pytest.raises(ValueError,
                       match="ModelConfig has invalid loss binary_crossentropy for problem type multiclass classification."):
        ModelConfig(config)


def test_model_config_is_na():
    with pytest.raises(ValueError, match="ModelConfig has no config set."):
        ModelConfig()


def test_model_config_check_config_invalid_num_classes_with_classification():
    config = {
        "target_layer": (3, "softmax"),
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "problem_type": "classification"
    }
    with pytest.raises(ValueError, match="ModelConfig has no num_classes set for problem type Classification."):
        ModelConfig(config)

    with pytest.raises(ValueError, match="ModelConfig has no num_classes set for problem type Classification."):
        ModelConfig.parse_options_to_model_options(config)


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
    mc = ModelConfig.parse_options_to_model_options(config_one)
    rc = ModelConfig.parse_options_to_model_options(config_two)
    assert mc.config is not None
    assert rc.config is not None


def test_model_config_check_config_invalid_loss_binary_classification():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "problem_type": "binary_classification"
    }
    with pytest.raises(ValueError,
                       match="ModelConfig has invalid loss categorical_crossentropy for problem type binary classification."):
        ModelConfig(config)
