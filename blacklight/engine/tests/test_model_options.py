import pytest
from blacklight.engine import ModelConfig


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
