import tensorflow as tf
from unittest.mock import MagicMock
from blacklight.engine import BlacklightModel
from blacklight.engine import ModelConfig

# Helper function to create a ModelConfig instance


def get_model_config():
    config = {
        "target_layer": (1, "sigmoid"),
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ModelConfig.get_default_metrics(),
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "verbose": 0,
        "validation_split": 0.1,
        "early_stopping": True,
        "callbacks": ModelConfig.get_default_callbacks(),
        "input_shape": (32, 32, 3),
        "fitness_metric": "auc"
    }
    return ModelConfig(config)


# Helper function to create a sample gene list
def get_sample_genes():
    return [
        ("Conv2D", 32, (3, 3), "relu"),
        ("MaxPooling2D", (2, 2)),
        ("Flatten",),
        ("Dense", 128, "relu"),
    ]


def test_blacklight_model_init():
    model_config = get_model_config()
    genes = get_sample_genes()
    bl_model = BlacklightModel(model_config, genes)
    assert bl_model.model_config == model_config
    assert bl_model.genes == genes
    assert bl_model.model is None


def test_blacklight_model_create_model():
    model_config = get_model_config()
    genes = get_sample_genes()
    bl_model = BlacklightModel(model_config, genes)
    bl_model.create_model()
    assert isinstance(bl_model.model, tf.keras.Model)


def test_blacklight_model_train_model():
    model_config = get_model_config()
    genes = get_sample_genes()
    bl_model = BlacklightModel(model_config, genes)
    bl_model.create_model()
    # Mock the fit method to avoid actual training
    bl_model.model.fit = MagicMock(return_value="Some Training History")
    bl_model.train_model(tf.random.normal([100, 32, 32, 3]))
    assert bl_model.model_history is not None


def test_blacklight_model_evaluate_model():
    model_config = get_model_config()
    genes = get_sample_genes()
    bl_model = BlacklightModel(model_config, genes)
    bl_model.create_model()
    # Mock the fit and evaluate methods to avoid actual training and evaluation
    bl_model.model.fit = MagicMock(return_value=None)
    bl_model.model.evaluate = MagicMock(return_value={"auc": 0.8})
    fitness = bl_model.evaluate_model(tf.random.normal(
        [100, 32, 32, 3]), tf.random.normal([20, 32, 32, 3]))
    assert fitness == 0.8
