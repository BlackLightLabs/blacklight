from typing import Optional
from tensorflow import keras


class ModelConfig:
    def __init__(self, config: Optional[dict] = None):

        model_options = {
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

        default_config = model_options

        # Update default_config with values from config (if not None)
        if config is not None:
            for key in config.keys():
                if key in default_config:
                    default_config[key] = config[key]

        # Update default_config based on the problem type
        if default_config.get("problem_type") == "classification":
            if "num_classes" not in default_config:
                raise ValueError("ModelConfig has no num_classes set for problem type Classification.")
            default_config["target_layer"] = (default_config.get("num_classes"), "softmax")
            default_config["loss"] = "categorical_crossentropy"
        elif default_config.get("problem_type") == "binary_classification":
            default_config["target_layer"] = (1, "sigmoid")
            default_config["loss"] = "binary_crossentropy"
            default_config["num_classes"] = 2
        elif default_config.get("problem_type") == "regression":
            default_config["target_layer"] = (1, "linear")
            default_config["loss"] = "mse"

        # Set default values for the remaining attributes, not set by the user
        default_config["verbose"] = default_config.get("verbose", 0)
        default_config["class_weight"] = default_config.get("class_weight", None)
        default_config["validation_data"] = default_config.get("validation_data", None)
        default_config["use_multiprocessing"] = default_config.get("use_multiprocessing", False)
        default_config["early_stopping"] = default_config.get("early_stopping", True)
        default_config["callbacks"] = default_config.get("callbacks", ModelConfig.get_default_callbacks())
        default_config["output_bias"] = default_config.get("output_bias", None)
        default_config["fitness_metric"] = default_config.get("fitness_metric", "auc")

        # Set self.config to the final default_config
        self.config = default_config
        self.check_config()

    def get(self, key: str, default=None):
        if key is None or self.config is None:
            return ValueError("Key cannot be None")
        else:
            return self.config.get(key, default)

    def check_config(self):
        print(self.config)

        if self.config is None:
            raise ValueError("ModelConfig has no config set.")
        if "target_layer" not in self.config:
            raise ValueError("ModelConfig has no target_layer set.")
        if "loss" not in self.config:
            raise ValueError("ModelConfig has no loss set.")
        if "optimizer" not in self.config:
            raise ValueError("ModelConfig has no optimizer set.")
        if "metrics" not in self.config:
            raise ValueError("ModelConfig has no metrics set.")
        if "learning_rate" not in self.config:
            raise ValueError("ModelConfig has no learning_rate set.")
        if "epochs" not in self.config:
            raise ValueError("ModelConfig has no epochs set.")
        if "batch_size" not in self.config:
            raise ValueError("ModelConfig has no batch_size set.")
        if self.config.get("problem_type") == "classification":
            if "num_classes" not in self.config:
                raise ValueError(
                    "ModelConfig has no num_classes set for problem type Classification.")
            if self.config.get("loss") != "categorical_crossentropy":
                raise ValueError(
                    f"ModelConfig has invalid loss {self.config.get('loss')} for problem type multiclass classification.")
        elif self.config.get("problem_type") == "binary_classification":
            if self.config.get("loss") != "binary_crossentropy":
                raise ValueError(
                    f"ModelConfig has invalid loss {self.config.get('loss')} for problem type binary classification.")

    @staticmethod
    def get_default_metrics():
        return [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

    @staticmethod
    def get_default_callbacks():
        return keras.callbacks.EarlyStopping(
            monitor='auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)
