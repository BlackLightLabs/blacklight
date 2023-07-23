from typing import Optional
from tensorflow import keras


class ModelConfig:
    # def __init__(self, config: Optional[dict] = None):
    #     self.config = config
    #     self.check_config()

    def __init__(self, config: Optional[dict] = None):

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
            "target_layer": (1, "sigmoid"),
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ModelConfig.get_default_metrics(),
            "learning_rate": 0.001,
            "epochs": 1000,
            "batch_size": 32,
            "problem_type": "classification",
            "num_classes": 3, }


        default_config = model_options

        if config is not None:

            # layer_information_config = {"layer_information": config["layer_information"]}
            # other_config = {key: value for key, value in config.items() if key != "layer_information"}

            # for key, value in layer_information_config.items():
            #     if key == "layer_information":
            #         default_config[key] = value
            default_config["layer_information"].update(config["layer_information"])
            print(default_config)
            self.config = default_config
        else:
            self.config = default_config

        #split the dictionary into two dictionaries, one for layer_information and one for the rest, and try use update witht that

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def check_config(self):

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

    @staticmethod
    def parse_options_to_model_options(options):
        config = {}
        if options is None:
            options = {"layer_information": {
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

        # if options = none, set it to the dictonary used in test
        # write documentation explaining what each of the paramters are, and what choices the user has
        # try if the user only implements one parameter, and not the other, what happens?
        # HINT; dont write a bunch of if conditions, use the get function. if it dosent exist set it to the default.

        # implement value errors when important things arent given. when u have layer information, u must have layer shape
        if options.get("layer_information"):
            layer_information = options.get("layer_information")
            config["input_shape"] = layer_information.get("input_shape")
            # Get Dense Layer Information
            config["max_dense_layers"] = layer_information.get(
                "max_dense_layers")
            config["min_dense_layers"] = layer_information.get(
                "min_dense_layers")
            config["min_dense_neurons"] = layer_information.get(
                "min_dense_neurons")
            config["max_dense_neurons"] = layer_information.get(
                "max_dense_neurons")
            config["dense_activation_types"] = layer_information.get(
                "dense_activation_types", ["relu", "sigmoid", "tanh", "selu"])
            # Get Convolutional Layer Information
            config["max_conv_layers"] = layer_information.get(
                "max_conv_layers")
            config["min_conv_layers"] = layer_information.get(
                "min_conv_layers")
            # Get Dropout Layer Information
            config["max_dropout_layers"] = layer_information.get(
                "max_dropout_layers")

        # Determine problem type, which sets the last layer of the model.

        if options.get("problem_type") == "classification":
            if "num_classes" not in options:
                raise ValueError(
                    "ModelConfig has no num_classes set for problem type Classification.")
            config["target_layer"] = (options["num_classes"], "softmax")
            config["loss"] = "categorical_crossentropy"
            config["num_classes"] = options.get("num_classes")
        elif options.get("problem_type") == "binary_classification":
            config["target_layer"] = (1, "sigmoid")
            config["loss"] = "binary_crossentropy"
            config["num_classes"] = 2
        elif options.get("problem_type") == "regression":
            config["target_layer"] = (1, "linear")
            config["loss"] = "mse"

        config["problem_type"] = options.get("problem_type")

        # Add all the model creation options to the config
        config["optimizer"] = options.get("optimizer", "adam")
        config["metrics"] = options.get(
            "metrics", ModelConfig.get_default_metrics())
        config["learning_rate"] = options.get("learning_rate", 0.001)

        # Add all the training options to the config

        config["epochs"] = options.get("epochs", 1000)
        config["batch_size"] = options.get("batch_size", 32)
        config["verbose"] = options.get("verbose", 0)
        config["class_weight"] = options.get("class_weight", None)
        config["validation_data"] = options.get("validation_data", None)
        config["use_multiprocessing"] = options.get(
            "use_multiprocessing", False)
        config["early_stopping"] = options.get("early_stopping", True)
        config["callbacks"] = options.get(
            "callbacks", ModelConfig.get_default_callbacks())
        config["output_bias"] = options.get("output_bias", None)
        config["fitness_metric"] = options.get("fitness_metric", "auc")

        return ModelConfig(config)
