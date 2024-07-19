import torch
from .model_options import ModelConfig
import random


class BlacklightModel:
    def __init__(self, model_config: ModelConfig, genes: None | None = None):
        self.model_history = None
        self.model_config = model_config
        self.genes = genes
        self.model = None
    
    def serialize_model_options_into_genes(self, model_config: ModelConfig):
        self.genes = { # pyright: ignore [reportUnknownMemberType]
            "input" : self.model_config.default_config.input_size,
            "hidden_layers" : [], # layer_type, size, activation
            "output" : [], # layer_type, size, activation | None
            "optimizer": torch.optim.Adam,
        }

        for i in range(model_config.default_config.min_dense_layers, random.randint(model_config.default_config.min_dense_layers, model_config.default_config.max_dense_layers)):
            self.genes["hidden_layers"].append((torch.nn.Linear, random.randint(model_config.default_config.min_dense_neurons, model_config.default_config.max_dense_neurons), random.choice(model_config.default_config.dense_activation_types))) # ppyright: ignore []
            # self.genes["hidden_layers"].append(1)
