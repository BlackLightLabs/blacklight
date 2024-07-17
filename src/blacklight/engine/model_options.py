import torch
from dataclasses import dataclass

@dataclass
class ModelOptions:
    input_shape: int
    min_dense_layers: int
    max_dense_layers: int
    min_dense_neurons: int
    max_dense_neurons: int
    dense_activation_types: torch.nn.ModuleList
    output_layer: tuple[int, torch.nn.ModuleList]
    loss: torch.nn.ModuleList
    optimizer: object
    learning_rate: float
    epochs: int
    batch_size: int
    num_classes: int

class ModelConfig: 
    def __init__(self, config: ModelOptions):
        return
