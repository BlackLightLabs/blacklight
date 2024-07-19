import torch
from dataclasses import dataclass, fields

@dataclass
class ModelOptions:
    input_size: int = 4
    min_dense_layers: int = 1
    max_dense_layers: int = 8
    min_dense_neurons: int = 2
    max_dense_neurons: int = 8
    dense_activation_types: list[object] = [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU]
    output_layer: tuple[int, list[object]] = (1, [torch.nn.Sigmoid])
    loss: object = torch.nn.BCELoss
    optimizer: object = torch.optim.Adam
    learning_rate: float = 0.001
    epochs: int = 1000
    batch_size: int = 32
    num_classes: int = 3
    verbose: int = 0
    class_weight = None
    validation_data = None
    use_multiprocessing: bool = False
    early_stopping: bool = True



class ModelConfig: 
    default_config: ModelOptions

    def __init__(self, config: ModelOptions | None = None):
        model_options = ModelOptions(
            input_size=4,  
            min_dense_layers=1, 
            max_dense_layers=8, 
            min_dense_neurons=2, 
            max_dense_neurons=8, 
            dense_activation_types=[torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU],
            output_layer=(1, [torch.nn.Sigmoid]),
            loss=torch.nn.BCELoss,
            optimizer=torch.optim.Adam,
            learning_rate=0.001,
            epochs=1000,
            batch_size=32,
            num_classes=3
        )

        self.default_config = model_options

        if config is not None:
            for field in fields(config):
                setattr(self.default_config, field.name, field)

    def print_config(self):
        for field in fields(self.default_config):
            print(field.name + ":", field)


