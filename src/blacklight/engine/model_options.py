import torch
from dataclasses import dataclass, fields, field

# @dataclass
# class ModelOptions:
#     input_size: int = 4
#     min_dense_layers: int = 1
#     max_dense_layers: int = 8
#     min_dense_neurons: int = 2
#     max_dense_neurons: int = 8
#     dense_activation_types: list[object] = field(default_factory=lambda: [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU])
#     output_layer: tuple[int, list[object]] = (1, [torch.nn.Sigmoid])
#     loss: object = torch.nn.BCELoss
#     optimizer: object = torch.optim.Adam
#     learning_rate: float = 0.001
#     epochs: int = 1000
#     batch_size: int = 32
#     num_classes: int = 3
#     verbose: int = 0
#     class_weight = None
#     validation_data = None
#     use_multiprocessing: bool = False
#     early_stopping: bool = True



class ModelConfig:
    # input_size: int = 4
    # min_dense_layers: int = 1
    # max_dense_layers: int = 8
    # min_dense_neurons: int = 2
    # max_dense_neurons: int = 8
    # dense_activation_types: list[object] = field(default_factory=lambda: [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU])
    # output_layer: tuple[int, list[object]] = (1, [torch.nn.Sigmoid])
    # loss: object = torch.nn.BCELoss
    # optimizer: object = torch.optim.Adam
    # learning_rate: float = 0.001
    # epochs: int = 1000
    # batch_size: int = 32
    # num_classes: int = 3
    # verbose: int = 0
    # class_weight = None
    # validation_data = None
    # use_multiprocessing: bool = False
    # early_stopping: bool = True
    # def __init__(self, config: ModelOptions | None = None):
    #     self.model_options = ModelOptions(
    #         input_size=4,  
    #         min_dense_layers=1, 
    #         max_dense_layers=8, 
    #         min_dense_neurons=2, 
    #         max_dense_neurons=8, 
    #         dense_activation_types=[torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU],
    #         output_layer=(1, [torch.nn.Sigmoid]),
    #         loss=torch.nn.BCELoss,
    #         optimizer=torch.optim.Adam,
    #         learning_rate=0.001,
    #         epochs=1000,
    #         batch_size=32,
    #         num_classes=3
    #     )
    
    def __init__(self, 
                input_size: int=4,
                min_dense_layers: int = 1,
                max_dense_layers: int = 8,
                min_dense_neurons: int = 2,
                max_dense_neurons: int = 8,
                dense_activation_types: list[object] | None = None,
                output_layer: tuple[int, list[object]] | None = None,
                loss: object = torch.nn.BCELoss,
                optimizer: object = torch.optim.Adam,
                learning_rate: float = 0.001,
                epochs: int = 10,
                batch_size: int = 32,
                num_classes: int = 3,
                verbose: int = 0,
                class_weight = None, # pyright: ignore [reportUnknownParameterType, reportMissingParameterType]
                validation_data = None, # pyright: ignore [reportUnknownParameterType, reportMissingParameterType]
                use_multiprocessing: bool = False,
                early_stopping: bool = True,
                 ):
        self.input_size: int = input_size
        self.min_dense_layers: int = min_dense_layers
        self.max_dense_layers: int = max_dense_layers
        self.min_dense_neurons: int = min_dense_layers
        self.max_dense_neurons: int = max_dense_layers
        self.dense_activation_types: list[object] = [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.SELU] if dense_activation_types is None else dense_activation_types
        self.output_layer: tuple[int, list[object]] = (1, [torch.nn.Sigmoid]) if output_layer is None else output_layer
        self.loss: object = loss
        self.optimizer: object = optimizer
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.num_classes: int = num_classes
        self.verbose: int = verbose
        self.class_weight = class_weight # pyright: ignore [reportUnknownMemberType]
        self.validation_data = validation_data # pyright: ignore [reportUnknownMemberType]
        self.use_multiprocessing: bool = use_multiprocessing
        self.early_stopping: bool = early_stopping

        # if config is not None:
        #     for field in fields(config):
        #         setattr(self.model_options, field.name, field)

    def print_config(self):
        print(vars(self))
