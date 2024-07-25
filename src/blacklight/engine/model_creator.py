import torch
from torch.optim import optimizer
from blacklight.engine.model_options import ModelConfig
from torch.utils.data import DataLoader

# the BlacklightModel class inherits nn.Module effectively making the class itself represent the neural network
class BlacklightModel(torch.nn.Module):
    # model_config: ModelConfig is a class representing the model configuration
    # config properties are declared withing the ModelConfig constructor 
    # these values can later be changed by model_config.item = value
    def __init__(self, model_config: ModelConfig, genes: list[tuple[object, *tuple[int, ...], object]]):
        super(BlacklightModel, self).__init__() # pyright: ignore [reportUnknownMemberType]
        self.model_config = model_config
        self.genes = genes
        # self.model = torch.nn.Sequential()

        # layers is the neural network
        # when you call print on the model it will print the layers variable
        self.layers = torch.nn.ModuleList()
        self.input_shape = model_config.input_size
        
        # turn genes into layers
        # TODO: Instead of gene[0] being a string, make it a module.
        # EX: gene = (torch.nn.Linear, ...opts..., activation)
        for gene in genes:
            # print(gene[0])
            # if gene[0] == "Conv2D":
            #     self.layers.append(
            #         torch.nn.Conv2d(
            #             in_channels=self.input_shape,
            #             out_channels=gene[1],
            #             kernel_size=gene[2]
            #         )
            #     ) 
            # elif gene[0] == "Linear":
            #     self.layers.append(
            #         torch.nn.Linear(gene[1], gene[2])
            #     )
            # elif gene[0] == "Flatten":
            #     self.layers.append(torch.nn.Flatten())
            # elif gene[0] == "MaxPooling2D":
            #     self.layers.append(
            #         torch.nn.MaxPool2d(kernel_size=gene[1])
            #     )
            # else:
            #     raise ValueError(f"Invalid gene type: {gene[0]}")
            opt = True if callable(gene[len(gene)-1]) else False
            params = []
            if opt and len(gene) > 1:
                for p in range(1,len(gene)-1):
                    params.append(gene[p])
                self.layers.append(gene[0](*params))
                self.layers.append(gene[len(gene)-1]())
            else:
                for p in range(1, len(gene)):
                    params.append(gene[p])
                self.layers.append(gene[0](*params))
            # if callable(gene[len(gene)-1]):
            #     # activation = getattr(torch.nn, gene[len(gene)-1])()
            #     activation = gene[len(gene)-1]()
            #     self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # return the layers that represent the model
    def get_model(self):
        return self.layers

    def get_model_history(self):
        return self.model_history

    def train_model(self, data: DataLoader):
        # set the model to train mode
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train
        self.train()
       
        # pull the optimizer from the model config and pass in the parameters (optimizer)(parameters)
        optimizer = (self.model_config.optimizer)(self.parameters(), self.model_config.learning_rate)
        # loss function
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.model_config.class_weight,
        )
        
        # TODO: The training paradigm for this library needs to be determined
        # The current implementation requires training for the right number of epochs to be handled elsewhere
        # EX:
        # ```py
        # for e in range(config.epochs):
        #   print(f"Epoch {e+1}\n------------------")
        #   model.train_model(train_dataloader)
        #   model.evaluate(real_dataloader)
        # print("Done!")
        # ```
        # Alternatively, the training loop could take run for the right number of epochs right here, but that would remove the
        # ability to train for 1 step

        # for epoch in range(self.model_config.model_options.epochs):
        size = len(data.dataset)
        batch_size = self.model_config.batch_size 
        self.train()
        for batch, (X, y) in enumerate(data):
            # self(X) is the same as using model(X) after compilation
            # EX: model = torch.load("some_model.pt")
            # model(X)
            pred = self(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)

            # TODO: Make conditional on a verbosity variable
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate(self, data):
        criterion = torch.nn.CrossEntropyLoss()
        self.eval()
        size = len(data.dataset)
        num_batches = len(data)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in data:
                # self(X) is the same as using model(X) after compilation
                # EX: model = torch.load("some_model.pt")
                # model(X)
                pred = self(X)
                # criterion is the loss function
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

 

    # def _serialize_model_options_into_genes(self, model_config: ModelConfig):
    #     # self.genes = { # pywrong: ignore [reportUnknownMemberType]
    #     #     "input" : self.model_config.default_config.input_size,
    #     #     "hidden_layers" : [], # layer_type, size, activation
    #     #     "output" : [], # layer_type, size, activation | None
    #     #     "optimizer": torch.optim.Adam,
    #     # }
    #     
    #     # layers
    #     self.genes = [
    #         ("linear", )
    #     ]
    # ^!!!THIS IS NOT HOW THE GENES ARE STRUCTURED ANYMORE!!!
