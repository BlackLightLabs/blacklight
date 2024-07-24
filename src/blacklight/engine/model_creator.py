import torch
from torch.optim import optimizer
from blacklight.engine.model_options import ModelConfig

class BlacklightModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, genes: list[tuple[any]]):
        super(BlacklightModel, self).__init__() # pyright: ignore [reportUnknownMemberType]
        self.model_config = model_config
        self.genes = genes
        # self.model = torch.nn.Sequential()
        self.layers = torch.nn.ModuleList()
        self.input_shape = model_config.input_size

        for gene in genes:
            print(gene[0])
            if gene[0] == "Conv2D":
                self.layers.append(
                    torch.nn.Conv2d(
                        in_channels=self.input_shape,
                        out_channels=gene[1],
                        kernel_size=gene[2]
                    )
                ) 
            elif gene[0] == "Linear":
                self.layers.append(
                    torch.nn.Linear(gene[1], gene[2])
                )
            elif gene[0] == "Flatten":
                self.layers.append(torch.nn.Flatten())
            elif gene[0] == "MaxPooling2D":
                self.layers.append(
                    torch.nn.MaxPool2d(kernel_size=gene[1])
                )
            else:
                raise ValueError(f"Invalid gene type: {gene[0]}")
            if callable(gene[len(gene)-1]):
                # activation = getattr(torch.nn, gene[len(gene)-1])()
                activation = gene[len(gene)-1]()
                self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_model(self):
        return self.model

    def get_model_history(self):
        return self.model_history

    def train_model(self, data):
        self.train()
        
        optimizer = (self.model_config.optimizer)(self.parameters(), self.model_config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.model_config.class_weight,
        )

        # for epoch in range(self.model_config.model_options.epochs):
        size = len(data.dataset)
        batch_size = self.model_config.batch_size 
        self.train()
        for batch, (X, y) in enumerate(data):
            pred = self(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # TODO: Make conditional on a verbosity variable
            # if True:
            #     print(f"Epoch [{epoch+1}/{self.model_config.model_options.epochs}], Loss: {epoch_loss}, Accuracy: {epoch_acc}")
            #
    def evaluate(self, data):
        criterion = torch.nn.CrossEntropyLoss()
 #        self.model.eval()
 #        
 #        self.train_model(train_data)
 #        
 #        running_loss = 0.0
 #        correct = 0
 #        total = 0
 #
 #        with torch.no_grad():
 #            for inputs, labels in train_data: 
 #                outputs = self(inputs)
 #                loss = criterion(outputs, labels)
 #                _, predicted = torch.max(outputs.data, 1)
 #                total += labels.size(0)
 #                correct += (predicted == labels).sum().item()
 #                running_loss += loss.item()
 #
 #        epoch_loss = running_loss / len(test_load)
 #        epoch_acc = correct / total
 #
 #        print(f"Evaluation Loss: {epoch_loss:.4f}, Evaluation Accuracy: {epoch_acc:.2%}")
 # 
 #        return epoch_acc
        self.eval()
        size = len(data.dataset)
        num_batches = len(data)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in data:
                pred = self(X)
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
