# very icky code maximum badness

import torch
import torch.nn as nn
import torch.optim as optim

from blacklight.engine.model_options import ModelConfig

class BlacklightModel:
    def __init__(self, model_config: ModelConfig, genes):
        self.model_history = None
        self.model_config = model_config
        self.genes = genes
        self.model = None

    def create_model(self):
        layers = []
        input_shape = tuple(self.model_config.get("input_shape"))
        in_channels = input_shape[0]

        for gene in self.genes:
            if gene[0] == "Conv2D":
                out_channels = gene[1]
                kernel_size = gene[2]
                activation = gene[3]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                in_channels = out_channels
            elif gene[0] == "MaxPooling2D":
                kernel_size = gene[1]
                layers.append(nn.MaxPool2d(kernel_size))
            elif gene[0] == "Flatten":
                layers.append(nn.Flatten())
            elif gene[0] == "Dense":
                out_features = gene[1]
                activation = gene[2]
                layers.append(nn.Linear(in_channels, out_features))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                in_channels = out_features
            else:
                raise ValueError(f"Invalid gene type: {gene[0]}")

        target_layer = self.model_config.get("target_layer")
        layers.append(nn.Linear(in_channels, target_layer[0]))
        if target_layer[1] == "sigmoid":
            layers.append(nn.Sigmoid())
        elif target_layer[1] == "softmax":
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

        optimizer_name = self.model_config.get("optimizer")
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters())
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.model_config.get("learning_rate"))
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")

        self.loss_fn = nn.CrossEntropyLoss()  # Assuming CrossEntropyLoss for classification
        self.optimizer = optimizer

    def train_model(self, train_data):
        if self.model is None:
            self.create_model()

        epochs = self.model_config.get("epochs")
        batch_size = self.model_config.get("batch_size")
        validation_split = self.model_config.get("validation_split")

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

        self.model_history = []

    def get_model(self):
        return self.model

    def get_model_history(self):
        return self.model_history

    def evaluate_model(self, test_data):
        if self.model is None:
            self.create_model()

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.model_config.get("batch_size"), shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

