import blacklight.engine.model_options
import blacklight.engine.model_creator
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# TODO: You should be able to do ModelConfig(option=choice) this is only possible if you use ModelOptions
# why does ModelConfig exists in this context
config = blacklight.engine.model_options.ModelConfig(epochs=10)

genes = [
    ("Flatten",),
    ("Linear", 28*28, 512, torch.nn.ReLU),
    ("Linear", 512, 512, torch.nn.ReLU),
    ("Linear", 512, 10)
]

model = blacklight.engine.model_creator.BlacklightModel(model_config=config, genes=genes)

print(model)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

real = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
real_dataloader = DataLoader(real, batch_size=config.batch_size)

for e in range(config.epochs):
    print(f"Epoch {e+1}\n------------------")
    model.train_model(train_dataloader)
    model.evaluate(real_dataloader)
print("Done!")

torch.save(model, 'model.pth')
