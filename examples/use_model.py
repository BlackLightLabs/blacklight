import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import PILImage
import sys

n = int(sys.argv[1])

model = torch.load('model.pth')
model.eval()

real = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

real_dataloader = DataLoader(real, batch_size=16)

tran = torchvision.transforms.ToPILImage()

img = tran(real_dataloader.dataset[n][0])

img.show()

with torch.no_grad():
    pred = model(real_dataloader.dataset[n][0])
    print(torch.argmax(pred))
