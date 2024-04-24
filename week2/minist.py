import torch
from torchvision import transforms, datasets


minist_dataset = datasets.MNIST(
    "data/",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    ),
)
