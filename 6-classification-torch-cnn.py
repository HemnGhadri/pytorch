from torch import nn
from torch.utils import data
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="./mnist", train=True, download=True, transform=transforms)
test_data = torchvision.datasets.CIFAR10(root="./mnist", train=False, download=True, transform=transforms)

train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=10, num_workers=1)
test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=10, num_workers=1)

print()
