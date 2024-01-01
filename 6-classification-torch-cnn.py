from torch import nn
from torch.utils import data
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
from torch.nn import functional

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root="./mnist", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./mnist", train=False, download=True, transform=transform)

train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=4, num_workers=1)
test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=4, num_workers=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.pool(functional.relu(self.cnn1(input)))
        out = self.pool(functional.relu(self.cnn2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return out


model = Model()

device = "cuda"
epochs = 10
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.to(device)
for epoch in range(epochs):
    for i, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)

        # Forward
        output = model(image)

        # Backward
        l = loss(output, label)
        l.backward()

        # Optimizer
        optimizer.step()
        optimizer.zero_grad()

    print(f"Loss: {l}")

print()
