import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dataset = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=10)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=10)

input_size = torch.tensor(28 * 28)
output_size = torch.tensor(10)
hidden_size = torch.tensor(100)


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        return output


epochs = 10
lr_rate = 0.001
device = "cuda"
model = Model(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_rate)
loss = nn.CrossEntropyLoss()

model.to(device)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        predicts = model(images)

        # gradiant
        l = loss(predicts, labels)

        # Loss
        l.backward()

        # gradiant
        optimizer.step()
        optimizer.zero_grad()
    print(f"Loss is {l}")

true_labels = np.array([])
predict_labels = np.array([])
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.numpy()
        predicts = model(images)
        predicts = torch.argmax(predicts, dim=-1).to("cpu").numpy()

        true_labels = np.concatenate((true_labels, labels))
        predict_labels = np.concatenate((predict_labels, predicts))
print("Result on test data:")
print(classification_report(true_labels, predict_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
