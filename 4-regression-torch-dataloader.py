import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Dataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("dataset/simple_dataset.csv", dtype=np.float32, skiprows=1, delimiter=",")
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy((xy[:, :1]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = Dataset()
print("Dataset len is: ", dataset.__len__())
print("Dataset example is: ", dataset.__getitem__(5))
sample_data = dataset.__getitem__(5)
input_size, output_size = sample_data[0].size(0), sample_data[1].size(0)

train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output


lr_rate = 0.001
epochs = 10

model = Model(input_dim=input_size, hidden_dim=100, output_dim=output_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss = nn.MSELoss()

for epoch in range(epochs):
    for index, (input, label) in enumerate(train_loader):
        # forward
        y_predict = model(input)

        # loss
        l = loss(label, y_predict)

        # gradiant
        l.backward()

        # update weight
        optimizer.step()
        optimizer.zero_grad()

    print(f"Loss is {l}")

print("Predict on test data:")
with torch.no_grad():
    test_data = torch.tensor([12.34, 2.45, 2.46, 21, 98, 2.56, 2.11, .34, 1.31, 2.8, .8, 3.38, 438])
    print(model(test_data))
