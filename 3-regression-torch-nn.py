import torch
from torch import nn
from torch import optim

X_TRAIN = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]], dtype=torch.float)
Y_TRAIN = X_TRAIN * 2

input_sample, input_size = X_TRAIN.shape
input_sample, output_size = Y_TRAIN.shape

X_TEST = torch.tensor([[22], [3.5], [1.25]], dtype=torch.float)


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.lin = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, input):
        return self.lin(input)


# model = nn.Linear(in_features=input_size, out_features=output_size)
model = LinearModel(input_size, output_size)

print('Predict on test data before train:')
result = model(X_TEST)
print(result)

lr_rate = 0.001
epochs = 100

loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr_rate)

for epoch in range(epochs):
    # forward
    y_predict = model(X_TRAIN)

    # loss
    l = loss(Y_TRAIN, y_predict)

    # gradiant
    l.backward()

    # update weight
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {l:.3f}, weight: {model.lin.weight.item():.3f}')

print('Predict on test data ofter train:')
result = model(X_TEST)
print(result)
