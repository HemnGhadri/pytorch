import torch
from torch import optim
from torch import nn

# train data
X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=torch.float)

# test data
X_TEST = torch.tensor([12, 3.5, 10.25], dtype=torch.float)


def forward(x, w):
    return x * w


def predict(x, w):
    with torch.no_grad():
        return x * w


w = torch.tensor(0, dtype=torch.float, requires_grad=True)
epochs = 100
lr_rate = torch.tensor(0.001)
loss = nn.MSELoss()
optimizer = optim.SGD([w], lr=lr_rate)

for epoch in range(epochs):
    # prediction
    y_predict = forward(X, w)

    # loss
    l = loss(Y, y_predict)

    # gradiant
    l.backward()

    # update weight
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {l:.3f}, weight: {w:.3f}')

# predict test data
print('Predict on test data:')
result = predict(X_TEST, w)
print(result)
