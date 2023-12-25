import numpy
import numpy as np

# train data
X = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = numpy.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# test data
X_TEST = np.array([12, 3.5, 10.25])


def forward(x, w):
    return x * w


def loss(y, y_predict):
    return ((y - y_predict) ** 2).mean()


def gradiant(x, y, y_predict):
    # return d/dw = Derivative(loss funtion) = Derivative((xw - y)^2) = Derivative((xw)^2 - y^2 - 2xwy) = 2x(xw - y)
    #             = 2x(y_pred-y) gradiant = 1/n (d/dw)
    return np.dot(2 * x, (y_predict - y)).mean()


def predict(x, w):
    return x * w


w = 0
epochs = 100
lr_rate = 0.0001

for epoch in range(epochs):
    # prediction
    y_predict = forward(X, w)

    # loss
    l = loss(Y, y_predict)

    # gradiant
    dw = gradiant(X, Y, y_predict)

    # update weight
    w -= dw * lr_rate

    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {l:.3f}, weight: {w:.3f}')

# predict test data
result = predict(X_TEST, w)
print(result)
