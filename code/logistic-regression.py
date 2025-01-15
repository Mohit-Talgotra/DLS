import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

learning_rate = 0.01
epochs = 1000
m, n = X.shape
weights = np.zeros(n)
bias = 0

for _ in range(epochs):
    model = np.dot(X, weights) + bias
    predictions = 1 / (1 + np.exp(-model))

    dw = (1 / m) * np.dot(X.T, (predictions - y))
    db = (1 / m) * np.sum(predictions - y)

    weights -= learning_rate * dw
    bias -= learning_rate * db

model = np.dot(X, weights) + bias
predictions = 1 / (1 + np.exp(-model))
predictions = [1 if p >= 0.5 else 0 for p in predictions]
print(predictions)