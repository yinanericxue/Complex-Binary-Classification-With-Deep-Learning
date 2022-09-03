import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset

X, Y = load_planar_dataset()
sampleSize= X.shape[1]
learning_rate = 0.1

# Initiate the parameters: (2 + 1) x 4 + (4 + 1) = 17
W1 = np.random.randn(4, X.shape[0]) * 0.01
b1 = np.zeros(shape=(4, 1))
W2 = np.random.randn(Y.shape[0], 4) * 0.01
b2 = np.zeros(shape=(Y.shape[0], 1))

# process of calculation and storage of intermediate variables
# W1 = 4 x 2, W2 = 1 x 4, b1 = 4 x 1, b2 = 1 x 1
def forward_propagation(X, W1, W2, b1, b2):
    Z1 = np.dot(W1, X) + b1 # Z1 = 4 x 400
    A1 = np.tanh(Z1) # A1 = 4 x 400
    Z2 = np.dot(W2, A1) + b2 # Z2 = 1 x 400
    A2 = sigmoid(Z2) # A2 = 1 x 400
    return A2, {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}

# cross-entropy is used in this loss function
def compute_cost(A2, Y, m):
    return np.squeeze(- np.sum(np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))) / m)

# fine-tuning the weights based on the error rate
def back_propagation(W1, b1, W2, b2, cache, m):
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, W2, b1, b2

# 100,000 interations performed
for i in range(0, 100000):
    A2, parameters = forward_propagation(X, W1, W2, b1, b2)
    cost = compute_cost(A2, Y, sampleSize)
    W1, W2, b1, b2 = back_propagation(W1, b1, W2, b2, parameters, sampleSize)
    if cost and i % 10000 == 0:
        print("Cost after iteration % i: % f" % (i, cost))

finalPrediction = A2.tolist()

# rounding each result value to 0 and 1
for index in range(400):
    if finalPrediction[0][index] >= 0.5:
        finalPrediction[0][index] = 1
    else:
        finalPrediction[0][index] = 0

plt.scatter(X[0, :], X[1, :], c=finalPrediction, s=40, cmap=plt.cm.Spectral);
plt.show()
