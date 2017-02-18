'''
Example of a neural network using MiniFlow
'''

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import resample

import miniflow

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = miniflow.Input(), miniflow.Input()
W1, b1 = miniflow.Input(), miniflow.Input()
W2, b2 = miniflow.Input(), miniflow.Input()

l1 = miniflow.Linear(X, W1, b1)
s1 = miniflow.Sigmoid(l1)
l2 = miniflow.Linear(s1, W2, b2)
cost = miniflow.MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 100000
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

graph = miniflow.topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        X.value = X_batch
        y.value = y_batch

        miniflow.forward_and_backward(graph)
        miniflow.sgd_update(trainables)

        loss += graph[-1].value

    print("\rEpoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch), end="")
print()
