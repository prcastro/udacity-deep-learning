"""
Simple implementation of a TensorFlow-like library
"""

from collections import defaultdict

import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplementedError('forward pass not implemented for this node')

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum([node.value for node in self.inbound_nodes])


class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for node in self.inbound_nodes:
            self.value *= node.value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        input_values = self.inbound_nodes[0]
        weights = self.inbound_nodes[1]
        bias = self.inbound_nodes[2]

        self.value = np.dot(input_values.value, weights.value) + bias.value

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        inputs = self.inbound_nodes[0]
        weights = self.inbound_nodes[1]
        bias = self.inbound_nodes[2]

        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[inputs] += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(inputs.value.T, grad_cost)
            self.gradients[bias] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self):
        inputs = self.inbound_nodes[0]
        self.value = self._sigmoid(inputs.value)

    def backward(self):
        inputs = self.inbound_nodes[0]
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            # NOTE: Since sigmoid is a many-to-many function (one that receives a vector as input
            # and also outputs a vector), the gradient of it with respect to the input should be a
            # vector of the same dimension of the output
            self.gradients[inputs] += self.value * (1 - self.value) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of
        # shape (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0]
        a = self.inbound_nodes[1]
        self.diff = y.value.reshape(-1, 1) - a.value.reshape(-1, 1)
        self.value = np.mean(np.square(self.diff))

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        y = self.inbound_nodes[0]
        a = self.inbound_nodes[1]
        num_examples = len(y.value)

        self.gradients = {}
        self.gradients[y] = (2 / num_examples) * self.diff
        self.gradients[a] = (-2 / num_examples) * self.diff


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is
    the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = defaultdict(lambda: {'in': set(), 'out': set()})
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        for m in n.outbound_nodes:
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    for node in trainables:
        node.value -= learning_rate * node.gradients[node]
