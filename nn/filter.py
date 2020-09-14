import numpy as np


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-1, 1e-1, (depth, height, width))
        self.bias = np.random.uniform(-1e-1, 1e-1, 1)[0]
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
