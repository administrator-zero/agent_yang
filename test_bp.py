import numpy as np
from functools import reduce
from nn.full_connected import FullConnected
from nn.activators import Sigmoid, Identity


class Network(object):
    """
    Neural network.
    """

    def __init__(self, layers):
        """
        Init the neural network.
        """
        self.layers = []
        length = len(layers)
        for i in range(length - 1):
            self.layers.append(
                FullConnected(
                    layers[i], layers[i + 1],
                    Sigmoid()
                )
            )

        pre_activator = Identity
        for layer in self.layers:
            layer.set_pre_activator(pre_activator)
            pre_activator = layer.activator

    def predict(self, sample):
        """
        Predict result from input.
        sample: input data.
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        Train the neural network
        labels: The label of training data.
        data_set: The training data.
        rate: learn rate.
        epoch: traing epoch.
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label, rate)

    def calc_gradient(self, label, rate):
        # Out layer pre_delta.
        pre_delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)

        # Hider layer pre_delta.
        for layer in self.layers[::-1]:
            layer.backward(pre_delta, rate)
            pre_delta = layer.pre_delta

    def dump(self):
        for layer in self.layers:
            layer.dump()

    @staticmethod
    def loss(output, label):
        diff = (output - label)
        return 0.5 * (diff * diff).sum()

    def gradient_check(self, sample_feature, sample_label):
        """
        Check gradient of BP.
        :param sample_feature: The feature of sample.
        :param sample_label: The label of sample.
        :return:
        """
        # Get gradient by network back propagation.
        self.predict(sample_feature)
        self.calc_gradient(sample_label, 0)

        # check gradient.
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected:%.4e=====actural:%.4e' % (
                        i, j, expect_grad, -fc.W_grad[i, j]))


class Normalizer(object):
    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        data = [0.9 if number & m else 0.1 for m in self.mask]
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = [1 if i > 0.5 else 0 for i in vec[:, 0]]
        for i in range(len(self.mask)):
            binary[i] *= self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def transpose(args):
    return [[np.array(line).reshape(len(line), 1) for line in arg] for arg in args]


def get_train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def gradient_check():
    """
    Check gradient by define.
    """
    labels, data_set = transpose(get_train_data_set())
    net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net


if __name__ == '__main__':
    gradient_check()
