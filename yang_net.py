# -*- coding: UTF-8 -*-

from nn.activators import Sigmoid, Identity, Relu
from nn.conv import Conv
from nn.max_pooling import MaxPooling
from nn.full_connected import FullConnected


class YangNet(object):
    """
    YangNet neural Net
    """
    def __init__(self):
        """
        Constructor
        """
        self.layers = []
        c1 = Conv(28, 28, 1, 5, 5, 32, 0, 1, Relu())
        self.layers.append(c1)
        s2 = MaxPooling(24, 24, 32, 2, 2, 2)
        self.layers.append(s2)
        c3 = Conv(12, 12, 32, 5, 5, 64, 0, 1, Relu())
        self.layers.append(c3)
        s4 = MaxPooling(8, 8, 64, 2, 2, 2)
        self.layers.append(s4)
        fc5 = FullConnected(1024, 10, Sigmoid(), 0.1)
        self.layers.append(fc5)

        pre_activator = Identity
        for layer in self.layers:
            layer.set_pre_activator(pre_activator)
            pre_activator = layer.activator

    def predict(self, sample):
        """
        Get predict result from network.
        sample: The data to be predicted.
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        Train the net.
        labels: labels of sample.
        data_set: the data set of sample.
        rate: learning rate.
        epoch: The epoch of train.
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label, rate)

    def calc_gradient(self, label, rate):
        layer = self.layers[-1]
        pre_delta = layer.activator.backward(layer.output) * (layer.output - label)
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
