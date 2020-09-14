# -*- coding: UTF-8 -*-
from nn.activators import Sigmoid, Identity, Relu
from nn.full_connected import FullConnected


class YangNet(object):
    def __init__(self):
        """
        Init neural network.
        """
        self.layers = []
        self.learn_rate = 0.01
        layer = FullConnected(784, 100, Relu())
        self.layers.append(layer)
        layer = FullConnected(100, 10, Sigmoid())
        self.layers.append(layer)

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
