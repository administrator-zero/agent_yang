# -*- coding: UTF-8 -*-
import numpy as np


class Relu(object):
    @staticmethod
    def forward(weighted_input):
        return np.maximum(0, weighted_input)

    @staticmethod
    def backward(output):
        x = np.maximum(0, output)
        x[x > 0] = 1
        return x


class Identity(object):
    @staticmethod
    def forward(weighted_input):
        return weighted_input

    @staticmethod
    def backward(output):
        return 1


class Sigmoid(object):
    @staticmethod
    def forward(weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    @staticmethod
    def backward(output):
        return output * (1 - output)


class Tanh(object):
    @staticmethod
    def forward(weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    @staticmethod
    def backward(output):
        return 1 - output * output
