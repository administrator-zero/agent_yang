import numpy as np
from nn.conv_util import ConvUtil
from nn.activators import Identity
from nn.base import Base


class MaxPooling(Base):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, stride):
        Base.__init__(self)
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output = np.zeros((self.channel_number, self.output_height, self.output_width))
        self.input_array = None
        self.learning_rate = None
        self.activator = Identity

    def forward(self, input_array):
        self.input_array = input_array
        for i in range(self.output_height):
            for j in range(self.output_width):
                patch = ConvUtil.get_patch(input_array, i, j, self.filter_width, self.filter_height, self.stride)
                self.output[:, i, j] = np.max(patch, (-2, -1))

    def backward(self, sensitivity, learning_rate):
        self.learning_rate = learning_rate
        self.pre_delta = np.zeros(self.input_array.shape)
        for i in range(self.output_height):
            for j in range(self.output_width):
                patch = ConvUtil.get_patch(self.input_array, i, j, self.filter_width, self.filter_height, self.stride)
                for d in range(self.channel_number):
                    k, l = self.get_max_index(patch[d])
                    self.pre_delta[d, i * self.stride + k, j * self.stride + l] = sensitivity[d, i, j]
        self.pre_delta *= self.pre_activator.backward(self.input_array)

    @staticmethod
    def get_max_index(array):
        """
        Get index of max value in 2D array.
        """
        pos = int(np.argmax(array))
        max_i, max_j = divmod(pos, array.shape[1])
        return max_i, max_j
