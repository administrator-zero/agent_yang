import numpy as np
from nn.filter import Filter
from nn.conv_util import ConvUtil
from nn.base import Base


class Conv(Base):
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, filter_number,
                 zero_padding, stride, activator):
        Base.__init__(self)
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = int(self.calculate_output_size(self.input_width, filter_width, zero_padding, stride))
        self.output_height = int(self.calculate_output_size(self.input_height, filter_height, zero_padding, stride))
        self.output = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator
        self.input_array = None
        self.padded_input_array = None
        self.expanded_array = None

    def forward(self, input_array):
        """
        Calculate out of the convolution layer that save to self.output
        """
        self.input_array = input_array
        self.padded_input_array = self.padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            fi = self.filters[f]
            self.cross_correlation(self.padded_input_array, fi.weights, self.output[f], self.stride, fi.bias)
        self.output = np.array(self.activator.forward(self.output))

    def backward(self, sensitivity_array, learning_rate):
        """
        Calculate the error item passed to the previous layer and calculate the gradient for each weight.
        The error of the previous layer is stored in self.pre_delta
        The gradient is saved in the weights_grad of the Filter
        :param sensitivity_array: The sensitivity map of this layer from down layer.
        :param learning_rate: Current learning rate.
        :return: The active function of previous layer.
        """
        # Restore sensitivity map to stride equal 1.
        self.expanded_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_width = self.expanded_array.shape[2]
        # W2=(W1-F+2P)/S+1 => P=(W1+F-1-W1)/2 W1=expanded_width W2=input_width F=filter_width
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = self.padding(self.expanded_array, zp)
        # Save previous errors items init by zeros.
        self.pre_delta = self.create_delta_array()
        # Calculate an array of error items corresponding to one filter.
        delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            fi = self.filters[f]
            # Turn 180 degree of all filter weights.
            flipped_weights = np.array([np.rot90(i, 2) for i in fi.weights])
            self.back_cross_correlation(padded_array[f], flipped_weights, delta_array)
            # All chanel error items for previous layer.
            self.pre_delta += delta_array

            # Set gradient of every Weight.
            self.back_cross_correlation(self.padded_input_array, self.expanded_array[f], fi.weights_grad)
            # Set gradient of the bias.
            fi.bias_grad = self.expanded_array[f].sum()
            # Update weights according to stochastic gradient descent.
            fi.update(learning_rate)

        # The output of the previous layer as input of the partial derivative of the activation function.
        self.pre_delta *= self.activator.backward(self.input_array)

    def expand_sensitivity_map(self, sensitivity):
        """
        Expand sensitivity map with stride equal 1 and use zero fill interval.
        :param sensitivity: Original sensitivity.
        :return:
        """
        depth = sensitivity.shape[0]
        # expended sensitivity map size
        # W2=(W1-F+2P)/S+1 where S is 1.
        z1 = 2 * self.zero_padding + 1
        expanded_width = (self.input_width - self.filter_width + z1)
        expanded_height = (self.input_height - self.filter_height + z1)
        # create new sensitivity_map.
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # Get error items from original sensitivity.
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1

    @staticmethod
    def cross_correlation(input_array, kernel_array, output, stride, bias):
        """
        Calculate convolution that support 2 dim and 3 dim.
        """
        output_width = output.shape[1]
        output_height = output.shape[0]
        kernel_width = kernel_array.shape[-1]
        kernel_height = kernel_array.shape[-2]
        for i in range(output_height):
            for j in range(output_width):
                patch = ConvUtil.get_patch(input_array, i, j, kernel_width, kernel_height, stride)
                net = patch * kernel_array
                output[i][j] = net.sum() + bias

    @staticmethod
    def back_cross_correlation(input_array, kernel_array, output):
        """
        Calculate back convolution.
        """
        output_width = output.shape[-1]
        output_height = output.shape[-2]
        kernel_width = kernel_array.shape[-1]
        kernel_height = kernel_array.shape[-2]
        for i in range(output_height):
            for j in range(output_width):
                patch = ConvUtil.get_patch(input_array, i, j, kernel_width, kernel_height, 1)
                net = patch * kernel_array
                output[:, i, j] = np.sum(net, (-2, -1))

    @staticmethod
    def padding(input_array, zp):
        """
        Add zero padding to the array, automatically adapt the input to 2D and 3D.
        :param input_array:
        :param zp: The zero padding expend of input_array.
        :return:
        """
        zp = int(zp)
        if zp == 0:
            return input_array
        else:
            if input_array.ndim == 3:
                input_width = int(input_array.shape[2])
                input_height = int(input_array.shape[1])
                input_depth = input_array.shape[0]
                padded_array = np.zeros((
                    input_depth,
                    input_height + 2 * zp,
                    input_width + 2 * zp))
                padded_array[:, zp: zp + input_height, zp: zp + input_width] = input_array
                return padded_array
            elif input_array.ndim == 2:
                input_width = input_array.shape[1]
                input_height = input_array.shape[0]
                padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
                padded_array[zp: zp + input_height, zp: zp + input_width] = input_array
                return padded_array
