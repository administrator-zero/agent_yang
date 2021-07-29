import numpy as np
from nn.base import Base


class FullConnected(Base):
    """
    Full connected layer of BP
    """
    def __init__(self, input_size, output_size, activator, r):
        """
        Constructor
        input_size: The input size of this layer.
        output_size: The output size of the layer.
        activator: The active function of this layer.
        """
        Base.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # The matrix of weights in this layer.
        self.W = np.random.uniform(-r, r, (output_size, input_size))
        # The vector of bias in this layer.
        self.b = np.random.uniform(-r, r, (output_size, 1))
        # The out result of this layer.
        self.output = np.zeros((output_size, 1))
        # The real input matrix of this layer, maybe is 3 dim.
        self.input = None
        # Convert to input vector.
        self.input_fc = None
        # The gradient of all weight.
        self.W_grad = None
        # The gradient of all bias.
        self.b_grad = None

    def forward(self, input_data):
        """
        The forward propagation of BP.
        input_data: The data of input that the size must equal input_size.
        """
        # formula 2
        # flatten dim for supporting more than 3 dim.
        self.input = input_data
        arr = input_data.flatten()
        self.input_fc = arr.reshape(len(arr), 1)
        self.output = self.activator.forward(np.dot(self.W, self.input_fc) + self.b)

    def backward(self, input_delta, learning_rate):
        """
        Calculate the gradient with Back Propagation.
        input_delta: Current layer input delta from down layer.
        """
        # formula 8
        self.pre_delta = self.pre_activator.backward(self.input_fc) * np.dot(self.W.T, input_delta)
        # restore dim.
        in_shape = np.array(self.input).shape
        self.pre_delta = self.pre_delta.reshape(in_shape)

        self.W_grad = np.dot(input_delta, self.input_fc.T)
        self.b_grad = input_delta

        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad

    def dump(self):
        """
        print W and b information.
        :return:
        """
        print('W: %s\nb:%s' % (self.W, self.b))
