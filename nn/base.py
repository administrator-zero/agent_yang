class Base(object):
    def __init__(self):
        self.output = None
        self.activator = None

        # The error item propagation to previous layer.
        self.pre_delta = None
        # The previous activator.
        self.pre_activator = None

    def set_pre_activator(self, pre_activator):
        self.pre_activator = pre_activator
