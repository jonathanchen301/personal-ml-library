import numpy as np

class Initialization:
    def __init__(self):
        pass

    def initialize_with_zeros(self, dim):

        """
        Initializes weight and bias vectors to zeros

        Params:
        dim -- size of the w vector that we want

        Returns:
        w -- initialized vector of shape dim
        b -- initialized scalar
        """

        w = np.zeros(dim)
        b = 0

        return w, b
