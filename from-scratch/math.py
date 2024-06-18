import numpy as np

class ActivationFunctions:
    def __init__(self):
        pass

    def sigmoid(self, x):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        return 1/(1+np.exp(-x))
