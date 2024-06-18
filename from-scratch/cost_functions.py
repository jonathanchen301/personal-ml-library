import numpy as np

class CostFunctions:
    def __init__():
        pass

    def cross_entropy_loss(self, A, Y):
        """
        Compute the cross-entropy cost given in equation

        Arguments:
        A -- predicted vector of (dim)
        Y -- true label vector of size (dim)

        Returns:
        cost -- cross-entropy cost
        """
        m = A.shape[1]

        return np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m
