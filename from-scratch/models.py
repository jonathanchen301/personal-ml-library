import numpy as np
from preprocessing import sigmoid

class LogisticRegression:
    def __init__(self, w, b):
        self.w = None
        self.b = None

    def propagate_helper(self, w, b, X, Y, cost_function):
        """
        Implement forward propogation and backward propogation of logistic regression

        Params:
        w -- weights, a numpy array of size (dim, 1)
        b -- bias, a scalar
        X -- data of size (number of features, number of examples)
        Y -- true label vector of size (1, number of examples)
        cost_function -- cost function to use
        """
        m = X.shape[1]
        A = sigmoid(np.dot(w.T, X) + b)
        cost = cost_function(A, Y)

        dw = (np.dot(X,(A-Y).T))/m
        db = (np.sum(A-Y))/m

        grads = {"dw": dw, "db": db}
    
        return grads, cost
    
    def train(self, w, b, X, Y, cost_function, epochs, lr):
        """
        Complete epochs iterations of the training loop, updating the weights and bias
        using gradient descent and returning the updated weights and bias.

        Params:
        w -- weights, a numpy array of size (dim, 1)
        b -- bias, a scalar
        X -- data of size (number of features, number of examples)
        Y -- true label vector of size (1, number of examples)
        cost_function -- cost function to use
        epochs -- number of iterations of the optimization loop
        lr -- learning rate of the gradient descent update rule

        Returns:
        w -- the learned weights
        b -- the learned bias
        """
        costs = []
       
        for i in range(epochs):
            grads, cost = self.propagate_helper(w, b, X, Y, cost_function)
            dw = grads["dw"]
            db = grads["db"]

            w = w - lr * dw
            b = b - lr * db

            if i % 100 == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" %(i, cost))

        self.w = w
        self.b = b
    
    def predict(self, X):
        """
        Predict the labels of a dataset using learned logistic regression parameters

        Params:
        X -- data of size (number of features, number of examples)

        Returns:
        Y_prediction -- a numpy array containing all predictions for the examples in X
        """

        assert self.w is not None and self.b is not None, "Model has not been trained yet"
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = sigmoid(np.dot(self.w.T, X) + self.b)
        for i in range(A.shape[1]):
            Y_prediction[0, i] = A