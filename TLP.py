import numpy as np
import matplotlib.pyplot as plt

class TLP():
    """Two layer perceptron class. d is the dimension of input,
    M is the dimension of output and h is the number of hidden nodes."""


    def __init__(self, d, M, nodes):

        self.W = np.random.rand([d+1, nodes])
        self.V = np.random.rand([nodes+1, M])

        self.H = None
        self.H_star = None
        self.O = None
        self.O_star = None


    def forward(self, X):

        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis = 1)
        self.H_star = np.dot(X, self.W)
        H = self.activation(self.H_star)
        self.H = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), H), axis = 1)
        self.O_star = np.dot(H, self.V)
        self.O = self.activation(self.O_star)


    def backward(self, T):


        delta_out = (self.O - T)*self.d_activation(self.O_star)
        delta_hidden = np.dot(delta_out, V[:, 1:].T)*self.d_activation(self.H_star)

        # todo: here or in fit?
        dW = - eta

        pass

    def fit(self, X, T, epochs, eta):
        """Backprop algorithm."""




    def activation(self, z):

        return 2/(1 + np.exp(-z)) - 1

    def d_activation(self, a):
        """a is an activation from activation()"""
        # todo: is this a good idea?
        return 0.5*(1 + a)*(1 - a)