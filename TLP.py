import numpy as np
import matplotlib.pyplot as plt

class TLP():
    """Two layer perceptron class. d is the dimension of input,
    M is the dimension of output and h is the number of hidden nodes."""


    def __init__(self, d, M, nodes):

        self.W = np.random.rand([d+1, nodes])
        self.V = np.random.rand([nodes+1, M])
        self.Theta = np.zeros([d+1, nodes])
        self.Psi = np.zeros([nodes+1, M])

    def forward(self, X):

        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis = 1)
        H_star = np.dot(X, self.W)
        H = self.activation(H_star)
        H = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), H), axis = 1)
        O_star = np.dot(H, self.V)
        O = self.activation(O_star)


        return H, H_star, O, O_star


    def backward(self, T, H_star, O, O_star ):


        delta_out = (O - T)*self.d_activation(O_star)
        delta_hidden = np.dot(delta_out, V[:, 1:].T)*self.d_activation(H_star)
        return delta_out, delta_hidden

    def fit(self, X, T, epochs, eta = 0.01, alpha = None, verbose = False):
        """Backprop algorithm."""

        E = np.zeros(epochs)
        for epoch in range(epochs):
        	H, H_star, O, O_star = self.forward(X)
        	delta_out, delta_hidden = self.backward(T, H_star, O, O_star)

        	E[epoch] = 0.5*np.einsum('ij, ij', O-T, O-T)
        	if verbose:
        		print('error for iteration %i: %f' %epoch, %E[epoch])

        	# if momentum
        	if alpha:
        		self.Theta = alpha*self.Theta - (1 - alpha)*np.dot(X.T, delta_hidden)
        		self.Psi = alpha*self.Psi - (1 - alpha)*np.do(H.T, delta_out)
        		self.W = self.W + eta*self.Theta
        		self.V = self.V + eta*self.Psi
        	# if no momentum
        	else:
        		self.W = self.W - eta*np.dot(X.T, delta_hidden)
        		self.V = self.V - eta*np.dot(H.T, delta_out)

        plt.plot(np.arange(epochs), E)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')

    def activation(self, z):

        return 2/(1 + np.exp(-z)) - 1

    def d_activation(self, a):
        """a is an activation from activation()"""
        # todo: is this a good idea?
        return 0.5*(1 + a)*(1 - a)



