import numpy as np
from datasets import encoder_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TLP():
    """Two layer perceptron class. d is the dimension of input,
    M is the dimension of output and h is the number of hidden nodes."""

    def __init__(self, d, M, nodes):

        self.W = np.random.rand(d+1, nodes)
        self.V = np.random.rand(nodes+1, M)
        self.Theta = np.zeros([d+1, nodes])
        self.Psi = np.zeros([nodes+1, M])

    def forward(self, X):

        H_star = np.dot(X, self.W)
        H = self.activation(H_star)
        H = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), H), axis = 1)
        O_star = np.dot(H, self.V)
        O = self.activation(O_star)

        return H, H_star, O, O_star


    def backward(self, T, O,H):


        delta_out = (O - T)*self.d_activation(O)
        delta_hidden = np.dot(delta_out, self.V[1:].T)*self.d_activation(H[:,1:])
        return delta_out, delta_hidden

    def fit(self, X, T, epochs, eta = 0.01, alpha = None, verbose = False, plot = True):
        """Backprop algorithm."""

        E = np.zeros(epochs)
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        for epoch in range(epochs):

            H, H_star, O, O_star = self.forward(X)
            delta_out, delta_hidden = self.backward(T, O, H)

            E[epoch] = 0.5*np.einsum('ij, ij', O-T, O-T)
            if verbose and epoch%500==0:
                print('error for iteration %i: %f' %(epoch ,E[epoch]))

            # if momentum
            if alpha:
                self.Theta = alpha*self.Theta - (1 - alpha)*np.dot(X.T, delta_hidden)
                self.Psi = alpha*self.Psi - (1 - alpha)*np.dot(H.T, delta_out)
                self.W = self.W + eta*self.Theta
                self.V = self.V + eta*self.Psi
            # if no momentum
            else:
                self.W = self.W - eta*np.dot(X.T, delta_hidden)
                self.V = self.V - eta*np.dot(H.T, delta_out)

        if plot:
            plt.plot(np.arange(epochs), E)
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.show()

    def predict(self, X):

        X = np.concatenate((np.ones(N).reshape(-1, 1), X), axis = 1)
        H, H_star, O, O_star = self.forward(X)

        return H, O

    def activation(self, z):

        return 2/(1 + np.exp(-z)) - 1

    def d_activation(self, a):
        """a is an activation from activation()"""
        # todo: is this a good idea?
        return 0.5*(1 + a)*(1 - a)


N = 8
X, T = encoder_data(N)

M, d = T.shape
hidden_nodes = 3
epochs = 15000
eta = 0.01

tlp = TLP(d, M, hidden_nodes)
tlp.fit(X, T, epochs, eta, plot = False)

H, Y = tlp.predict(X)

H = H[:, 1:] # remove bias column

print('output: ')
print(np.round(Y))
H_sign = np.sign(H)
print('3D representation: ')
print(H_sign)
print('weight matrix: ')
print(tlp.W)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(H_sign[:,0], H_sign[:,1], H_sign[:,2], c = 'r')
plt.savefig()
plt.show()
