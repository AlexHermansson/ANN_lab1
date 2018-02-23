import numpy as np
import matplotlib.pyplot as plt

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

    def fit(self, X, T, epochs, eta = 0.01, alpha = None, verbose = False):
        """Backprop algorithm."""

        E = np.zeros(epochs)
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        for epoch in range(epochs):
            H, H_star, O, O_star = self.forward(X)
            delta_out, delta_hidden = self.backward(T, O, H)

            E[epoch] = 0.5*np.einsum('ij, ij', O-T, O-T)
            if verbose and epoch%100==0:
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

        plt.plot(np.arange(epochs), E)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.show()

    def activation(self, z):

        return 2/(1 + np.exp(-z)) - 1

    def d_activation(self, a):
        """a is an activation from activation()"""
        # todo: is this a good idea?
        return 0.5*(1 + a)*(1 - a)


def plot_boundary(w):
    #
    plt.scatter(X_1[:,0], X_1[:,1], c="r")
    plt.scatter(X_2[:,0], X_2[:,1], c="b")
    b=w[0]
    w=w[1:]
    classify = lambda x: np.dot(w.T, x) + b
    xgrid = np.linspace(-4, 4)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[classify(np.array([x, y])) for x in xgrid] for y in ygrid]).reshape(xgrid.shape[0], -1)


    plt.contour(xgrid, ygrid, grid, 0)
    plt.axis([-4, 4, -4, 4])
    plt.show()


def data_3_1(mean, sigma, N = 100):

    X = np.random.multivariate_normal(mean, sigma, N)
    return X

epochs=1000
d=2
M=1
N = 100
hidden_nodes=10


mean_1 = np.array([0.5, 0.5])
mean_2 = np.array([-0.5, -0.5])

# linearly separable data
#sigma = np.identity(2)*0.2

# non-linearly seperable data
sigma = np.identity(2)*0.8

X_1 = data_3_1(mean_1, sigma, N)
T_1 = np.ones(N).reshape(N, 1)
X_2 = data_3_1(mean_2, sigma, N)
T_2 = -np.ones(N).reshape(N, 1)

X = np.concatenate((X_1, X_2), axis = 0)
T = np.concatenate((T_1, T_2), axis = 0)

tlp=TLP(d,M,hidden_nodes)
tlp.fit(X,T,epochs,0.001,0.9,True)