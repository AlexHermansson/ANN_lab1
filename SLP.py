import numpy as np
import matplotlib.pyplot as plt

class SLP():
    """A single layer perceptron class. Can be trained with delta rule or perceptron
    learning rule. With batch learning or sequential learning."""

    def __init__(self, d, M):
        """d is the dimension of the input and
        M is the dimension of the output."""

        self.W = (1- 2*np.random.rand(d+1, M)) # d+1 one since we add the bias term


    def fit(self, X, T, epochs, learning_rule = 'delta', eta = 0.1, sequential = False):

        assert X.shape[0] == T.shape[0]

        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis = 1)
        X, T = self.shuffle(X, T)

        if learning_rule == 'delta':
            self.fit_delta(X, T, epochs, eta, sequential)

        elif learning_rule == 'perceptron':
            self.fit_perceptron(X, T, epochs, eta, sequential)

        else:
            raise ValueError('Not a defined learning rule')


    def fit_delta(self, X, T, epochs, eta, sequential):
        """Train the weights with the delta learning rule"""

        E = np.zeros(epochs)

        if sequential:
            for epoch in range(epochs):

                # shuffle
                X, T = self.shuffle(X, T)
                Y = np.dot(X, self.W)
                error = 0.5 * np.einsum('ij, ij', Y - T, Y - T)
                E[epoch] = error
                for x, t in zip(X, T):
                    dW = -eta * x.reshape(1, -1).T.dot((np.dot(x.reshape(1, -1), self.W) - t.reshape(1,-1)))
                    self.W = self.W + dW

                if epoch % 10 == 0:
                    plot_boundary(self.W)

            plt.plot(np.arange(epochs), E)
            plt.show()

        else: # batch instead
            for epoch in range(epochs):
                Y = np.dot(X, self.W)
                error = 0.5 * np.einsum('ij, ij', Y - T, Y - T)
                E[epoch] = error
                dW = -eta*X.T.dot((np.dot(X, self.W) - T))
                self.W = self.W + dW

                # In every 10th epoch, plot boundary
                if epoch % 10 == 0:
                    plot_boundary(self.W)

            # Plot graph of final error vs epochs

            plt.plot(np.arange(epochs), E)
            plt.show()


    def fit_perceptron(self, X, T, epochs, eta, sequential):
        """Train the weights with the perceptron learning rule"""

        E = np.zeros(epochs)

        if sequential:
            for epoch in range(epochs):

                # shuffle
                X, T = self.shuffle(X, T)
                Y = self.activation(np.dot(X, self.W))
                error = 0.5 * np.einsum('ij, ij ', Y - T, Y - T)
                E[epoch] = error
                for x, t in zip(X, T):
                    y = self.activation(np.dot(x.reshape(1, -1), self.W))
                    dW = -eta * x.reshape(1, -1).T.dot(y - t.reshape(1, -1))
                    self.W = self.W + dW

                # In every 10th epoch, plot boundary
                if epoch % 10 == 0:
                    plot_boundary(self.W)

            # Plot graph of final error vs epochs
            plt.plot(np.arange(epochs), E)
            plt.show()

        else: # batch
            for epoch in range(epochs):
                Y = self.activation(np.dot(X, self.W))
                error = 0.5*np.einsum('ij, ij ', Y-T, Y-T)
                E[epoch] = error
                dW =  -eta*X.T.dot(Y - T)
                if dW.all() == 0:
                    print('No update, break.')
                    #break
                self.W = self.W + dW

                # In every 10th epoch, plot boundary
                if epoch % 10 == 0:
                    plot_boundary(self.W)

            # Plot graph of final error vs epochs
            plt.plot(np.arange(epochs), E)
            plt.show()

    def activation(self, z):
        """A sign activation function."""

        return np.sign(z)

    def shuffle(self, X, T):
        """A small function to shuffle data."""
        change = np.arange(X.shape[0])
        np.random.shuffle(change)
        return X[change, :], T[change, :]

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

epochs=100
d=2
M=1
N = 100


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

slp=SLP(d, M)

slp.fit(X, T, epochs, 'delta', sequential = True, eta = 0.001)

