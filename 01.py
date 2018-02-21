import numpy as np
import matplotlib.pyplot as plt

class SLP():

    def __init__(self, d, M):
        """d is the dimension of the input and
        M is the dimension of the output."""

        self.W = np.random.rand(d+1, M) # d+1 one since we add the bias term


    def fit(self, X, T, epochs, learning_rule = 'delta', eta = 0.1, sequential = False):

        assert X.shape[0] == T.shape[0]

        X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis = 1)

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
                change = np.arange(X.shape[0])
                np.random.shuffle(change)
                X = X[change, :]
                T = T[change, :]
                Y = np.dot(X, self.W)
                error = 0.5 * np.einsum('ij, ij', Y - T, Y - T)
                E[epoch] = error
                for x, t in zip(X, T):
                    dW = -eta * x.reshape(1, -1).T.dot((np.dot(x.reshape(1, -1), self.W) - t.reshape(1,-1)))
                    self.W = self.W + dW
                    # todo: maybe all dW will be zero?

            plt.plot(np.arange(epochs), E)
            plt.show()

        else: # batch instead
            for epoch in range(epochs):
                Y = np.dot(X, self.W)
                error = 0.5 * np.einsum('ij, ij', Y - T, Y - T)
                E[epoch] = error
                dW = -eta*X.T.dot((np.dot(X, self.W) - T))
                self.W = self.W + dW

            plt.plot(np.arange(epochs), E)
            plt.show()


    def fit_perceptron(self, X, T, epochs, eta, sequential):
        """Train the weights with the perceptron learning rule"""

        E = np.zeros(epochs)

        if sequential:
            for epoch in range(epochs):

                # shuffle
                change = np.arange(X.shape[0])
                np.random.shuffle(change)
                X = X[change,:]
                T = T[change,:]
                Y = self.activation(np.dot(X, self.W))
                error = 0.5 * np.einsum('ij, ij ', Y - T, Y - T)
                E[epoch] = error
                for x, t in zip(X, T):
                    y = self.activation(np.dot(x.reshape(1, -1), self.W))
                    dW = -eta * x.reshape(1, -1).T.dot(y - t.reshape(1, -1))
                    self.W = self.W + dW

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
                    break
                self.W = self.W + dW

            plt.plot(np.arange(epochs), E)
            plt.show()

    def activation(self, z):

        return np.sign(z)



def data_3_1(mean, sigma, N = 100):


    X = np.random.multivariate_normal(mean, sigma, N)
    return X

N = 50
mean_1 = np.array([1, 1])
mean_2 = np.array([-1, -1])
sigma = np.identity(2)*0.2

X_1 = data_3_1(mean_1, sigma, N)
T_1 = np.ones(N).reshape(N, 1)
X_2 = data_3_1(mean_2, sigma, N)
T_2 = -np.ones(N).reshape(N, 1)

X = np.concatenate((X_1, X_2), axis = 0)
T = np.concatenate((T_1, T_2), axis = 0)

change = np.arange(2*N)
np.random.shuffle(change)
X = X[change,:]
T = T[change,:]

slp=SLP(2,1)
slp.fit(X,T,100,'delta',sequential=False)
#plt.scatter(X[:,0], X[:,1])
#plt.show()