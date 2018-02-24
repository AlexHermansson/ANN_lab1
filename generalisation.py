import numpy as np
from datasets import gauss_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TLP():
    """Two layer perceptron class. d is the dimension of input,
    M is the dimension of output and h is the number of hidden nodes."""

    def __init__(self, d, M, nodes, regression = False):

        self.W = (2*np.random.rand(d+1, nodes) - 1)
        self.V = (2*np.random.rand(nodes+1, M) - 1)
        self.Theta = np.zeros([d+1, nodes])
        self.Psi = np.zeros([nodes+1, M])
        self.regression = regression

    def forward(self, X):

        H_star = np.dot(X, self.W)
        H = self.activation(H_star)
        H = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), H), axis = 1)
        O_star = np.dot(H, self.V)

        if self.regression:
            O = O_star
        else:
            O = self.activation(O_star)

        return H, H_star, O, O_star


    def backward(self, T, O, H):

        if self.regression:
            delta_out = (O - T)
        else:
            delta_out = (O - T)*self.d_activation(O)
        delta_hidden = np.dot(delta_out, self.V[1:].T)*self.d_activation(H[:,1:])
        return delta_out, delta_hidden

    def fit(self, X, T, X_test, T_test, epochs, eta = 0.01, alpha = None, verbose = False, plot = True):
        """Backprop algorithm."""

        E = np.zeros(epochs)
        E_test=np.zeros(epochs)
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        for epoch in range(epochs):

            H, H_star, O, O_star = self.forward(X)
            _,O_test=self.predict(X_test)
            delta_out, delta_hidden = self.backward(T, O, H)

            E[epoch] = 0.5*np.einsum('ij, ij', O-T, O-T)
            E[epoch] = 0.5*np.einsum('ij, ij', O_test-T_test, O_test-T_test)
            if verbose and epoch%500==0:
                print('error for iteration %i: %f' %(epoch ,E[epoch]))
                print('TEST error for iteration %i: %f' % (epoch, E_test[epoch]))

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
            plt.plot(np.arange(epochs), E,'g')
            plt.plot(np.arange(epochs), E_test, 'r')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.axis([0, epochs, 0, 50])
            plt.show()

    def predict(self, X):

        X = np.concatenate((np.ones(N).reshape(-1, 1), X), axis = 1)
        H, _, O, _ = self.forward(X)

        return H, O

    def activation(self, z):

        return 2/(1 + np.exp(-z)) - 1

    def d_activation(self, a):
        """a is an activation from activation()"""

        return 0.5*(1 + a)*(1 - a)

def shuffle(X, T):
    """A small function to shuffle data."""
    change = np.arange(X.shape[0])
    np.random.shuffle(change)
    return X[change, :], T[change, :]


n=20
N=n*n
test=N//4*3
X, T = gauss_data(n)
#X,T=shuffle(X,T)

X_train=shuffle(X,T)[0][:test]
T_train=shuffle(X,T)[1][:test]

M = T.shape[1]
d = X.shape[1]
hidden_nodes = 5
epochs = 5000
#eta = 1/N
eta = 0.001
print(eta)


# Train the model
tlp = TLP(d, M, hidden_nodes, regression = True)
tlp.fit(X_train, T_train, X,T, epochs, eta, plot = True)

# Show the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X[:,0].reshape(n,n), X[:,1].reshape(n,n), T.reshape(n,n), cmap = 'inferno')
plt.title('Original Gauss function')
plt.show()

# Plot the approximated function
_, Y = tlp.predict(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X[:,0].reshape(n,n), X[:,1].reshape(n,n), Y.reshape(n,n), cmap = 'inferno')
plt.title('Approximated function, %i hidden nodes' %hidden_nodes)
#plt.savefig('Approx, %i hidden' %hidden_nodes)
plt.show()
