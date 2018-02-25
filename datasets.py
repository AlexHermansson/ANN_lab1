import numpy as np

def encoder_data(N = 8):
    """Data set with data samples = targets. To train an "autoencoding" network."""
    data = -np.ones([N, N])
    for i in range(N):
        data[i, i] = 1
    targets = np.copy(data)

    return data, targets

def gauss_data(N=100):
    """generating data using Gauss function"""
    x=np.linspace(-5,5,N)
    y=np.linspace(-5,5,N)
    X,Y=np.meshgrid(x,y)
    Z=np.array(np.exp(-(X ** 2 + Y ** 2) / 10) - 0.5)

    data=np.concatenate((X.reshape(X.size,1),Y.reshape(Y.size,1)),axis=1)
    targets= Z.reshape(Z.size,1)

    return data, targets


def mackey_glass():
    # parameters
    beta = 0.2; gamma = 0.1; n = 10; tau = 25
    time = np.arange(2000)

    x = np.zeros(time.size)
    x[0] = 1.5

    for t in range(time.size - 1):

        if t < tau:
            x[t+1] = x[t] - gamma*x[t]

        else:
            x[t+1] = x[t] + beta*x[t - tau]/(1 + x[t - tau]**n) - gamma*x[t]

    return x