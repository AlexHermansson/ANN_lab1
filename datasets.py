import numpy as np


def encoder_data(N = 8):
    """Data set with data samples = targets. To train an "autoencoding" network."""
    data = -np.ones([N, N])
    for i in range(N):
        data[i, i] = 1
    targets = np.copy(data)

    return [data, targets]

