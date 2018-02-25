from datasets import mackey_glass
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import regularizers


x = mackey_glass() # 2000 samples, from time = 0 to time = 1999

X = np.zeros((1200, 5))
T = np.zeros((1200, 1))
for i, t in enumerate(range(301, 1501)):
    X[i] = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    T[i] = x[t+5]

test_size = 200
X_test = X[-test_size:]
X_train = X[:-test_size]
T_test = T[-test_size:]
T_train = T[:-test_size]

hidden_first = 10
hidden_second = 5
d = 5
M = 1
epochs = 100
b_size = X_train.shape[0]

# two layer perceptron
model = Sequential()
reg = regularizers.l2(0.01)
model.add(Dense(hidden_first, input_shape=(d,), activation='tanh', kernel_regularizer=reg)) # first hidden layer and also the dimensions of input layer
model.add(Dense(M))
sgd = optimizers.SGD(lr=0.01, momentum=0.9) # add learning rate decay or nesterov momentum?
model.compile(optimizer=sgd, loss='mean_squared_error')
model.fit(X_train, T_train, batch_size=32, epochs=epochs, callbacks=[EarlyStopping()], verbose=2)





# three layer perceptron
#model.add(Dense(hidden_second))


