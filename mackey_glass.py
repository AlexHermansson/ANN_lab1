from datasets import mackey_glass
import numpy as np
import matplotlib.pyplot as plt

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

hidden_first = 20
hidden_second = 10
d = 5
M = 1
epochs = 500
b_size = X_train.shape[0]

# two layer perceptron
model = Sequential()
reg = regularizers.l2(0.01)
model.add(Dense(hidden_first, input_shape=(d,), activation='relu', kernel_regularizer=reg)) # first hidden layer and also the dimensions of input layer
model.add(Dense(hidden_second, activation='relu', kernel_regularizer=reg))
model.add(Dense(M))
sgd = optimizers.SGD(lr=0.01, momentum=0.9) # add learning rate decay or nesterov momentum?
model.compile(optimizer=sgd, loss='mean_squared_error')
history = model.fit(X_train, T_train, batch_size=32, epochs=epochs, callbacks=[EarlyStopping(patience = 20)], verbose=0, validation_split=0.4)
#history=model.fit(X_train, T_train, batch_size=b_size, epochs=epochs, verbose=0,validation_split=0.4)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.axis([0, epochs, 0, 0.1])
plt.legend(['train', 'val'])
plt.show()

y = model.predict(X_test, verbose=2, batch_size=32)
t = np.arange(y.size)

test_MSE = model.evaluate(X_test, T_test, verbose = 0, batch_size=32)
print('test error: ', test_MSE)

plt.plot(t, y, 'r', label = 'Predicted series')
plt.plot(t, T_test, 'b', label = 'Real series')
plt.legend()
plt.show()



