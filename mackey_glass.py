from datasets import mackey_glass
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import regularizers

np.random.seed(100)
x = mackey_glass() # 2000 samples, from time = 0 to time = 1999

X = np.zeros((1200, 5))
T = np.zeros((1200, 1))
for i, t in enumerate(range(301, 1501)):
    X[i] = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    T[i] = x[t+5]

sigma=0.03

test_size = 200
X_test = X[-test_size:]
X_train = X[:-test_size]
T_test = T[-test_size:]
T_train = T[:-test_size]
T_train_noise = T_train + np.random.normal(0,sigma, T_train.shape)

hidden_first = 15
hidden_second = 10
d = 5
M = 1
epochs = 5000
b_size = X_train.shape[0]
b_size_test=X_test.shape[0]
lambd=0.000001

# two layer perceptron
model = Sequential()
reg = regularizers.l2(lambd)
model.add(Dense(hidden_first, input_shape=(d,), activation='relu', kernel_regularizer=reg)) # first hidden layer and also the dimensions of input layer
#model.add(Dense(hidden_second, activation='relu', kernel_regularizer=reg))
model.add(Dense(M))
sgd = optimizers.SGD(lr=0.01, momentum=0.9) # add learning rate decay or nesterov momentum?
model.compile(optimizer=sgd, loss='mean_squared_error')
history = model.fit(X_train, T_train_noise, batch_size=b_size, epochs=epochs, verbose=0)#, callbacks=[EarlyStopping(patience = 20)], validation_split=0.4)
#history=model.fit(X_train, T_train, batch_size=b_size, epochs=epochs, verbose=0,validation_split=0.4)

plt.plot(history.history['loss'],c="r")
#plt.plot(history.history['val_loss'],c="b")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.axis([0, epochs,0,1])
#plt.legend(['train', 'val'])
plt.legend('train')
plt.show()

y = model.predict(X_train, verbose=2, batch_size=b_size)
t = np.arange(y.size)

test_MSE = model.evaluate(X_train, T_train, verbose = 0, batch_size=b_size)
print('test error: ', test_MSE)

prediction=model.predict(X_train, verbose=2, batch_size=b_size_test)
t_prediction=np.arange(prediction.size)

plt.plot(t, y, 'r', label = 'Predicted series')
plt.plot(t, T_train, 'b', label = 'Real series')
plt.legend()
#plt.axis([400, 600, -2.5, 3])
plt.show()

'''plt.plot(t_prediction, prediction, 'r', label = 'Predicted series')
plt.plot(t_prediction, T, 'b', label = 'Real series')
plt.legend()
plt.show()'''


