from datasets import mackey_glass
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

np.random.seed(100)
x = mackey_glass()

X = np.zeros((1200, 5))
T = np.zeros((1200, 1))
for i, t in enumerate(range(301, 1501)):
    X[i] = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    T[i] = x[t+5]

sigma=0.18

test_size = 200
X_test = X[-test_size:]
X_train = X[:-test_size]
T_test = T[-test_size:]
T_train = T[:-test_size]
T_train_noise = T_train + np.random.normal(0,sigma, T_train.shape)

hidden_first = 30
hidden_second = 4
d = 5
M = 1
epochs = 15000
b_size = X_train.shape[0]
b_size_test=X_test.shape[0]
lambd=0.5
eta=0.0001


model=MLPRegressor((hidden_first,hidden_second),activation="relu",solver="sgd",alpha=lambd,batch_size="auto",
                   learning_rate="constant",learning_rate_init=eta,power_t=0.5, max_iter=epochs,
                   shuffle=True,random_state=None,tol=1e-8,verbose=True,warm_start=False,momentum=0.9,
                   nesterovs_momentum=True,early_stopping=True,validation_fraction=0.4)

model.fit(X_train,T_train_noise)
prediction=model.predict(X_test)
plt.plot(prediction,label="pred")
plt.plot(T_test,label="true")
plt.legend()
plt.show()