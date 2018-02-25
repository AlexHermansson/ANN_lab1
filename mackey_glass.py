from datasets import mackey_glass
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers


model = Sequential()



time, x = mackey_glass()