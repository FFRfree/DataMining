import sklearn
from mydata import get_basic_information_and_y
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

tf.random.set_seed(666)
EPOCHS = 300
LEARNING_RATE = 5e-4
BATCH_SIZE = 20

y_label = 'MntFruits'
x, y = get_basic_information_and_y(y_label)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=321)

# 搞成一列
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

model = keras.Sequential([
    keras.layers.LayerNormalization(axis=0, input_shape=(18,)),
    keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2'),
    keras.layers.Dense(1)
])

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['mean_squared_error'],
              )

model.summary()

