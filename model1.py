import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from preprocess_data_script import preprocess_data
import tensorflow.keras as keras
import tensorflow as tf
tf.random.set_seed(666)
EPOCHS = 500
LEARNING_RATE = 5e-5
BATCH_SIZE = 20

x, y = preprocess_data()

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x.to_numpy())
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1,1))


model = keras.Sequential([
        keras.layers.Dense(36, activation='relu', input_shape=(9,), kernel_regularizer='l1_l2'),
        keras.layers.BatchNormalization(1,),
        keras.layers.Dense(144, activation='relu', kernel_regularizer='l1_l2'),
        keras.layers.BatchNormalization(1,),
        keras.layers.Dense(72, activation='relu', kernel_regularizer='l1_l2'),
        keras.layers.BatchNormalization(1,),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['mean_squared_error'],
              )
model.summary()

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
]
history=model.fit(x_scaled,y_scaled,epochs=EPOCHS,
                  validation_split=0.2,
                  batch_size=BATCH_SIZE,
                  callbacks=my_callbacks)
# y_scaler.inverse_transform(model.predict(x_scaled))[32:47]

from finetune_tools import visualize_history
visualize_history(history)


import time
val_mse = model.history.history['val_mean_squared_error'][-10]
filename = time.strftime("%d_%H_%M_")+str(val_mse)+'.h5'
# with open('trained_models\\'+filename , 'wb') as file_pi:
#     pickle.dump(trained_models, file_pi)
model.save('trained_models\\'+filename)