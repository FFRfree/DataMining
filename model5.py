import sklearn
from mydata import get_basic_information_and_y
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import pickle
tf.random.set_seed(666)
EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 20

y_label = 'MntGoldProds'
x, y = get_basic_information_and_y(y_label)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=321)

# 搞成一列
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print(f'x_train shape:{x_train.shape}')
print(f'x_test shape:{x_test.shape}')
print(f'y_train shape:{y_train.shape}')
print(f'y_test shape:{y_test.shape}')

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x_train)
y_scaled = y_scaler.fit_transform(y_train)

with open(r'trained_models/x_scaler.pkl','wb') as f:
    pickle.dump(x_scaler, f)
with open(r'trained_models/y_scaler.pkl','wb') as f:
    pickle.dump(y_scaler, f)


model = keras.Sequential([
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2', input_shape=(16,)),
        # keras.layers.BatchNormalization(1,),
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2'),
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2'),
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2'),
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2'),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['mean_squared_error'],
              )
model.summary()

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=100),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-20, verbose=1)
]
history=model.fit(x_scaled,y_scaled,epochs=EPOCHS,
                  validation_split=0.2,
                  batch_size=BATCH_SIZE,
                  callbacks=my_callbacks)

model.save('trained_models/my_model.h5')
