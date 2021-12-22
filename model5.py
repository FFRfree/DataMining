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

y_label = 'MntMeatProducts'
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

model = keras.Sequential([
        # keras.layers.Dense(7, activation='relu', input_shape=(17,), kernel_regularizer='l1_l2'),
        # keras.layers.BatchNormalization(1,),
        keras.layers.Dense(3, activation='relu', kernel_regularizer='l1_l2',input_shape=(17,)),
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

def pipeline_predict(x):
    x_scaled = x_scaler.transform(x)
    y_pred_scaled = model.predict(x_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred

'''
评价模型
'''

y_test_pred = pipeline_predict(x_test)

print(f'MSE: {sklearn.metrics.mean_squared_error(y_test, y_test_pred)}')

y_test =y_test.flatten()
y_pred = y_test_pred.flatten()
i = np.argsort(y_test)
y_test = y_test[i]
y_pred = y_pred[i]

plt.plot(y_test)
plt.plot(y_pred)
plt.show()


'''
fake data
'''
fake_data = np.tile(x[0], (61, 1))
print(x[0])

for index, age in enumerate(np.linspace(20, 80, 61)):
    fake_data[index, 5] = age

fake_data_pred = pipeline_predict(fake_data)

fake_data_income = fake_data[:,5].flatten()
MntWinePrediciton = fake_data_pred.flatten()
plt.plot(fake_data_income, MntWinePrediciton)
plt.xlabel('age')
plt.ylabel(y_label)
plt.show()