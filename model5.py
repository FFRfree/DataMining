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
# 20-80 61个年龄段 10k-80k 8个income段
fake_data = np.tile(x[0], (61*8 ,1))
print(f'使用的假数据: {x[0]}')

income_range = np.linspace(1e4,8e4,8)
age_range = np.linspace(20, 80, 61)

for index_outer, income in enumerate(income_range):
    for index_inner, age in enumerate(age_range):
        index = index_outer * 61 + index_inner
        fake_data[index, 5] = age
        fake_data[index, 0] = income

fake_data_pred = pipeline_predict(fake_data)
np.save('fake_data_pred.npy',fake_data_pred)
np.save('fake_data.npy', fake_data)
# fake_data_income = fake_data[:,5].flatten()
# MntWinePrediciton = fake_data_pred.flatten()
# plt.plot(fake_data_income, MntWinePrediciton)
# plt.xlabel('age')
# plt.ylabel(y_label)
# plt.show()


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for index ,income in enumerate(np.linspace(1e4,8e4,8)):
    xs_age = fake_data[index*61:(index+1)*61, 5].flatten()
    ys_MntPred = fake_data_pred[index*61:(index+1)*61].flatten()

    ax.plot(xs_age, ys_MntPred, zs=income, zdir='y')

ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Amount')
ax.set_title('Money spent on gold products Prediction')
ax.view_init(azim=-75,elev=12)
plt.show()