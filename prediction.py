import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from preprocess_data_script import preprocess_data

loaded_model = tf.keras.models.load_model('trained_models/19_00_59_1.6293252.h5')

x, y = preprocess_data()

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x.to_numpy())
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1,1))

def predict(x_test):
    x_test = x_scaler.transform(x_test)
    loaded_model.predict()