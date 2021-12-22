from tensorflow.keras.models import load_model
import pickle
from sklearn.model_selection import train_test_split
from mydata import get_basic_information_and_y

# load model
model = load_model('trained_models/my_model.h5')


with open(r'trained_models/x_scaler.pkl','rb') as f:
    x_scaler = pickle.load(f)
with open(r'trained_models/y_scaler.pkl','rb') as f:
    y_scaler = pickle.load(f)

def pipeline_predict(x):
    x_scaled = x_scaler.transform(x)
    y_pred_scaled = model.predict(x_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred

'''
reload data
'''
y_label = 'MntGoldProds'
x, y = get_basic_information_and_y(y_label)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=321)


'''
评价模型
'''
from model_evaluation_tools import *
y_test_pred = pipeline_predict(x_test)

print(f'MSE: {sklearn.metrics.mean_squared_error(y_test, y_test_pred)}')
print(f'R^2: {sklearn.metrics.r2_score(y_test, y_test_pred)}')

fig1 = plot_regression(y_test, y_test_pred)
fig1.show()
'''
fake data
'''
fig2 = visualize_model_prediction(pipeline_predict)
fig2.show()