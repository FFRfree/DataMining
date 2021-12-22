import sklearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from mydata import get_basic_information_and_y
import numpy as np

# y_label = 'MntGoldProds'
# x, y = get_basic_information_and_y(y_label)
# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=321)
# # 搞成一列
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# print(f'x_train shape:{x_train.shape}')
# print(f'x_test shape:{x_test.shape}')
# print(f'y_train shape:{y_train.shape}')
# print(f'y_test shape:{y_test.shape}')

def plot_regression(y_true, y_pred):
    print(f'MSE: {sklearn.metrics.mean_squared_error(y_true, y_pred)}')
    print(f'R^2: {sklearn.metrics.r2_score(y_true, y_pred)}')
    y_test = y_true.flatten()
    y_pred = y_pred.flatten()
    i = np.argsort(y_test)
    y_test = y_test[i]
    y_pred = y_pred[i]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test)
    ax.plot(y_pred)
    return fig

def visualize_model_prediction(model):
    '''
    :param model: function or keras model
    :return: None
    x axis -> age
    y axis -> income
    z axis -> amount prediction
    '''
    fake_data = np.array([5.8138e+04, 0.0000e+00, 0.0000e+00, 1.7200e+02, 6.0000e+01,
       1.5800e+03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
       0.0000e+00])
    # 20-80 61个年龄段 10k-80k 8个income段
    fake_data = np.tile(fake_data, (61*8,1))
    income_range = np.linspace(1e4, 8e4, 8)
    age_range = np.linspace(20, 80, 61)

    for index_outer, income in enumerate(income_range):
        for index_inner, age in enumerate(age_range):
            index = index_outer * 61 + index_inner
            fake_data[index, 5] = age
            fake_data[index, 0] = income

    fake_data_pred = model(fake_data)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for index, income in enumerate(np.linspace(1e4, 8e4, 8)):
        xs_age = fake_data[index * 61:(index + 1) * 61, 5].flatten()
        ys_MntPred = fake_data_pred[index * 61:(index + 1) * 61].flatten()

        ax.plot(xs_age, ys_MntPred, zs=income, zdir='y')

    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    ax.set_zlabel('Amount')
    ax.set_title('Money spent on gold products Prediction')
    ax.view_init(azim=-75, elev=12)
    return fig



