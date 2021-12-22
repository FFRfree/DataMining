import numpy as np
import pandas as pd

def get_basic_information_and_y(y_label:str= 'MntWines')-> 'ndarray':
    data = pd.read_csv('basic_info.csv')
    x = data[data.columns.drop(['MntMeatProducts', 'MntSweetProducts', 'MntGoldProds','MntWines', 'MntFruits'])]
    try:
        y = data[y_label]
    except Exception as e:
        print(f"{y_label} not in the table")
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


def generate_fake_data():
    data = pd.read_csv('basic_info.csv')
    fake_data = np.tile(data[123], (100, 1))

    for index, income in enumerate(np.linspace(1e4, 1e5, 100)):
        fake_data[index, 1] = income