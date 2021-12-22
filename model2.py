import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from preprocess_data_script import preprocess_data

# x = x.reshape(-1,1)
# y = y.re
# Kidhome MntWines MntFruits MntMeatProducts MntFishProducts

# prepare the model
# regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# regr.fit(x, y)
x, y = preprocess_data()
#
#
# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
#
# x_scaled = x_scaler.fit_transform(x.to_numpy())
# y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1,1))
# regr = SVR()
# regr.fit(x_scaled, y_scaled.flatten())
#
# from keras_regr import build_model2
# model = build_model2()
# model.summary()
# history=model.fit(x_scaled,y_scaled,epochs=10,
#                   validation_split=0.2,
#                   batch_size=20)

X_train,X_test, y_train, y_test  = train_test_split(x,y,test_size=0.2, random_state=123)

hp = {'criterion': 'mse',
 'max_depth': 8,
 # 'min_impurity_decrease': 0.18421052631578946,
 'min_samples_leaf': 1,
 'splitter': 'random'}
tree = DecisionTreeRegressor(**hp)   #max_depth设置树深
tree.fit(X_train,y_train)
print(tree.score(X_test,y_test))

y_test = y_test.to_numpy()
y_pred = tree.predict(X_test)

temp = np.array( list(map( lambda x,y:True if int(x/20000) == int(y/20000) else False , y_test, y_pred)) )