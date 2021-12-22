import matplotlib.pyplot as plt
from preprocess_data_script import preprocess_data
import numpy as np
x, y = preprocess_data()
y = y.to_numpy()

fig, axs = plt.subplots(nrows=3, ncols=3)

for ax,feature in zip(axs.flatten(),x.columns):
    # data = np.stack((x[feature].to_numpy() , y ))
    # data.sort()
    data_x = x[feature].to_numpy()
    data_y = y.copy()
    i = np.argsort(data_x)
    data_x = data_x[i]
    data_y = data_y[i]
    ax.set_title(feature)
    ax.plot(data_x,data_y)
plt.tight_layout()
fig.show()
