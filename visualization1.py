import numpy as np
from matplotlib import pyplot as plt

fake_data_pred = np.load('fake_data_pred.npy')
fake_data = np.load('fake_data.npy')


# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index ,income in enumerate(np.linspace(1e4,8e4,8)):
    xs_age = fake_data[index*61:(index+1)*61, 5].flatten()
    ys_MntPred = fake_data_pred[index*61:(index+1)*61].flatten()

    ax.plot(xs_age, ys_MntPred, zs=income, zdir='y')

ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Amount')
ax.set_title('Money spent on gold products Prediction')
fig.tight_layout()
plt.show()