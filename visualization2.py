import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fake_data_pred = np.load('fake_data_pred.npy')
fake_data = np.load('fake_data.npy')

x = np.linspace(20, 80, 61)
y = np.linspace(1e4,8e4,8)



print(f'Z shape: {x.shape[0], y.shape[0]}')
Z = fake_data_pred.reshape((x.shape[0], y.shape[0]))

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(x, y, Z, rstride=8, cstride=8, alpha=0.3)
# ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
plt.show()
# h = plt.contourf(X,Y,Z)
# plt.show()