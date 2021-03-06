import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd

data_path = os.path.join('dataset', 'studentdata.csv')
data = pd.read_csv(data_path)
data = data[data['Age'] != 0]
data = data[data['Arm span'] != 0]
data = data[data['Height'] != 0]

xs = data['Age'].to_numpy()
ys = data['Arm span'].to_numpy()
zs = data['Height'].to_numpy()

plt.figure()
ax = plt.subplot(111, projection='3d')

ax.scatter(xs, ys, zs, color='b')

tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])

b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

print(f"Equation: {fit[0]} x + {fit[1]} y + {fit[2]} = z")
print("Errors:")
print(errors)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))

Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]

ax.plot_surface(X, Y, Z, color='crimson')

ax.set_xlabel('Age')
ax.set_ylabel('Arm span')
ax.set_zlabel('Height')

plt.show()