import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

data_path = os.path.join('dataset', 'studentdata.csv')
data = pd.read_csv(data_path)
data.head()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

# ax.scatter(data['Age'], data['Height'], data['Arm span'])

data = data[['Age', 'Height', 'Arm span']].values

datamean = data.mean(axis=0)
uu, dd, vv = np.linalg.svd(data - datamean)
linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

# shift by the mean to get the line in the right place
linepts += datamean

ax.set_xlabel('Age')
ax.set_ylabel('Height')
ax.set_zlabel('Arm Span')

ax.legend()
ax.scatter3D(*data.T)
ax.plot3D(*linepts.T)

plt.show()