import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
import os
import pandas as pd

data_path = os.path.join('dataset', 'studentdata.csv')
data = pd.read_csv(data_path)
data = data[data['Age'] != 0]
data = data[data['Arm span'] != 0]
data = data[data['Height'] != 0]

ix = data['Age'].to_numpy()
iy = data['Arm span'].to_numpy()
iz = data['Height'].to_numpy()

x = torch.tensor(np.array([  ix, iy ]).T).float()
y = torch.tensor(iz).float()
degree =1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        if degree == 1 : 
            weights = torch.distributions.Uniform(-1, 1).sample((4,))
        if degree == 3 : 
            weights = torch.distributions.Uniform(-1, 1).sample((8,))

        self.weights = nn.Parameter(weights)        
        
    def forward(self, X):
        if degree ==1 : 
            a_1, a_2, a_3, a_4 = self.weights
            return (a_1 *X[:,0] + a_2*X[:,1] +a_3) 
        if degree == 3 : 
            a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8 = self.weights
            return (a_1 * X[:,0]**3 + a_2*X[:,1]**3  + a_3*X[:,0]**2 + a_4 *X[:,1] **2  + a_5 *X[:,0] + a_6 *X[:,1] + a_7)


def training_loop(model, optimizer):
    losses = []
    loss = 10000
    it = 0
    if degree == 3 : 
        lim = 0.1
    if degree == 1 : 
        lim = 0.1
    while loss > lim:
        it +=1
        if it > 100000:
            break
        print(it)
        preds1= model(x).float()
        l1 = torch.nn.L1Loss()
        loss = l1(preds1, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().numpy())
        print(loss)  
    return losses


m = Model()
if degree == 1 :
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    losses = np.array(training_loop(m, opt))
if degree == 3 : 
    opt= torch.optim.Adam(m.parameters(), lr=0.001)
    losses = np.array(training_loop(m, opt))

params=list(m.parameters())[0].detach().numpy()

X = np.arange(0, 2, 0.1)
Y = np.arange(0, 2, 0.1)
X, Y = np.meshgrid(X, Y)

if degree == 1 : 
    Z = (params[0] * X + params[1]*Y + params[2])
if degree == 3: 
    Z = (params[0] * X**3 + params[1]*Y**3 + params[2]*X**2 + params[3]*Y**2 + params[4]*X + params[5]*Y + params[6])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
surf = ax.plot_surface(X, Y, Z, color='tab:orange', alpha = 0.5,linewidth=0, antialiased=False)
ax.scatter3D(ix,iy,iz, alpha = 0.3, s=2)
plt.show()