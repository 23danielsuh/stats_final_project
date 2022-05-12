import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('dataset/500_Person_Gender_Height_Weight_Index.csv')
sns.set(style = "darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = df['Height']
y = df['Weight']
z = df['Index']

ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('index')

ax.scatter(x, y, z)

plt.show()