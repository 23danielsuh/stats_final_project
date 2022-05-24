import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = [(1.1,2.1,8.1),
          (3.2,4.2,8.0),
          (5.3,1.3,8.2),
          (3.4,2.4,8.3),
          (1.5,4.5,8.0)]

xs, ys, zs = zip(*points)

print(xs)
print(ys)
print(zs)