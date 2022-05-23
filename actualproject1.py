import numpy as np
import pandas as pd
import os
from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d

data_path = os.path.join('dataset', 'studentdata.csv')
data = pd.read_csv(data_path)
data = data[['Age', 'Arm span', 'Height']]
points = Points(data.to_numpy())

plane = Plane.best_fit(points)

plot_3d(
    points.plotter(c='k', s=50, depthshade=False),
    plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
)