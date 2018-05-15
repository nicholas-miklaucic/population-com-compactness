"""This file implements the size-elastic k-means clustering algorithm described in algorithm.org
in numpy."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans

from assignmentalg import AssignmentAlg

blocks = gpd.read_file("data_full.shp")
blocks["polygons"] = blocks["geometry"]
blocks["geometry"] = blocks["geometry"].centroid
blocks["x"] = blocks["geometry"].x
blocks["y"] = blocks["geometry"].y

pop_blocks = blocks[blocks["POP10"] > 0].copy()
pop_blocks = pop_blocks[::500]  # reduce the amount of time it takes to run this

points = pop_blocks[["x", "y"]].as_matrix()
# normalize distances by constant factor so that the max values in any dimension is 1, preserving
# relative scale
norm = np.max(points)
points /= norm



kmeans = MiniBatchKMeans(13)
pop_blocks["kmeans"] = kmeans.fit_predict(points)

centers = kmeans.cluster_centers_


alg = AssignmentAlg()

fig = plt.figure()
ax = fig.add_subplot("311")
pop_blocks.plot(ax=ax, column="kmeans", categorical=True,
                cmap="tab20", legend=True, figsize=(20, 10))

ax2 = fig.add_subplot("312")
pop_blocks["new"] = alg.optimize(points, pop_blocks["POP10"], centers, 10, 1000)
pop_blocks.plot(ax=ax2, column="new", categorical=True,
                cmap="tab20", legend=True, figsize=(20, 10))

district_pops = np.array([np.sum(pop_blocks[pop_blocks["new"] == d]["POP10"])
                          for d in pd.unique(pop_blocks["new"])])
ax3 = fig.add_subplot("313")
ax3.bar(np.arange(district_pops.shape[0]), district_pops)
plt.show()
