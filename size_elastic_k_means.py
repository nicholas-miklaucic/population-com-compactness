"""This file implements the size-elastic k-means clustering algorithm described in
algorithm.org in numpy.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster

# from assignmentalg import AssignmentAlgs
from balanced_k_means import assign

blocks = gpd.read_file("data_full.shp")
blocks["polygons"] = blocks["geometry"]
blocks["geometry"] = blocks["geometry"].centroid
blocks["x"] = blocks["geometry"].x
blocks["y"] = blocks["geometry"].y

pop_blocks = blocks[blocks["POP10"] > 0].copy()

points = pop_blocks[["x", "y"]].as_matrix()
# normalize distances by constant factor so that the max values in any dimension is 1000,
# preserving relative scale
norm = np.max(points)
points /= norm
points *= 1000



kmeans = sklearn.cluster.KMeans(n_clusters=13, n_init=100)
pop_blocks["kmeans"] = kmeans.fit_predict(points)
pops = pop_blocks["POP10"]
centers = kmeans.cluster_centers_

fig = plt.figure(figsize=(40, 30))
ax = fig.add_subplot("221")
pop_blocks.plot(ax=ax, column="kmeans", categorical=True,
                cmap="tab20", legend=True, figsize=(20, 10))

district_pops1 = np.array([np.sum(pop_blocks[pop_blocks["kmeans"] == d]["POP10"])
                           for d in range(13)])
ax2 = fig.add_subplot("243")
ax2.bar(np.arange(district_pops1.shape[0]), district_pops1)

ax3 = fig.add_subplot("223")
min_size = 0.99 * (np.sum(pops) // 13)
pop_blocks["new"] = assign(points, pops, min_size, 13, s=50, conf=0.9, max_tries=1e3)
dpops2 = np.array([np.sum(pop_blocks[pop_blocks["new"] == d]["POP10"])
                   for d in range(13)])

pop_blocks.plot(ax=ax3, column="new", categorical=True,
                cmap="tab20", legend=True, figsize=(20, 10))
ax4 = fig.add_subplot("247")
ax4.bar(np.arange(dpops2.shape[0]), dpops2)
plt.show()
