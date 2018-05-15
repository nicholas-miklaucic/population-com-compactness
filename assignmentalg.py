"""This provides a class with hyperparameters to assign points to cluster centers in a way that
ensures rough equality of population between districts. For more information see algorithm.org."""

import numpy as np
from tqdm import tqdm


class AssignmentAlg:
    def __init__(self):
        pass
    def metrics(self, point, centers, pops, max_pop, alpha, beta):
        """Given a point and centers, returns the metric as an array [m1, m2, m3, ...] for each center. Pops
        is the population of each center's current cluster. Alpha is the weight for proximity; beta
        controls how sensitive the capacity cost is.
        """
        distance = np.sum((centers - point) ** 2, axis=1)
        capacity = np.tan((np.pi / 2) - ((-pops + max_pop) / (max_pop * beta)))
        
        return (alpha * distance) + capacity
    def assign(self, points, pops, centers, alpha, beta):
        """Given points, their populations, and centers, assigns them as described, using a random order.
        Alpha is the proximity weight, beta controls the sensitivity of the compactness.
        """
        n_points = points.shape[0]
        n_centers = centers.shape[0]
        # add one to avoid taking log of 0
        max_pop = np.ceil(np.sum(pops) / n_centers) + 1        
        order = np.argsort(np.apply_along_axis(self.nearest, 1, points, centers))
        # to store district assignments
        assignments = np.zeros_like(order)
        # to store district populations
        pops = np.zeros(centers.shape[0])
        for ind in tqdm(order):
            metrics = self.metrics(points[ind], centers, pops, max_pop, alpha, beta)
            assignment = np.argmin(metrics)
            assignments[ind] = assignment
            # increase population in cluster by one after assignment
            pops[assignment] += 1
        return assignments
    def nearest(self, point, others):
        """Finds the smallest distance between the given point and given list of other points."""
        return np.min(np.sum((others - point) ** 2, axis=1))
    def population_com(self, points, pops):
        """Computes the population-weighted center of mass for points with populations."""
        return np.average(points, axis=0, weights=pops)
    def optimize(self, points, pops, centers, alpha, beta, gamma=0.8, n_iter=100):
        """Given a set of points, their populations, and cluster centers, repeatedly assigns the points to
        clusters and then resets the centers, hoping to achieve an optimal assignment.  Alpha and
        beta are proximity and capacity sensitivity weights, but they change for more efficient
        optimization: the given values are only the initials, over time they weight more towards
        distance as controlled by gamma, which lies between 0 and 1.
        """
        assignments = self.assign(points, pops, centers, alpha, beta)
        new_alpha = alpha
        new_beta = beta
        for i in range(n_iter-1):  # already assigned once
            # compute alpha and beta for this run
            new_alpha /= gamma
            new_beta *= gamma
            # compute new centers
            new_centers = np.array([self.population_com(points[assignments == d], pops[assignments == d])
                                    for d in range(centers.shape[0])])
            assignments = self.assign(points, pops, new_centers, new_alpha, new_beta)
        return assignments
