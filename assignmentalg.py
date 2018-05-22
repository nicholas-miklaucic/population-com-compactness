"""This provides a class with hyperparameters to assign points to cluster centers in a way that
ensures rough equality of population between districts. For more information see algorithm.org."""

import numpy as np
from scipy.optimize import brute, minimize
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
    def distance_assign(self, points, centers, weights):
        """Given points, centers, and weights for each center, assigns points accordingly."""
        distances = np.apply_along_axis(self.distances, 1, points, centers)
        return np.argmin(distances / weights, axis=1)
    
    def distance_assign_score(self, weights, points, pops, centers):
        """Returns the variance of the populations assigned using distance_assign, used as an input
        to scipy.minimize. Note the changed argument order to fit scipy's expectations."""
        curr_plan = self.distance_assign(points, centers, weights)
        dist_pops = np.array([np.sum(pops[curr_plan == d]) for d in range(curr_plan.shape[0])])
        return np.sum((dist_pops - np.mean(dist_pops)) ** 10)
    
    def minimize_assign(self, points, pops, centers):
        """Assigns points using scipy.minimize to minimize the variance in populations."""
        args = (points, pops, centers)
        bounds = np.array([(1, 100) for i in range(centers.shape[0])])
        res = brute(self.distance_assign_score, bounds, args, Ns=1)
        print(res)
        return self.distance_assign(points, centers, res)
    
    def distance_adapt_assign(self, points, pops, centers, gamma=1.3, max_iter=1000):
        """Returns a distance-based measure that ensures equal population by doing a Voronoi
        cell-lie allocation and adjusting weights to ensure population equality.
        Max_iter is the maximum number of iterations before stopping.
        Gamma controls the speed of convergence: it should be positive."""
        weights = np.ones_like(centers[:, 0])
        curr_plan = self.distance_assign(points, centers, weights)
        dist_pops = np.array([np.sum(pops[curr_plan == d]) for d in np.unique(curr_plan)])
        # adjust weights depending on populations
        weights = dist_pops ** 0.5
        weights = (weights + np.min(weights)) / np.sum(weights)
        # reassign
        curr_plan = self.distance_assign(points, centers, weights)
        prev_dist_pops = dist_pops
        dist_pops = np.array([np.sum(pops[curr_plan == d]) for d in range(centers.shape[0])])
        for _ in tqdm(range(max_iter - 2)):
            if np.sum(prev_dist_pops - dist_pops):
                return curr_plan
            # adjust weights depending on populations
            weights = weights + (dist_pops * gamma)
            weights = (weights + np.min(weights)) / np.sum(weights)
            # reassign
            curr_plan = self.distance_assign(points, centers, weights)
            prev_dist_pops = dist_pops
            dist_pops = np.array([np.sum(pops[curr_plan == d]) for d in range(centers.shape[0])])
        return curr_plan
    
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
    def distances(self, point, others):
        """Finds the distances between the given point and given list of other points."""
        return np.sum((others - point) ** 2, axis=1)
    def nearest(self, points, others):
        """Finds the nearest distance between the given point and given list of other points."""
        return np.min(self.distances(points, others))
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
        return self.distance_adapt_assign(points, pops, new_centers)
