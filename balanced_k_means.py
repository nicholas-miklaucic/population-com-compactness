"""This file implements the algorithm described in Banerjee and Ghosh, "Scalable Clustering
Algorithms with Balance Constraints" allowing for balanced k-means clustering."""

import networkx as nx
import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def sample(points, k, s, min_size, conf=0.99):
    """Given a minimum size, points, the number of clusters, and the desired number of points in
    each cluster (at least expected), returns a k-vector reps that contains k representative points
    of k different clusters using kmeans. Confidence is the desired certainty that the sampling was
    effective.
    The heuristic that Banerjee and Ghosh prove is expected to match the actual data is csk ln k,
    where c is a constant larger than the reciprocal of ln k. The desired confidence is used to
    derive c, from the equation summarized in the following link:
    http://www.wolframalpha.com/input/?i=solve+d+%3D+s+%2F+(ln+k)+*+(c+ln+k+-+ln(4c+ln+k))+-+1+for+c"""
    # 1 - 1/(k^d) is the confidence bound, so we solve for d
    # http://www.wolframalpha.com/input/?i=solve+x+%3D+1+-+1%2F(k%5Ed)+for+d
    d = np.log(np.reciprocal(1 - conf)) / np.log(k)

    # now we use the equation for c cited above, but we take out the denominator because we'll just
    # end up multiplying by it anyway
    # print(-0.25 * k ** ((-1 - d) / s))
    c = -scipy.special.lambertw(-0.25 * k**((-1 - d) / s), -1).real
    # print(c * s * k)
    num_samples = int(np.ceil(c * s * k))
    print(f"Num to sample: {num_samples}")
    if not (0 < num_samples < points.shape[0]):
        raise ValueError(
            ("Confidence too high: would require {} samples but only have" +
             " {}!").format(num_samples, points.shape[0]))
    # now we sample csk ln k points at random and do k-means
    sample_inds = np.random.choice(
        np.arange(points.shape[0]), size=num_samples, replace=False)
    samples = points[sample_inds]

    # now compute k-means centers
    mini_kmeans = MiniBatchKMeans(n_clusters=k, compute_labels=False)
    mini_kmeans.fit(samples)
    return mini_kmeans.cluster_centers_


def initialize(points, pops, reps, min_size):
    """Returns a new nx1 vector clusters that assigns each point in points to a cluster
    represented by reps such that no cluster goes below the minimum size.

    """
    # initialize everything to -1
    clusters = np.zeros_like(points[:, 0]) - 1
    k = reps.shape[0]
    if k * min_size > np.sum(pops):  # untenable
        raise ValueError("k={} and m={} are invalid with n={}!".format(
            k, min_size, np.sum(pops)))
    # first assign the minimum number of points to every cluster
    for i in range(k):
        dists = np.sum((points - reps[i])**2, axis=1)
        dists_sort = np.argsort(dists)
        cluster_pop = 0
        curr_ind = 0
        while cluster_pop < min_size:
            # if next point is free, assign it
            if clusters[dists_sort[curr_ind]] == -1:
                cluster_pop += pops.iloc[dists_sort[curr_ind]]
                clusters[dists_sort[curr_ind]] = i
            curr_ind += 1
    # now just greedily assign everything else
    for i in range(points.shape[0]):
        if clusters[i] == -1:
            clusters[i] = np.argmin(np.sum((reps - points[i])**2, axis=1))
    return clusters


def populate(points, pops, existing, reps, min_size, max_tries=1e4):
    """Assigns the remaining unassigned points to a cluster, returning a discrete
    k-partitioning of the total set that is balanced to min_size.  Points is an nx2 matrix
    of x and y coordinates. Existing is an n-vector of cluster assignments, with -1 for
    unassigned points. Reps is a kx2 matrix of representative points in each
    cluster. Min_size is the minimum size of a cluster.  Pops is the populations of each
    point. Max_tries is the number of times to attempt convergence before throwing an
    error.

    """
    # see the paper for more info on this
    # this is the algorithm "poly-stable"
    # basically a form of stable marriage: clusters "propose" to points
    k = reps.shape[0]
    dists_per_rep = []
    sorted_dist_inds = []
    nums_to_assign = []
    inds = [0 for i in range(k)
            ]  # keep track of how many have been proposed to already
    for i in range(k):
        rep = reps[i]
        dists = np.sum((rep - points[existing == -1])**2, axis=1)
        dists_per_rep.append(dists)
        sorted_dist_inds.append(np.argsort(dists))
        nums_to_assign.append(max(0, min_size - np.sum(pops[existing == i])))
    num_tries = max_tries
    while np.sum(nums_to_assign) > 0 and num_tries > 0:
        num_tries -= 1
        for h in range(k):
            ind = inds[h]
            pi_h = sorted_dist_inds[h]
            m_h_star = nums_to_assign[h]
            # select next m_h non-proposed points
            for i in range(ind, ind + nums_to_assign[h]):
                if existing[pi_h[i]] == -1:  # point is free
                    existing[pi_h[i]] = h  # assign
                    nums_to_assign[h] -= pops.iloc[pi_h[i]]
                # if the point is assigned, but this assignment is better
                elif dists_per_rep[h][pi_h[i]] < dists_per_rep[existing[pi_h[i]]][pi_h[i]]:
                    nums_to_assign[h] -= pops[pi_h[i]]
                    nums_to_assign[existing[pi_h[i]]] += pops.iloc[pi_h[i]]
                    # reassign now mark the number that we went
                    existing[pi_h[i]] = h
            # through, the original m_h value from before the loop, as already done
            inds[h] += m_h_star

    return existing


def refine(points, pops, clusters, min_size, max_tries=1e4):
    """Iteratively refines the assignments given in clusters (an n-vector with the numbers 0 ->
    k-1) for the given points, such that the balancing criterion is preserved.  Max_tries
    is the upper limit on attempted convergence.

    Returns the changed clusters.

    """
    if max_tries == 0:
        return clusters

    # flag for if anything changed: if never set to True, we're done
    did_change = False
    # Also given in paper
    k = int(np.max(clusters)) + 1
    reps = np.array([
        np.average(
            points[np.array(clusters == i)],
            weights=pops[clusters == i],
            axis=0) for i in range(k)
    ])
    # First step: individual refinement If a point can be switched from a cluster to a
    # closer cluster, and the clusters can support that without going under the limit, then
    # do so
    h = []  # to store potential new clusters
    for i in range(points.shape[0]):
        p = points[i]
        dists = np.sum((reps - p)**2, axis=1)
        dist_sort = np.argsort(dists)
        # append clusters that are closer than the current one
        h.append(dist_sort[dist_sort < clusters[i]])
        if dist_sort[0] != clusters[i]:  # not assigned to closest cluster
            # if point is not required for cluster population requirement
            if np.sum(pops.iloc[clusters ==
                                clusters[i]]) > min_size + pops.iloc[i]:
                # reassign
                clusters[i] = dist_sort[0]
                did_change = True
    h = np.array(h)
    # Now the fun part. We construct a graph with vertices corresponding to each cluster
    # and weighted directed edges between each corresponding to the number of points in
    # that cluster that would prefer the other one. This is where NetworkX comes in.
    DG = nx.DiGraph()
    DG.add_nodes_from(list(range(k)))
    # most efficient to go through the points once and store weights and then put them in
    weights = np.zeros(
        (k, k))  # weights[i][j] is the weight of the edge from i to j
    for i, h_i in enumerate(h):
        for j in h_i:
            weights[int(clusters[i])][int(j)] += pops.iloc[i]

    for i in range(k):
        for j in range(k):
            if weights[i][j] != 0 and i != j:
                DG.add_edge(i, j, weight=weights[i][j])

    # now we destroy every component, as each represents an inefficiency
    components = [
        c for c in nx.strongly_connected_components(DG) if len(c) >= 2
    ]
    print(f"The number of components is {len(components)}")
    print(f"The components are: {components}")
    for component in components:
        for node in component:
            for cycle in sorted([c for c in nx.simple_cycles(DG.subgraph(component).copy()) if node in
                                c], key=len, reverse=True):
                # give the nodes in order: cycle gives edges
                to_transfer = min([
                    weights[cycle[i]][cycle[i + 1]] for i in range(len(cycle) - 1)
                ])
                # transfer the given number of points from each
                for ind, i in tqdm(enumerate(cycle[:-1])):
                    t = to_transfer  # max capacity to transfer
                    j = cycle[ind + 1]
                    for h_ind in np.arange(points.shape[0])[clusters == i]:
                        pop = pops.iloc[int(h_ind)]
                        if j in h[h_ind] and t > pop:  # point can move and not done transferring yet
                            # reassign point
                            print(f"Refining point {h_ind} from {i} to {j}")
                            clusters[h_ind] = j
                            # change weights
                            weights[i][j] -= pop
                            DG[i][j]['weight'] -= pop
                            # if no more weight can be transferred along this edge
                            arr = np.array([j in h_i for h_i in h]) & (clusters == i)           
                            if DG[i][j]['weight'] < np.min(arr):
                                DG.remove_edge(i, j)
                            # indicate that more has been transferred
                            t -= pop
                            did_change = True
    if not did_change:  # we're converged
        print(f"Convergence achieved with {max_tries} steps to go!")
        return clusters
    else:
        # try once more for convergence
        return refine(points, pops, clusters, min_size, max_tries - 1)


def assign(points, pops, min_size, k, s=50, conf=0.98, max_tries=1e4):
    """Returns an assignment of points to clusters as returned by balanced k-means.  Max_tries
    controls how long to search for convergence. S controls how much sampling is done
    versus populating. Confidence controls it in a similar fashion: it measures the
    confidence that the samples did their job.
    """
    reps = sample(points, k, s, min_size, conf)
    clusters = initialize(points, pops, reps, min_size)
    clusters = populate(points, pops, clusters, reps, min_size, max_tries)
    return refine(points, pops, clusters, min_size, max_tries)
