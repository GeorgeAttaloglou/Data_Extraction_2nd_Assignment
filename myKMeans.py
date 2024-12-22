import numpy as np
import random

def calculate_distances(data_points, centroids):
    """
    Calculate distances from each data point to each centroid.
    Args:
        data_points (list): List of data points.
        centroids (list): List of centroid coordinates.
    Returns:
        list: A list where each element contains distances of a data point to all centroids.
    """
    distances = []
    for point in data_points:
        distances_to_centroids = [
            float(np.linalg.norm(np.array(point) - np.array(centroid)))
            for centroid in centroids
        ]
        distances.append(distances_to_centroids)
    return distances


def my_kmeans(data_points, k):
    """
    Perform k-means clustering on given data points.
    Args:
        data_points (list): List of data points, where each point is a list of coordinates.
        k (int): Number of clusters.
    Returns:
        list: A list of clusters, where each cluster contains data points assigned to it.
    """
    # Validate data dimensions
    dimensions = len(data_points[0])
    for point in data_points:
        if len(point) != dimensions:
            raise ValueError("All data points must have the same number of dimensions")

    # Initialize centroids by randomly sampling from data points
    centroids = random.sample(data_points, k)

    # Compute distances from data points to centroids
    distances = calculate_distances(data_points, centroids)

    # Assign each data point to the nearest centroid
    cluster_assignments = [
        distances_to_centroids.index(min(distances_to_centroids))
        for distances_to_centroids in distances
    ]

    # Group data points into clusters based on assignments
    clusters = [[] for _ in range(k)]
    for point_index, cluster_index in enumerate(cluster_assignments):
        clusters[cluster_index].append(data_points[point_index])

    # TODO: Update centroids and repeat until convergence
    return clusters


def main():
    test_data = [[1, 1], [5, 3], [10, 10], [2, 19], [19, 2]]
    clusters = my_kmeans(test_data, 3)
    print(clusters)


if __name__ == "__main__":
    main()
