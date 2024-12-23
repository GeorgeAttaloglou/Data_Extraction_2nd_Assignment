import numpy as np
import matplotlib.pyplot as plt
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

def calculate_sse(clusters, centroids):
    """
    Calculate the Sum of Squared Errors (SSE).
    Args:
        clusters (list): A list of clusters, where each cluster contains data points assigned to it.
        centroids (list): A list of centroid coordinates.
    Returns:
        float: The Sum of Squared Errors.
    """
    sse = 0
    for cluster_index, cluster_points in enumerate(clusters):
        centroid = np.array(centroids[cluster_index])
        for point in cluster_points:
            distance = np.linalg.norm(np.array(point) - centroid)
            sse += distance**2
    return sse

def my_kmeans(data_points, k):
    """
    Perform k-means clustering on given data points.
    Args:
        data_points (list): List of data points, where each point is a list of coordinates.
        k (int): Number of clusters.
    Returns:
        tuple: A list of clusters, where each cluster contains data points assigned to it, and the final SSE value.
    """
    #Έλεγχος των διαστάσεων
    dimensions = len(data_points[0])
    for point in data_points:
        if len(point) != dimensions:
            raise ValueError("All data points must have the same number of dimensions")

    #Αρχικοποίηση κέντρων
    centroids = random.sample(data_points, k)

    while True:
        #Υπολογισμός αποστάσεων
        distances = calculate_distances(data_points, centroids)

        #Κατηγοριοποίηση των δεδομένων
        cluster_assignments = [
            distances_to_centroids.index(min(distances_to_centroids))
            for distances_to_centroids in distances
        ]

        #Υπολογισμός ομάδων
        clusters = [[] for _ in range(k)]
        for point_index, cluster_index in enumerate(cluster_assignments):
            clusters[cluster_index].append(data_points[point_index])

        #Επαναυπολογισμός των κέντρων μετά την ομαδοποίηση
        new_centroids = [
            np.mean(cluster, axis=0).tolist() if cluster else centroids[i]
            for i, cluster in enumerate(clusters)
        ]

        #οπτικοποίηση βημάτων
        markers = ['+', 'x', 'o']
        colors = ['r', 'g', 'b']
        for i, cluster in enumerate(clusters):
            arr = np.array(cluster)
            plt.scatter(arr[:,0], arr[:,1], c=colors[i], marker=markers[i])
        plt.scatter(np.array(centroids)[:,0], np.array(centroids)[:,1], c='black', marker='.', s=200)
        plt.show()

        #Αν δεν έχουν κουνηθεί τα καινούρια κέντρα βγαίνουμε απο την λούπα
        if np.allclose(centroids, new_centroids, atol=1e-3):
            break
        centroids = new_centroids

    #Υπολογισμός του SSE
    sse = calculate_sse(clusters, centroids)
    print("SSE:", sse)
    
    return centroids, cluster_assignments

def main():
    #Δημιουργία δεδομένων
    test_1 = np.random.normal([4,0], [0.29, 0.4], (50, 2))
    test_2 = np.random.normal([5,7], [0.4, 0.9], (50, 2))
    test_3 = np.random.normal([7,4], [0.64, 0.64], (50, 2))
    test_data = np.vstack((test_1, test_2, test_3)).tolist()

    centroids, assignments = my_kmeans(test_data, 3)
    print("Centroids:", centroids)
    print("Assignments:", assignments)

if __name__ == "__main__":
    main()
