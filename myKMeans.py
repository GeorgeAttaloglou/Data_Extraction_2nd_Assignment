import numpy as np
import random

def centroids_init(dimensions:int, k:int) -> list:
    list_of_centroids = []

    for _ in range(k):
        centroid = []
        for _ in range(dimensions):
            centroid.append(random.randrange(0,20))
        list_of_centroids.append(centroid)
    
    return list_of_centroids

def myKmeans(data:list, k:int):
    """
    The data list is rows: 1 for each object columns: ints of data for it like coordinates; data[0]=(1,2), data[1]=(3,2)
    k is how many cluster centers you want
    should return:
    a multidimensional list of the center coordinates: return[0]=(return_x,return_y)
    a list of the cluster values for each point of data

    """
    dimensions = len(data[0])
    centroids = centroids_init(dimensions,k)
    
    list_of_distances = []
    for data_point in data:
        distances = []
        for centroid in centroids:
            dist = float(np.linalg.norm(np.array(data_point) - np.array(centroid)))
            distances.append(dist)
        list_of_distances.append(distances)

    #κατηγοριοποίηση καθε στοιχείου.
    cluster_assignments = []

    for i in list_of_distances:
        cluster_assignments.append(i.index(min(i))+1)

    
    
    return cluster_assignments

def main():
    test = myKmeans([[1,1],[5,3],[10,10],[2,19],[19,2]], 2)
    print(test)

if __name__ == "__main__":
    main()
