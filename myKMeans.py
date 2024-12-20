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
    
    #TODO: Add euclidian distance calculation of centroids from data_points
    list_of_distances = []
    for centroid in centroids:
        distances = []
        for data_point in data:
            dist = int(np.linalg.norm(np.array(centroid) - np.array(data_point)))
            distances.append(dist)
        list_of_distances.append(distances)
    
    return list_of_distances

def main():
    test = myKmeans([[1,2],[2,3],[3,4]], 2)
    print(test)

if __name__ == "__main__":
    main()
