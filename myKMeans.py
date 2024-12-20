import numpy as np
import random

def centroids(dimensions:int, k:int) -> list:
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
    a list of the center coordinates: return[0]=(return_x,return_y)
    a list of the cluster values for each point of data

    """
    dimensions = len(data[0])
    centroids = centroids(dimensions,k)
    return centroids

def main():
    test = centroids(2,5)
    print(test)

if __name__ == "__main__":
    main()
