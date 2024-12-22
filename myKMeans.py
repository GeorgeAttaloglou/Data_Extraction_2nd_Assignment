import numpy as np
import random

def distance_calulator(data:list, centroids:list):
    """
    Receives two lists of the same dimensions
    Returns the distances of every data point from every centroid

    """
    list_of_distances = []
    for data_point in data:
        distances = []
        for centroid in centroids:
            dist = float(np.linalg.norm(np.array(data_point) - np.array(centroid)))
            distances.append(dist)
        list_of_distances.append(distances)

    return list_of_distances


def myKmeans(data:list, k:int):
    """
    The data list is rows: 1 for each object columns: ints of data for it like coordinates; data[0]=(1,2), data[1]=(3,2)
    k is how many cluster centers you want
    should return:
    a multidimensional list of the center coordinates: return[0]=(return_x,return_y)
    a list of the cluster values for each point of data

    """

    #Έλεγχος διαστάσεων δεδομένων
    dimensions = len(data[0])
    for element in data:
        if len(element) > dimensions or len(element) < dimensions:
            raise ValueError("All data points must have the same number of dimensions")

    #Δημιουργία αρχικών κέντρων
    centroids = random.sample(data,k)
    
    #Υπολογισμός των αποστάσεων
    list_of_distances = distance_calulator(data,centroids)

    #Κατηγοριοποίηση καθε στοιχείου.
    assigned_cluster = []
    for i in list_of_distances:
        assigned_cluster.append(i.index(min(i)))

    #Υπολογισμός των ομάδων
    clusters = []
    for i in range(len(centroids)):
        data_in_cluster = []
        for j in range(len(assigned_cluster)):
            if i == assigned_cluster[j]:
                data_in_cluster.append(data[j])
        clusters.append(data_in_cluster)

    #TODO: Recalculate the centroids
    #if new_centroid - old_centroid <= 10^-3: break

    
    return clusters

def main():
    test = myKmeans([[1,1],[5,3],[10,10],[2,19],[19,2]], 3)
    print(test)

if __name__ == "__main__":
    main()
