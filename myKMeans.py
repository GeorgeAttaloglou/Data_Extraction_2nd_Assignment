import numpy as np
import random

def myKmeans(data:list, k:int):
    """
    The data list is rows: 1 for each object columns: ints of data for it like coordinates; data[0]=(1,2), data[1]=(3,2)
    k is how many cluster centers you want
    should return:
    a list of the center coordinates: return[0]=(return_x,return_y)
    a list of the cluster values for each point of data

    """
    centroid_coordinates_init = []
    dimensions = len(data[0])

    for _ in range(1,dimensions):
        centroid_coordinates_init.append(random.randrange(0,20))

    return centroid_coordinates_init

def main():
    if __name__ == "__main__":
        data = [(1,2),(3,2),(1,3),(2,3),(8,9),(9,8),(8,8),(9,9)]
        k = 2
        print(myKmeans(data, k))
