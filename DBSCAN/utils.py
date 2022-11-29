import numpy as np
import math
import matplotlib.pyplot as plt


def plot(X, y):
    # Plot the dataset X and the corresponding labels y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.show()


def euclidean_distance(x1,x2):
    #calculates l2 distance between two vectors
    '''
    '''
    if len(x1)!=len(x2):
        print("ERROR")
    distance = 0
    for vector_index in range(len(x1)):
        distance += pow(x1[vector_index] - x2[vector_index], 2)
    return math.sqrt(distance)
    #return np.linalg.norm(x1, x2) # is this allowed?
