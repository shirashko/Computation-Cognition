import numpy as np
import matplotlib.pyplot as plt


# question 1
def create_matrix():
    mat = np.loadtxt("wines.csv", delimiter=",", skiprows=1)
    return mat
