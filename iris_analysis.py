import numpy as np
from sklearn.datasets import load_iris
from itertools import combinations
import math


def load_versicolor_virginica():
    """Load only Versicolor and Virginica samples from Iris dataset."""
    iris = load_iris()
    indices = np.where((iris.target == 1) | (iris.target == 2))[0] # 1 for Versicolor, 2 for Virginica
    X = iris.data[indices, 1:3] 
    y = np.where(iris.target[indices] == 1, -1, 1) 
    return X, y