"""
Utilities for loading and preprocessing datasets.
"""
import numpy as np
from sklearn.datasets import load_iris
from typing import Tuple

def load_versicolor_virginica() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the Iris dataset for binary classification,
    using only Versicolor and Virginica classes with their second and third features.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features matrix (X) and labels array (y)
    """
    iris = load_iris()
    indices = np.where((iris.target == 1) | (iris.target == 2))[0] # iris.target == 1: Versicolor, iris.target == 2: Virginica
    X = iris.data[indices, 1:3] # Use only the 2nd and 3rd features
    y = np.where(iris.target[indices] == 1, -1, 1) # Versicolor: -1, Virginica: 1
    return X, y

def split_train_test(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and test sets with equal size.
    
    Args:
        X: Feature matrix
        y: Labels array
    
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test)
    """
    n = X.shape[0]
    indices = np.random.permutation(n)
    mid = n // 2
    return X[indices[:mid]], y[indices[:mid]], X[indices[mid:]], y[indices[mid:]]