"""
K-Nearest Neighbors implementation with parameter evaluation.
"""
import numpy as np
from typing import Dict, List, Union, Tuple
from metrics import calculate_classification_error
from data_utils import split_train_test

class KNearestNeighbors:
    """K-Nearest Neighbors classifier implementation."""
    
    def __init__(self, k: int = 3, p: Union[int, float] = 2):
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for test instances."""
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, ord=self.p, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            predictions.append(np.sign(np.sum(nearest_labels)))
        return np.array(predictions)

def evaluate_knn_parameters(X: np.ndarray, y: np.ndarray, 
                          k_values: List[int] = [1, 3, 5, 7, 9],
                          p_values: List[Union[int, float]] = [1, 2, float('inf')],
                          n_iterations: int = 100) -> Dict:
    """
    Evaluate k-NN classifier with different parameters using consistent data splits.
    
    Args:
        X: Feature matrix
        y: Labels array
        k_values: List of k values to test
        p_values: List of p values (norms) to test
        n_iterations: Number of random train-test splits to perform
    
    Returns:
        Dictionary containing evaluation metrics for each parameter combination
    """
    results = {(k, p): {'train_errors': [], 'test_errors': []} 
              for k in k_values for p in p_values}
    
    for _ in range(n_iterations):
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        for k in k_values:
            for p in p_values:
                knn = KNearestNeighbors(k=k, p=p)
                knn.fit(X_train, y_train)
                
                train_pred = knn.predict(X_train)
                test_pred = knn.predict(X_test)
                
                train_error = calculate_classification_error(y_train, train_pred)
                test_error = calculate_classification_error(y_test, test_pred)
                
                results[(k, p)]['train_errors'].append(train_error)
                results[(k, p)]['test_errors'].append(test_error)
    
    final_results = {}
    for params, errors in results.items():
        final_results[params] = {
            'avg_train_error': np.mean(errors['train_errors']),
            'avg_test_error': np.mean(errors['test_errors']),
            'error_difference': np.mean(errors['test_errors']) - np.mean(errors['train_errors'])
        }
    
    return final_results