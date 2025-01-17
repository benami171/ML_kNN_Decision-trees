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


"""
From the data, sample a training set with half the points,
The remaining points are the test set.
"""
def split_train_test(X, y):
    n = X.shape[0] # number of samples
    indices = np.random.permutation(n)
    X_train = X[indices[:n//2]]
    y_train = y[indices[:n//2]]
    X_test = X[indices[n//2:]]
    y_test = y[indices[n//2:]]
    return X_train, y_train, X_test, y_test



def k_nearest_neighbors(X_train, y_train, X_test, k, p):
    """
    Predict the class of each sample in the test set using 
    the k-nearest neighbors algorithm.
    p: norm type (1 for Manhattan, 2 for Euclidean, np.inf for Frechet)
    """
    y_pred = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, ord=p, axis=1)
        nearest_indices = np.argsort(distances)[:k] # indices of the k nearest neighbors
        nearest_labels = y_train[nearest_indices] # labels of the k nearest neighbors
        y_pred.append(np.sign(np.sum(nearest_labels))) # majority vote
    return np.array(y_pred)


def calculate_error(y_true, y_pred):
    """
    returns percantage of incorrect predictions
    """
    return np.mean(y_true != y_pred)*100



# splitting the data each time
def evaluate_knn_parameters(X, y, k_values=[1, 3, 5, 7, 9], p_values=[1, 2, np.inf], n_iterations=100):
    """
    Evaluate k-NN classifier with different parameters over multiple iterations.
    
    Parameters:
    - X: feature matrix
    - y: labels
    - k_values: list of k values to test
    - p_values: list of p values (norms) to test
    - n_iterations: number of random train-test splits to perform
    
    Returns:
    - results: dictionary containing average errors for each parameter combination
    """
    results = {}
    
    for k in k_values:
        for p in p_values:
            train_errors = []
            test_errors = []
            
            for _ in range(n_iterations):
                # Split data
                X_train, y_train, X_test, y_test = split_train_test(X, y)
                
                # Train and predict
                y_train_pred = k_nearest_neighbors(X_train, y_train, X_train, k, p)
                y_test_pred = k_nearest_neighbors(X_train, y_train, X_test, k, p)
                
                # Calculate errors
                train_error = calculate_error(y_train, y_train_pred)
                test_error = calculate_error(y_test, y_test_pred)
                
                train_errors.append(train_error)
                test_errors.append(test_error)
            
            # Store average results
            key = (k, p)
            results[key] = {
                'avg_train_error': np.mean(train_errors),
                'avg_test_error': np.mean(test_errors),
                'error_difference': np.mean(test_errors) - np.mean(train_errors)
            }
    
    return results

# doesnt split the data each time
def evaluate_knn_parameters_2(X, y, k_values=[1, 3, 5, 7, 9], p_values=[1, 2, np.inf], n_iterations=100):
    """
    Evaluate k-NN classifier with different parameters using consistent data splits.
    """
    results = {(k, p): {'train_errors': [], 'test_errors': []} 
              for k in k_values for p in p_values}
    

    for iteration in range(n_iterations):
        # Create one split for this iteration
        
        X_train, y_train, X_test, y_test = split_train_test(X, y)

        # Test all parameter combinations on this split
        for k in k_values:
            for p in p_values:
                y_train_pred = k_nearest_neighbors(X_train, y_train, X_train, k, p)
                y_test_pred = k_nearest_neighbors(X_train, y_train, X_test, k, p)
                
                train_error = calculate_error(y_train, y_train_pred)
                test_error = calculate_error(y_test, y_test_pred)
                
                results[(k, p)]['train_errors'].append(train_error)
                results[(k, p)]['test_errors'].append(test_error)
    
    # Calculate final averages
    final_results = {}
    for params, errors in results.items():
        final_results[params] = {
            'avg_train_error': np.mean(errors['train_errors']),
            'avg_test_error': np.mean(errors['test_errors']),
            'error_difference': np.mean(errors['test_errors']) - np.mean(errors['train_errors'])
        }
    
    return final_results

def print_results(results):
    """
    Print results in the required format with proper alignment.
    """
   
    print("\t  Train Error\tTest Error\tDifference")
    print("-" * 50)
    for p in [1, 2, float('inf')]:
        print("")
        for k in [1, 3, 5, 7, 9]:
            p_str = 'inf' if p == float('inf') else f'{p:.1f}'
            key = (k, p)
            if key in results:
                metrics = results[key]
                
                # Convert from percentage and format with proper spacing
                train_error = metrics['avg_train_error'] / 100
                test_error = metrics['avg_test_error'] / 100
                diff = metrics['error_difference'] / 100
                
                # Use a consistent width for the p:value k:value section
                label = f"p:{p_str} k:{k}"
                
                # Format with proper spacing and alignment
                print(f"{label:<12} {train_error:>10.8f}    {test_error:>10.8f}    {diff:>10.8f}")

def main():

    # Load data
    X, y = load_versicolor_virginica()
    
    # Run evaluation
    results = evaluate_knn_parameters_2(X, y)
    
    print_results(results)


if __name__ == "__main__":
    main()