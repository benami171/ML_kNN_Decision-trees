"""
Evaluation metrics and entropy calculations.
"""
import numpy as np
from typing import Union, Dict, Tuple

def calculate_binary_entropy(y: np.ndarray) -> float:
    """
    Calculate the binary entropy of a set of binary labels using direct probability
    calculation.
    
    Args:
        y: Array of binary labels (-1 and 1)
    
    Returns:
        float: Binary entropy value between 0 and 1
    """
    if len(y) == 0:
        return 0
    
    p = np.mean(y == 1)
    if p == 0 or p == 1:
        return 0
    
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def calculate_split_entropy(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate the weighted sum of entropies for a binary split with pure node checks.
    
    Args:
        y_left: Labels for left split
        y_right: Labels for right split
    
    Returns:
        float: Weighted sum of binary entropies or infinity for invalid splits
    """
    if len(y_left) == 0 or len(y_right) == 0:
        return float('inf')
        
    if len(set(y_left)) == 1 or len(set(y_right)) == 1:
        return float('inf')
    
    total_samples = len(y_left) + len(y_right)
    weight_left = len(y_left) / total_samples
    weight_right = len(y_right) / total_samples
    
    return (weight_left * calculate_binary_entropy(y_left) + 
            weight_right * calculate_binary_entropy(y_right))

def calculate_classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification error rate as percentage.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Error rate percentage
    """
    return np.mean(y_true != y_pred) * 100

def evaluate_model_parameters(results: Dict[Tuple[int, float], Dict]) -> None:
    """
    Print formatted evaluation results for model parameters.
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    print("\nEvaluation Results:")
    print("\t  Train Error\tTest Error\tDifference")
    print("-" * 50)
    
    for p in [1, 2, float('inf')]:
        print("")
        for k in [1, 3, 5, 7, 9]:
            p_str = 'inf' if p == float('inf') else f'{p:.1f}'
            metrics = results.get((k, p), {})
            
            if metrics:
                train_error = metrics['avg_train_error'] / 100
                test_error = metrics['avg_test_error'] / 100
                diff = metrics['error_difference'] / 100
                
                print(f"p:{p_str} k:{k:<2} {train_error:>10.8f}    {test_error:>10.8f}    {diff:>10.8f}")