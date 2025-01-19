"""
Using Python 3.11.9
"""
import numpy as np
from sklearn.datasets import load_iris
from dataclasses import dataclass
from typing import List, Optional, Tuple


def load_versicolor_virginica():
    """Load only Versicolor and Virginica samples from Iris dataset."""
    iris = load_iris()
    indices = np.where((iris.target == 1) | (iris.target == 2))[0] # 1 for Versicolor, 2 for Virginica
    X = iris.data[indices, 1:3] 
    y = np.where(iris.target[indices] == 1, -1, 1) 
    return X, y

@dataclass
class Node:
    """Tree node class"""
    feature_idx: Optional[int] = None  # Index of feature to split on (0 or 1 for our 2D data)
    threshold: Optional[float] = None  # Value to split on
    label: Optional[int] = None  # For leaf nodes: -1 or 1
    left: Optional['Node'] = None  # Left child
    right: Optional['Node'] = None  # Right child
    level: int = 0  # Current level in tree

def calculate_error(node: Node, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate classification error for a tree"""
    predictions = np.array([predict(node, x) for x in X])
    return np.mean(predictions != y)

def predict(node: Node, x: np.ndarray) -> int:
    """Predict class for a single instance"""
    if node.label is not None:  # Leaf node
        return node.label
    
    if x[node.feature_idx] <= node.threshold:
        return predict(node.left, x)
    return predict(node.right, x)

def get_possible_splits(X: np.ndarray, y: np.ndarray) -> List[Tuple[int, float]]:
    """Get all possible split points (feature_idx, threshold)"""
    splits = []
    for feature_idx in [0, 1]:  # For each feature
        values = sorted(set(X[:, feature_idx]))  # Unique values
        # Use midpoints between consecutive values as thresholds
        thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
        splits.extend((feature_idx, threshold) for threshold in thresholds)
    return splits

def build_all_trees(X: np.ndarray, y: np.ndarray, max_level: int, current_level: int = 0) -> List[Node]:
    """Build all possible decision trees up to max_level"""
    if current_level >= max_level:
        # Create leaf nodes with majority labels
        majority = 1 if np.mean(y) >= 0 else -1
        return [Node(label=majority, level=current_level)]
    
    # If all examples have same label, make a leaf
    if len(set(y)) == 1:
        return [Node(label=y[0], level=current_level)]
    
    trees = []
    possible_splits = get_possible_splits(X, y)
    
    # Try all possible splits
    for feature_idx, threshold in possible_splits:
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Skip if split doesn't divide the data
        if not (np.any(left_mask) and np.any(right_mask)):
            continue
            
        # Recursively build subtrees
        left_trees = build_all_trees(X[left_mask], y[left_mask], 
                                   max_level, current_level + 1)
        right_trees = build_all_trees(X[right_mask], y[right_mask], 
                                    max_level, current_level + 1)
        
        # Combine all possible combinations of left and right subtrees
        for left_tree in left_trees:
            for right_tree in right_trees:
                node = Node(
                    feature_idx=feature_idx,
                    threshold=threshold,
                    left=left_tree,
                    right=right_tree,
                    level=current_level
                )
                trees.append(node)
    
    return trees

def find_best_tree(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[Node, float]:
    """Find the best decision tree using brute force"""
    all_trees = build_all_trees(X, y, k)
    best_error = float('inf')
    best_tree = None
    
    for tree in all_trees:
        error = calculate_error(tree, X, y)
        if error < best_error:
            best_error = error
            best_tree = tree
    
    return best_tree, best_error

def main():
    # Load data
    X, y = load_versicolor_virginica()  # Use your existing loading function
    
    # Find best tree with k=3 levels
    best_tree, error = find_best_tree(X, y, k=3)
    
    print(f"Best tree error: {error:.4f}")
    
    # TODO: We'll need to add visualization code to draw the tree
    
if __name__ == "__main__":
    main()