"""
Decision tree implementations using both brute-force and entropy-based approaches.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from metrics import calculate_binary_entropy, calculate_split_entropy, calculate_classification_error

@dataclass
class Node:
    """
    Represents a node in the decision tree.
    
    Attributes:
        feature_idx: Index of the feature used for splitting
        threshold: Value used for splitting
        label: Class label for leaf nodes
        left: Left child node
        right: Right child node
        level: Current depth in the tree
    """
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    label: Optional[int] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    level: int = 0

    def predict(self, x: np.ndarray) -> int:
        """Make a prediction for a single instance."""
        if self.label is not None:
            return self.label
        return (self.left.predict(x) if x[self.feature_idx] <= self.threshold 
                else self.right.predict(x))

class DecisionTreeBruteForce:
    """Decision tree implementation using brute-force approach."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.root = None

    def _build_trees(self, X: np.ndarray, y: np.ndarray, current_level: int = 0) -> List[Node]:
        """Build all possible valid decision trees up to max_depth."""
        if current_level >= self.max_depth or len(set(y)) == 1:
            return [Node(label=1 if np.mean(y) >= 0 else -1, level=current_level)]

        trees = []
        for feature_idx in range(X.shape[1]):
            values = sorted(set(X[:, feature_idx]))
            thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if not (np.any(left_mask) and np.any(right_mask)):
                    continue
                
                left_trees = self._build_trees(X[left_mask], y[left_mask], current_level + 1)
                right_trees = self._build_trees(X[right_mask], y[right_mask], current_level + 1)
                
                for left in left_trees:
                    for right in right_trees:
                        trees.append(Node(
                            feature_idx=feature_idx,
                            threshold=threshold,
                            left=left,
                            right=right,
                            level=current_level
                        ))
        
        return trees

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Node, float]:
        """Find the best decision tree using brute force approach."""
        trees = self._build_trees(X, y)
        
        def tree_error(tree: Node) -> float:
            predictions = np.array([tree.predict(x) for x in X])
            return np.mean(predictions != y)
        
        self.root, error = min(((tree, tree_error(tree)) for tree in trees), 
                             key=lambda x: x[1])
        return self.root, error

class DecisionTreeEntropy:
    """Decision tree implementation using entropy-based splitting."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.root = None

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split that minimizes binary entropy."""
        best_entropy = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            values = sorted(set(X[:, feature_idx]))
            thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold 
                right_mask = ~left_mask # Invert the mask
                
                if not (np.any(left_mask) and np.any(right_mask)):
                    continue
                    
                split_entropy = calculate_split_entropy(y[left_mask], y[right_mask])
                
                if split_entropy < best_entropy:
                    best_entropy = split_entropy
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_entropy

    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_level: int = 0) -> Node:
        """Build a decision tree using entropy-based splitting."""
        if (current_level >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) == 0):
            return Node(
                label=1 if np.mean(y) >= 0 else -1,
                level=current_level
            )
        
        feature_idx, threshold, _ = self._find_best_split(X, y)
        
        if feature_idx is None:
            return Node(
                label=1 if np.mean(y) >= 0 else -1,
                level=current_level
            )
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left_node = self._build_tree(X[left_mask], y[left_mask], current_level + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], current_level + 1)
        
        # Check if both children are leaves with the same label
        if (left_node.label is not None and 
            right_node.label is not None and 
            left_node.label == right_node.label):
            # If they have the same label, return a single leaf node
            return Node(
                label=left_node.label,
                level=current_level
            )
        
        # Otherwise, create the split node
        node = Node(
            feature_idx=feature_idx,
            threshold=threshold,
            level=current_level
        )
        node.left = left_node
        node.right = right_node
        
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Node, float]:
        """Fit the entropy-based decision tree."""
        self.root = self._build_tree(X, y)
        predictions = np.array([self.root.predict(x) for x in X])
        error = np.mean(predictions != y)
        return self.root, error