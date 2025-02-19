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
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.root = None
        
    def _get_possible_splits(self, X: np.ndarray) -> List[Tuple[int, float]]:
        """Generate all possible splits for each feature."""
        splits = []
        for feature_idx in range(X.shape[1]):
            # Use actual coordinate values as potential splits
            values = sorted(X[:, feature_idx])
            splits.extend([(feature_idx, value) for value in values])
        return splits

    def _generate_all_trees(self, X: np.ndarray, y: np.ndarray, depth: int = 1) -> List[Node]:
        """Generate all possible valid trees recursively."""
        if depth >= self.max_depth or len(set(y)) == 1:
            # Use majority vote for leaf nodes
            return [Node(label=1 if np.sum(y) > len(y)/2 else 0, level=depth)]

        trees = []
        # Always consider leaf node as an option with majority vote
        trees.append(Node(label=1 if np.sum(y) > len(y)/2 else 0, level=depth))
        
        # Try every possible split using actual coordinate values
        splits = self._get_possible_splits(X)
        for feature_idx, threshold in splits:
            # Split data
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # Skip invalid splits
            if not (np.any(left_mask) and np.any(right_mask)):
                continue
                
            # Generate all possible left and right subtrees
            left_trees = self._generate_all_trees(X[left_mask], y[left_mask], depth + 1)
            right_trees = self._generate_all_trees(X[right_mask], y[right_mask], depth + 1)
            
            # Create all possible combinations
            for left_tree in left_trees:
                for right_tree in right_trees:
                    # Skip if both children are leaves with same label
                    if (left_tree.label is not None and 
                        right_tree.label is not None and 
                        left_tree.label == right_tree.label):
                        continue
                        
                    node = Node(
                        feature_idx=feature_idx,
                        threshold=threshold,
                        left=left_tree,
                        right=right_tree,
                        level=depth
                    )
                    trees.append(node)
        
        return trees

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[Node, float]:
        """Find the best decision tree using true brute force approach."""
        # Generate all possible trees
        all_trees = self._generate_all_trees(X, y)
        
        # Find the tree with minimum error
        best_error = float('inf')
        best_tree = None
        
        for tree in all_trees:
            predictions = np.array([tree.predict(x) for x in X])
            error = np.mean(predictions != y)
            
            if error < best_error:
                best_error = error
                best_tree = tree
        
        self.root = best_tree
        return self.root, best_error

class DecisionTreeEntropy:
    """Decision tree implementation using entropy-based splitting."""
    
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.root = None

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split that minimizes binary entropy."""
        best_entropy = float('inf')  # Initialize best_entropy to infinity
        best_feature = None  # Variable to store the index of the best feature
        best_threshold = None  # Variable to store the best threshold for the split
        
        # Iterate over each feature in the dataset
        for feature_idx in range(X.shape[1]):
            values = sorted(set(X[:, feature_idx]))  # Get unique sorted values for the feature
            # Calculate potential thresholds as midpoints between consecutive values
            thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
            
            # Evaluate each threshold for the current feature
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold  # Mask for left split
                right_mask = ~left_mask  # Invert the mask for right split
                
                # Skip if either split is empty
                if not (np.any(left_mask) and np.any(right_mask)):
                    continue
                    
                # Calculate the entropy of the split
                split_entropy = calculate_split_entropy(y[left_mask], y[right_mask])
                
                # Update best split if the current one has lower entropy
                if split_entropy < best_entropy:
                    best_entropy = split_entropy  # Update best entropy
                    best_feature = feature_idx  # Update best feature index
                    best_threshold = threshold  # Update best threshold
        
        return best_feature, best_threshold, best_entropy  # Return the best feature, threshold, and entropy

    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_level: int = 1) -> Node:
        """Build a decision tree using entropy-based splitting."""
        if (current_level >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) == 0):
            return Node(
                label=1 if np.mean(y) >= 0.5 else 0,
                level=current_level
            )
        
        feature_idx, threshold, _ = self._find_best_split(X, y)
        
        if feature_idx is None:  # If no good split is found
            return Node(
                label=1 if np.mean(y) >= 0.5 else 0,
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