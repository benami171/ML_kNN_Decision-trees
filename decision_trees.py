import numpy as np
from sklearn.datasets import load_iris
from dataclasses import dataclass
from typing import List, Optional, Tuple
from graphviz import Digraph

@dataclass
class Node:
    """Represents a node in the decision tree"""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    label: Optional[int] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    level: int = 0

def load_versicolor_virginica():
    """Loads and preprocesses the Iris dataset for binary classification"""
    iris = load_iris()
    indices = np.where((iris.target == 1) | (iris.target == 2))[0]
    X = iris.data[indices, 1:3]  # Only second and third features
    y = np.where(iris.target[indices] == 1, -1, 1)  # Convert to -1/1 labels
    return X, y

def predict(node: Node, x: np.ndarray) -> int:
    """Makes a prediction for a single instance"""
    if node.label is not None:
        return node.label
    return predict(node.left, x) if x[node.feature_idx] <= node.threshold else predict(node.right, x)

def calculate_error(node: Node, X: np.ndarray, y: np.ndarray) -> float:
    """Calculates classification error rate"""
    predictions = np.array([predict(node, x) for x in X])
    return np.mean(predictions != y)

def find_split_points(X: np.ndarray) -> List[Tuple[int, float]]:
    """Finds all possible split points for both features"""
    split_points = []
    for feature_idx in [0, 1]:
        values = sorted(set(X[:, feature_idx]))
        thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
        split_points.extend((feature_idx, threshold) for threshold in thresholds)
    return split_points

def build_trees(X: np.ndarray, y: np.ndarray, max_level: int, current_level: int = 0) -> List[Node]:
    """Builds all possible valid decision trees up to max_level"""
    trees = []
    
    # Base cases
    if current_level >= max_level or len(set(y)) == 1:
        return [Node(label=1 if np.mean(y) >= 0 else -1, level=current_level)]

    # Try all possible splits
    for feature_idx, threshold in find_split_points(X):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if not (np.any(left_mask) and np.any(right_mask)):
            continue
        
        # Build subtrees recursively
        left_trees = build_trees(X[left_mask], y[left_mask], max_level, current_level + 1)
        right_trees = build_trees(X[right_mask], y[right_mask], max_level, current_level + 1)
        
        # Create all valid combinations
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

def find_best_tree(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[Node, float]:
    """Finds the best decision tree using brute force approach"""
    trees = build_trees(X, y, k)
    return min(((tree, calculate_error(tree, X, y)) for tree in trees), 
              key=lambda x: x[1])

def visualize_tree(node: Node, dot=None) -> Digraph:
    """Creates a visual representation of the decision tree"""
    if dot is None:
        dot = Digraph()
        dot.attr(rankdir='TB')
    
    node_id = str(id(node))
    if node.label is not None:
        dot.node(node_id, f'Leaf: {node.label}\nLevel: {node.level}')
    else:
        dot.node(node_id, f'Feature {node.feature_idx}\nThreshold: {node.threshold:.3f}\nLevel: {node.level}')
    
    if node.left:
        dot.edge(node_id, str(id(node.left)), 'True')
        visualize_tree(node.left, dot)
    if node.right:
        dot.edge(node_id, str(id(node.right)), 'False')
        visualize_tree(node.right, dot)
    
    return dot

def main():
    # Load and prepare data
    X, y = load_versicolor_virginica()
    
    # Find optimal tree
    best_tree, error = find_best_tree(X, y, k=3)
    print(f"Best tree error rate: {error:.4f}")
    
    # Visualize result
    dot = visualize_tree(best_tree)
    dot.render("decision_tree", view=True, format="png")

if __name__ == "__main__":
    main()