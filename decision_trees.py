import numpy as np
from sklearn.datasets import load_iris
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

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

def build_trees(X: np.ndarray, y: np.ndarray, max_level: int, current_level: int = 0) -> List[Node]:
    """Builds all possible valid decision trees up to max_level"""
    # Base cases
    if current_level >= max_level or len(set(y)) == 1:
        return [Node(label=1 if np.mean(y) >= 0 else -1, level=current_level)]

    trees = []
    # Find split points for both features
    for feature_idx in [0, 1]:
        values = sorted(set(X[:, feature_idx]))
        thresholds = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
        
        for threshold in thresholds:
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

def plot_tree(node: Node, ax=None, x=0.5, y=1.0, width=1.0):
    """Plots the hierarchical structure of the decision tree"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Draw current node
    circle = plt.Circle((x, y), 0.02, color='white', ec='black')
    ax.add_patch(circle)
    
    # Add node text
    text = f'Label: {node.label}' if node.label is not None else f'X[{node.feature_idx}] ≤ {node.threshold:.2f}'
    ax.text(x, y + 0.02, text, ha='center', va='bottom')
    
    # Draw children if they exist
    if node.left or node.right:
        child_y = y - 0.2
        if node.left:
            left_x = x - width/4
            ax.plot([x, left_x], [y, child_y], 'k-')
            plot_tree(node.left, ax, left_x, child_y, width/2)
        if node.right:
            right_x = x + width/4
            ax.plot([x, right_x], [y, child_y], 'k-')
            plot_tree(node.right, ax, right_x, child_y, width/2)
    
    return ax

def plot_decision_boundary(tree: Node, X: np.ndarray, y: np.ndarray, ax=None):
    """Plots the decision boundaries and training data points"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Set up the mesh grid
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Make predictions
    Z = np.array([predict(tree, np.array([x, y])) 
                 for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Versicolor')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Virginica')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.legend()
    return ax

def main():
    print("Loading data...")
    X, y = load_versicolor_virginica()
    
    print("Finding best decision tree...")
    best_tree, error = find_best_tree(X, y, k=3)
    print(f"Best tree error rate: {error:.4f}")
    
    print("Creating visualizations...")
    # Create tree structure visualization
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    plot_tree(best_tree, ax1)
    ax1.set_title("Decision Tree Structure")
    
    # Create decision boundary visualization
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(best_tree, X, y, ax2)
    ax2.set_title("Decision Boundary")
    
    plt.show(block=True)

if __name__ == "__main__":
    main()