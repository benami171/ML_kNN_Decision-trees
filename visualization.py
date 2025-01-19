"""
Visualization utilities for decision trees and boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from models.decision_trees import Node

def plot_tree(node: Node, ax: Optional[plt.Axes] = None, x: float = 0.5, 
              y: float = 1.0, width: float = 1.0) -> plt.Axes:
    """
    Plot the hierarchical structure of a decision tree.
    
    Args:
        node: Root node of the tree
        ax: Matplotlib axes object
        x, y: Current node position
        width: Width available for the subtree
    
    Returns:
        plt.Axes: The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    circle = plt.Circle((x, y), 0.02, color='white', ec='black')
    ax.add_patch(circle)
    
    text = (f'Label: {node.label}' if node.label is not None 
            else f'X[{node.feature_idx}] ≤ {node.threshold:.2f}')
    ax.text(x, y + 0.02, text, ha='center', va='bottom')
    
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

def plot_decision_boundary(tree: Node, X: np.ndarray, y: np.ndarray, 
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot decision boundaries and data points.
    
    Args:
        tree: Trained decision tree
        X: Feature matrix
        y: Labels array
        ax: Matplotlib axes object
    
    Returns:
        plt.Axes: The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = np.array([tree.predict(np.array([x, y])) 
                 for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Versicolor')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Virginica')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.legend()
    
    return ax