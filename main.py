"""
Main script demonstrating the decision tree and k-NN implementations
on the Iris dataset binary classification task.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from data_utils import load_versicolor_virginica
from metrics import evaluate_model_parameters
from models.decision_trees import DecisionTreeBruteForce, DecisionTreeEntropy
from models.knn import evaluate_knn_parameters
from visualization import plot_tree, plot_decision_boundary

def run_decision_tree_comparison(X: np.ndarray, y: np.ndarray, k: int = 3) -> None:
    """
    Compare brute-force and entropy-based decision tree approaches.
    
    Args:
        X: Feature matrix
        y: Labels array
        k: Maximum tree depth
    """
    print("\nDecision Tree Analysis")
    print("=====================")
    
    # Brute-force approach
    print("\nTraining brute-force decision tree...")
    brute_force_tree = DecisionTreeBruteForce(max_depth=k)
    bf_root, bf_error = brute_force_tree.fit(X, y)
    print(f"Brute-force tree error rate: {bf_error:.4f}")
    
    # Entropy-based approach
    print("\nTraining entropy-based decision tree...")
    entropy_tree = DecisionTreeEntropy(max_depth=k)
    entropy_root, entropy_error = entropy_tree.fit(X, y)
    print(f"Entropy-based tree error rate: {entropy_error:.4f}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Tree structures
    plot_tree(bf_root, ax1)
    ax1.set_title("Brute-Force Tree Structure")
    
    plot_tree(entropy_root, ax2)
    ax2.set_title("Entropy-Based Tree Structure")
    
    # Decision boundaries
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_decision_boundary(bf_root, X, y, ax3)
    ax3.set_title("Brute-Force Decision Boundary")
    
    plot_decision_boundary(entropy_root, X, y, ax4)
    ax4.set_title("Entropy-Based Decision Boundary")
    
    plt.show()

def run_knn_analysis(X: np.ndarray, y: np.ndarray) -> None:
    """
    Perform k-NN analysis with various parameters.
    
    Args:
        X: Feature matrix
        y: Labels array
    """
    print("\nK-Nearest Neighbors Analysis")
    print("==========================")
    
    print("\nEvaluating k-NN with different parameters...")
    results = evaluate_knn_parameters(X, y)
    evaluate_model_parameters(results)

def main():
    """
    Main execution function that runs both decision tree and k-NN analyses.
    """
    print("Machine Learning Analysis on Iris Dataset")
    print("========================================")
    
    # Load and preprocess data
    print("\nLoading Versicolor-Virginica dataset...")
    X, y = load_versicolor_virginica()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run k-NN analysis
    run_knn_analysis(X, y)

    # Run decision tree analysis
    run_decision_tree_comparison(X, y, k=3)


if __name__ == "__main__":
    main()