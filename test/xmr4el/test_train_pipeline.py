import os
import time

import platform

if platform.machine() == 'aarch64':  # ARM only
    os.environ['LD_PRELOAD'] = '/lib/aarch64-linux-gnu/libgomp.so.1'

# LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
import numpy as np

from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.model import XModel


"""
    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,
"""

def main():
    
    start = time.time()

    vectorizer_config = {
        "type": "tfidf", 
        "kwargs": {'max_features': 20000}
        }
    
    transformer_config = {
        "type": "sentencetbiobert",
        "kwargs": {"batch_size": 3000}
    }
    
    """
    clustering_config = {
        "type": "sklearnminibatchkmeans",
        "kwargs": {
            "n_clusters": 2,  # This should be determined by your tuning process
            "init": "k-means++",
            "max_iter": 500,  # Increased from 300
            "batch_size": 0,  # Larger batch size for more stable updates
            "verbose": 0,
            "compute_labels": True,
            "random_state": 42,  # Fixed for reproducibility
            "tol": 1e-4,  # Added small tolerance for early stopping
            "max_no_improvement": 20,  # More patience for improvement
            "init_size": 2*3,  # 3 * n_clusters (3*8=24)
            "n_init": 5,  # Run multiple initializations, pick best
            "reassignment_ratio": 0.01,
        }
    }
    
    """
    
    clustering_config = {
        "type": "balancedkmeans",
        "kwargs": {"n_clusters": 4,
                   "iter_limit": 400}
    }
    
    """
    clustering_config = {
    "type": "faisskmeans",  # Matches the registered name in your ClusterMeta system
    "kwargs": {
        "n_clusters": 2,           # Default cluster count (will be overridden by tuner)
        "max_iter": 500,           # Max iterations per run
        "nredo": 1,               # Number of initializations (FAISS calls this nredo)
        "gpu": False,               # Enable GPU acceleration
        "verbose": False,          # Disable progress prints
        "spherical": True,        # Set True for cosine similarity (L2 normalizes first)
        "seed": 42,                # Random seed (FAISS uses this for centroid init)
        "tol": 1e-4,               # Early stopping tolerance
        }
    }
    """
    """
    classifier_config = {
        "type": "sklearnlogisticregression",
        "kwargs": {
            "solver": "liblinear",
            "n_jobs": -1, 
            "random_state": 0,
            "penalty":"l2",           
            "C": 1.0,               
            "solver":"lbfgs",    
            "max_iter":1000,
            "verbose": 1
            },
    }
    """
    
    """
    matcher_config = {
        "type": "lightgbmclassifier",
        "kwargs": {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 300,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }
    }
    """
    
    matcher_config = {
    "type": "sklearnsgdclassifier",
    "kwargs": {
        "loss": "log_loss",            # Equivalent to LogisticRegression (probabilistic)
        "penalty": "l2",               # Default for SGDClassifier; use 'l1' for sparsity
        "alpha": 0.0001,               # Inverse of regularization strength (C=1/alpha)
        "max_iter": 1000,              # Ensure convergence
        "tol": 1e-4,                   # Early stopping tolerance
        "class_weight": "balanced",          # Balanced classes assumed
        "n_jobs": -1,                  # Parallelize OvR (if multi-class)
        "random_state": 0,             # Reproducibility
        "verbose": 0,
        "early_stopping": True,        # Stop if validation score plateaus
        "learning_rate": "optimal",    # Auto-adjusts step size
        "eta0": 0.0,                   # Initial learning rate (ignored if 'optimal')
        }
    }
    
    ranker_config = {
        "type": "sklearnsgdclassifier",
        "kwargs": {
            "loss": "hinge",               # Margin loss (like SVM; no probabilities)
            "penalty": "l2",               # Standard for ranking (use 'l1' for sparsity)
            "alpha": 0.0001,               # Stronger regularization (C=1/alpha)
            "max_iter": 1000,              # Ensure convergence
            "tol": 1e-4,
            "class_weight": "balanced", # Better for some reason # Emphasize positives # "class_weight": "balanced",    # Critical for imbalanced ranking data
            "n_jobs": -1,                  # Parallelize OvR if multi-label
            "random_state": 0,
            "verbose": 0,
            "early_stopping": True,
            "learning_rate": "adaptive",   # Handles noisy gradients better
            "eta0": 0.01,                  # Higher initial learning rate for ranking
        }
    }
    
    """
    ranker_config = {
        "type": "lightgbmclassifier",
        "kwargs": {
            "boosting_type": "gbdt",
            "objective": "binary",              # REQUIRED for OneVsRest
            "device": "cpu",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 7,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
            "force_col_wise": True  # Faster for sparse
        }
    }
    """
    
    training_file = os.path.join(os.getcwd(), "data/train/disease/train_Disease_100.txt")
    labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
    
    train_data = Preprocessor().load_data_labels_from_file(
        train_filepath=training_file,
        labels_filepath=labels_file,
        truncate_data=400
        )
    
    raw_labels = train_data["labels"]
    X_cross_train = train_data["corpus"] # list of lists
    
    min_leaf_size = 5
    max_leaf_size = 200
    cut_half_cluster=True
    ranker_every_layer=False
    depth = 2

    xmodel = XModel(vectorizer_config=vectorizer_config,
                    transformer_config=transformer_config,
                    dimension_config=None,
                    clustering_config=clustering_config,
                    matcher_config=matcher_config,
                    ranker_config=ranker_config,
                    min_leaf_size=min_leaf_size,
                    max_leaf_size=max_leaf_size,
                    cut_half_cluster=cut_half_cluster,
                    ranker_every_layer=ranker_every_layer,
                    n_workers=8,
                    depth=depth,
                    emb_flag=3
                    )
    
    xmodel.train(X_cross_train, raw_labels)

    # Print the tree structure
    # print(htree)

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
    xmodel.save(save_dir)

    end = time.time()
    print(f"{end - start} secs of running")

# Here is code
if __name__ == "__main__":
    main()
