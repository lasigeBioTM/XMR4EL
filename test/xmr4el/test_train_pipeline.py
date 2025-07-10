import os
import time
import copy

import numpy as np

from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.skeleton_builder import SkeletonBuilder


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

    min_leaf_size = 20
    depth = 3
    n_features = 768
    max_n_clusters = 6
    min_n_clusters = 2

    vectorizer_config = {
        "type": "tfidf", 
        "kwargs": {'max_features': 30000}
        }
    
    transformer_config = {
        # "type": "biobert",
        "type": "sentencetbiobert",
        "kwargs": {"batch_size": 3000}
    }
    
    """
    clustering_config = {
        "type": "sklearnminibatchkmeans",
        "kwargs": {
            "n_clusters": 8,  # This should be determined by your tuning process
            "init": "k-means++",
            "max_iter": 500,  # Increased from 300
            "batch_size": 0,  # Larger batch size for more stable updates
            "verbose": 0,
            "compute_labels": True,
            "random_state": 42,  # Fixed for reproducibility
            "tol": 1e-4,  # Added small tolerance for early stopping
            "max_no_improvement": 20,  # More patience for improvement
            "init_size": 24,  # 3 * n_clusters (3*8=24)
            "n_init": 5,  # Run multiple initializations, pick best
            "reassignment_ratio": 0.01,
        }
    }
    
    """
    clustering_config = {
    "type": "faisskmeans",  # Matches the registered name in your ClusterMeta system
    "kwargs": {
        "n_clusters": 6,           # Default cluster count (will be overridden by tuner)
        "max_iter": 300,           # Max iterations per run
        "nredo": 1,               # Number of initializations (FAISS calls this nredo)
        "gpu": False,               # Enable GPU acceleration
        "verbose": False,          # Disable progress prints
        "spherical": True,        # Set True for cosine similarity (L2 normalizes first)
        "seed": 42,                # Random seed (FAISS uses this for centroid init)
        "tol": 1e-4,               # Early stopping tolerance
        }
    }

    classifier_config = {
        "type": "sklearnlogisticregression",
        "kwargs": {
            "n_jobs": -1, 
            "random_state": 0,
            "penalty":"l2",           
            "C": 1.0,               
            "solver":"lbfgs",    
            "max_iter":1000
            },
    }
    
    """
    classifier_config = {
        "type": "lightgbmclassifier",
        "kwargs": {"random_state": 0}
    }
    """
    
    training_file = os.path.join(os.getcwd(), "data/train/disease/train_Disease_100.txt")
    labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
    
    train_data = Preprocessor().load_data_labels_from_file(
        train_filepath=training_file,
        labels_filepath=labels_file,
        # truncate_data=150
        )
    
    Y_train = train_data["labels_matrix"] # csr.matrix
    raw_labels = train_data["raw_labels"]
    X_train = train_data["corpus"] # List
    X_cross_train = train_data["cross_corpus"]
    label_enconder = train_data["label_encoder"]

    print(raw_labels[8240])
    print(X_cross_train[8240])
    
    print(raw_labels[9807])
    print(X_cross_train[9807])
    
    exit()

    # R_train = copy.deepcopy(Y_train)

    pipe = SkeletonBuilder(
        vectorizer_config,
        transformer_config, 
        clustering_config, 
        classifier_config, 
        n_features, 
        max_n_clusters, 
        min_n_clusters, 
        min_leaf_size, 
        depth, 
        dtype=np.float32)

    htree = pipe.execute(
        raw_labels,
        X_cross_train,
        X_train,
        Y_train,
    )

    # Print the tree structure
    print(htree)

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
    htree.save(save_dir)

    end = time.time()
    print(f"{end - start} secs of running")

# Here is code
if __name__ == "__main__":
    main()
