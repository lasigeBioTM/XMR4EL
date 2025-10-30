import os
import time

import platform

if platform.machine() == 'aarch64':  # ARM only
    os.environ['LD_PRELOAD'] = '/lib/aarch64-linux-gnu/libgomp.so.1'

# LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

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
        "kwargs": {'max_features': 30000}
        }
    
    transformer_config = {
        "type": "sentencetbiobert",
        "kwargs": {"batch_size": 3000}
    }
    
    dimension_config = {
        "type": "sklearntruncatedsvd", 
        "kwargs": {"n_components": 1500, 
               "algorithm": "randomized", 
               "n_iter": 5, 
               "n_oversamples": 10, 
               "power_iteration_normalizer": "auto", 
               "random_state": 42, 
               "tol": 0.0
        }
    }
    
    clustering_config = {
        "type": "balancedkmeans",
        "kwargs": {"n_clusters": 2,
                   "iter_limit": 400}
    }
    
    matcher_config = {
    "type": "sklearnsgdclassifier",
    "kwargs": {
        "loss": "log_loss",            # Equivalent to LogisticRegression (probabilistic)
        "penalty": "l1",               # Default for SGDClassifier; use 'l1' for sparsity
        "alpha": 0.0001,               # Inverse of regularization strength (C=1/alpha)
        "max_iter": 1000,              # Ensure convergence
        "tol": 1e-4,                   # Early stopping tolerance
        "class_weight": "balanced",          # Balanced classes assumed
        "n_jobs": -1,                  # Parallelize OvR (if multi-class)
        "random_state": 42,             # Reproducibility
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
            "penalty": "l1",               # Standard for ranking (use 'l1' for sparsity)
            "alpha": 0.0001,               # Stronger regularization (C=1/alpha)
            "max_iter": 1000,              # Ensure convergence
            "tol": 1e-4,
            # "class_weight": "balanced", # Better for some reason # Emphasize positives # "class_weight": "balanced",    # Critical for imbalanced ranking data
            "n_jobs": -1,                  # Parallelize OvR if multi-label
            "random_state": 42,
            "verbose": 0,
            "early_stopping": False,
            "learning_rate": "adaptive",   # Handles noisy gradients better
            "eta0": 0.01,                  # Higher initial learning rate for ranking
        }
    }
    
    cur_config = {
        "E_warm": 3,
        "ratios_warm": (0.70, 0.30),
        "ratios_hard": (0.60, 0.30, 0.10),
        "neg_mult": 1500,
        "seed": ranker_config["kwargs"]["random_state"],
    }
        
    train_data = Preprocessor.load_pubtator_file("datasets/MedMentions/st21pv/data/corpus_pubtator_test.txt")
    
    corpus = train_data["corpus"]
    labels = train_data["labels"]
    
    print(corpus[0], len(corpus))
    # print(labels[0], len(labels))
    
    del train_data
    
    min_leaf_size = 5
    max_leaf_size = 200
    cut_half_cluster=True
    ranker_every_layer=True
    depth = 4

    xmodel = XModel(vectorizer_config=vectorizer_config,
                    transformer_config=transformer_config,
                    dimension_config=dimension_config,
                    clustering_config=clustering_config,
                    matcher_config=matcher_config,
                    ranker_config=ranker_config,
                    cur_config=cur_config,
                    min_leaf_size=min_leaf_size,
                    max_leaf_size=max_leaf_size,
                    cut_half_cluster=cut_half_cluster,
                    ranker_every_layer=ranker_every_layer,
                    n_workers=-1,
                    depth=depth,
                    emb_flag=2,
                    verbose=2
                    )
    
    xmodel.train(corpus, labels)

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
    xmodel.save(save_dir)

    end = time.time()
    print(f"{end - start} secs of running")

# Here is code
if __name__ == "__main__":
    main()
