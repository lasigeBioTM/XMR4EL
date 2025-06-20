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

    min_leaf_size = 10
    depth = 1
    n_features = 768
    max_n_clusters = 6
    min_n_clusters = 2

    vectorizer_config = {
        "type": "tfidf", 
        "kwargs": {}
        }
    
    """
    tfidf = TfidfVectorizer(
        max_features=200000,  # Very large vocabulary
        min_df=2,            # Only include meaningful terms
        max_df=0.5,          # Filter out overly common terms
        ngram_range=(1, 3),  # Include unigrams, bigrams and trigrams
        analyzer='word',      # Word-level analysis
        sublinear_tf=True,    # Use log scaling
        use_idf=True,         # Use inverse document frequency
        smooth_idf=True,      # Smooth IDF weights
        lowercase=True,       # Case normalization
        stop_words='english'  # Remove stopwords
    ) 
    
    Why just dont make it to reduce features in tfidf ? 
    
    Use Truncate SVD ? 
    
    """
    
    transformer_config = {
        # "type": "biobert",
        "type": "sentencetbiobert",
        "kwargs": {"batch_size": 400}
    }
    
    """
    clustering_config = {
        "type": "sklearnminibatchkmeans",
        "kwargs": {
            "random_state": 0, 
            "max_iter": 300
            },
    }
    """
    
    clustering_config = {
    "type": "faisskmeans",  # Matches the registered name in your ClusterMeta system
    "kwargs": {
        "n_clusters": 8,           # Default cluster count (will be overridden by tuner)
        "max_iter": 300,           # Max iterations per run
        "nredo": 1,               # Number of initializations (FAISS calls this nredo)
        "gpu": True,               # Enable GPU acceleration
        "verbose": False,          # Disable progress prints
        "spherical": False,        # Set True for cosine similarity (L2 normalizes first)
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
    
    training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
    labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
    
    train_data = Preprocessor().load_data_labels_from_file(
        train_filepath=training_file,
        labels_filepath=labels_file,
        truncate_data=150
        )
    
    Y_train = train_data["labels_matrix"] # csr.matrix
    raw_labels = train_data["raw_labels"]
    X_train = train_data["corpus"] # List
    X_cross_train = train_data["cross_corpus"]
    label_enconder = train_data["label_encoder"]
        
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
