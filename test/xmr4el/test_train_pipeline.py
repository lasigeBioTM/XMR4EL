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

    # Must create an onnx_directory
    onnx_directory = "test/test_data/onnx_dir/model.onnx"

    start = time.time()

    min_leaf_size = 10
    depth = 3
    n_features = 100
    max_n_clusters = 16
    min_n_clusters = 6

    vectorizer_config = {
        "type": "tfidf", 
        "kwargs": {"max_features": n_features}
        }
    
    transformer_config = {
        "type": "biobert",
        "kwargs": {"batch_size": 400, "onnx_directory": onnx_directory}
    }
    
    clustering_config = {
        "type": "sklearnminibatchkmeans",
        "kwargs": {
            "random_state": 0, 
            "max_iter": 300
            },
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
    
    training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
    labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
    
    train_data = Preprocessor().load_data_labels_from_file(
        train_filepath=training_file,
        labels_filepath=labels_file,
        truncate_data=150
        )
    
    with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f]
    
    label_matrix, _ = Preprocessor().enconde_labels(labels)
    print(label_matrix, type(label_matrix))
    
    
    Y_train = train_data["labels_matrix"] # csr.matrix
    X_train = train_data["corpus"] # List
    label_enconder = train_data["label_encoder"]
    
    print(Y_train, type(Y_train))
    
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
