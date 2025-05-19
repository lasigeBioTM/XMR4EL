import os
import time
import copy

import numpy as np

from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.pipeline import XMRPipeline


"""
    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,
"""

def main():

    onnx_directory = "test/test_data/processed/vectorizer/biobert_onnx_cpu.onnx"

    start = time.time()

    n_features = 500

    vectorizer_config = {"type": "tfidf", "kwargs": {"max_features": n_features}}
    # vectorizer_config = {"type": "tfidf", "kwargs": {}}
    
    transformer_config = {
        "type": "biobert",
        "kwargs": {"batch_size": 400, "onnx_directory": onnx_directory},
    }
    
    # clustering_config = {
    #     "type": "sklearnminibatchkmeans",
    #     "kwargs": {"random_state": 0},
    # }
    
    clustering_config = {
        "type": "cumlkmeans",
        "kwargs": {"random_state": 0},
    }

    classifier_config = {
        "type": "sklearnlogisticregression",
        "kwargs": {"n_jobs": -1, "random_state": 0},
    }

    min_leaf_size = 10
    depth = 3

    training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
    labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
    
    train_data = Preprocessor().load_data_labels_from_file(
        train_filepath=training_file,
        labels_filepath=labels_file,
        truncate_data=16
        )
    
    Y_train = train_data["labels_matrix"] # csr.matrix
    X_train = train_data["corpus"] # List
    label_enconder = train_data["label_encoder"]
    
    R_train = copy.deepcopy(Y_train)

    htree = XMRPipeline.execute_pipeline(
        X_train,
        Y_train,
        label_enconder, # New
        vectorizer_config,
        transformer_config,
        clustering_config,
        classifier_config,
        n_features=n_features,  # Number of Features
        max_n_clusters=16,
        min_n_clusters=2, # Changed to 2, must be 6
        min_leaf_size=min_leaf_size,
        depth=depth,
        dtype=np.float32,
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
