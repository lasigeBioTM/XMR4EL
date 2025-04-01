# import pytest  # noqa: F401; pylint: disable=unused-variable
# from pytest import approx

import os
import time

import numpy as np

from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.xmr_pipeline import XMRPipeline


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

    n_features = 12

    vectorizer_config = {"type": "tfidf", "kwargs": {"max_features": n_features}}
    
    transformer_config = {
        "type": "biobert",
        "kwargs": {"batch_size": 400, "onnx_directory": onnx_directory},
    }
    clustering_config = {
        "type": "sklearnminibatchkmeans",
        "kwargs": {"random_state": 0},
    }
    classifier_config = {
        "type": "sklearnlogisticregression",
        "kwargs": {"n_jobs": -1, "random_state": 0},
    }

    min_leaf_size = 10
    depth = 1

    training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")

    trn_corpus = Preprocessor.load_data_from_file(train_filepath=training_file)

    htree = XMRPipeline.execute_pipeline(
        trn_corpus,
        vectorizer_config,
        transformer_config,
        clustering_config,
        classifier_config,
        n_features,  # Number of Features
        min_leaf_size,
        depth,
        dtype=np.float32,
    )

    # Print the tree structure
    print(htree)

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
    htree.save(save_dir)

    end = time.time()
    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()
