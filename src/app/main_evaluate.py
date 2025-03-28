import time
import numpy as np

from src.xmr.xmr_tree import XMRTree
from src.xmr.xmr_pipeline import XMRPipeline

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

    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"

    n_features = 12
    k = 3

    transformer_config = {
        "type": "biobert",
        "kwargs": {"batch_size": 500, "onnx_directory": onnx_directory},
    }

    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    tree_location = "data/saved_trees"

    # Read the file and extract unique names
    with open(file_test_input, "r") as file:
        unique_names = set(file.read().splitlines())

    name_list = sorted(unique_names)

    # embeddings tfdif -> clustering
    # transformer emb +  embeddings tfit -> classifier

    # train_disease_100
    trained_xtree = XMRTree.load(
        "data/saved_trees/XMRTree_2025-03-27_11-30-24-TRAIN_DATA"
    )

    # train_disease_100 + test_data == true_labels
    test_xtree = XMRTree.load("data/saved_trees/XMRTree_2025-03-27_11-39-26-TEST-DATA")

    """XMRTREE, 27-03, 10-25-01, Training with only training data"""

    true_labels = XMRPipeline.inference(
        test_xtree, name_list, transformer_config, n_features, k=1
    )
    true_labels = XMRPipeline.format_true_labels(true_labels)

    print(true_labels)

    predicted_labels = XMRPipeline.inference(
        trained_xtree, name_list, transformer_config, n_features, k=1
    )

    print(predicted_labels)

    top_k_scores = XMRPipeline.compute_top_k_accuracy(
        true_labels, predicted_labels, k=k
    )

    print(top_k_scores)

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()
