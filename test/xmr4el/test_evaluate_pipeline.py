# import pytest  # noqa: F401; pylint: disable=unused-variable
# from pytest import approx

import time

from xmr4el.predict.predict import Predict
from xmr4el.xmr.skeleton import Skeleton

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

    onnx_directory = "test/test_data/onnx_dir/model.onnx"

    k = 5

    transformer_config = {
        "type": "biobert",
        "kwargs": {"batch_size": 500},
    }

    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"

    with open(file_test_input, "r") as file:
        unique_names = set(file.read().splitlines())

    name_list = sorted(unique_names)

    # train_disease_100
    trained_xtree = Skeleton.load(
        "test/test_data/saved_trees/Skeleton_2025-05-29_12-53-44"
    )

    print(trained_xtree)

    predicted_labels = Predict.inference(
        trained_xtree, name_list, transformer_config, k=k
    )

    print(predicted_labels)

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()