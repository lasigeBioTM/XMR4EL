# import pytest  # noqa: F401; pylint: disable=unused-variable
# from pytest import approx

import time

from xmr4el.predict.predict import Predict
from xmr4el.predict.skeleton_inference import SkeletonInference
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

    k = 5

    transformer_config = {
        "type": "sentencetsapbert",
        "kwargs": {"batch_size": 400}
    }

    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"

    with open(file_test_input, "r") as file:
        unique_names = set(file.read().splitlines())

    input_text = sorted(unique_names)

    # train_disease_100
    trained_xtree = Skeleton.load(
        "test/test_data/saved_trees/Skeleton_2025-07-17_15-03-58"
    )

    # exit()

    print(trained_xtree)
    
    si = SkeletonInference(
        trained_xtree,
        trained_xtree.labels
    )
    
    with open("test/test_data/labels_bc5cdr_disease_medic.txt", 'r') as gold_labels_f:
        gold_labels = gold_labels_f.read()
    
    input_embs = si.transform_input_text(input_text)
    
    predicted_labels, hits = si.batch_inference(input_embs[:10], gold_labels[:10])

    print(predicted_labels)
    
    print(hits)

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()