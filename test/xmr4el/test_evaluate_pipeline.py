# import pytest  # noqa: F401; pylint: disable=unused-variable
# from pytest import approx
import numpy as np
from collections import Counter
import time

from xmr4el.predict.predict import Predict
from xmr4el.predict.skeleton_inference import SkeletonInference
from xmr4el.xmr.skeleton import Skeleton

from sklearn.metrics.pairwise import cosine_similarity

"""
    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,
"""

def read_codes_file(filepath):
    code_lists = []

    with open(filepath, 'r') as f:
        for line in f:
            # Strip whitespace and split by '|'
            codes = line.strip().split('|')
            if codes:
                code_lists.append(codes)

    return code_lists

def filter_labels_and_inputs(gold_labels, input_texts, allowed_labels):
    """
    Filters out gold_labels (list of lists) and corresponding input_texts
    where the first label in each gold label list is not in allowed_labels.

    Args:
        gold_labels (List[List[str]]): Nested list of gold labels.
        input_texts (List[str]): Raw input texts, aligned with gold_labels.
        allowed_labels (Iterable[str]): Set or list of valid labels.

    Returns:
        Tuple[List[List[str]], List[str]]: Filtered gold_labels and input_texts.
    """
    allowed_set = set(allowed_labels)

    filtered_labels = []
    filtered_texts = []

    for label_list, text in zip(gold_labels, input_texts):
        if label_list and label_list[0] in allowed_set:
            filtered_labels.append(label_list)
            filtered_texts.append(text)

    return filtered_labels, filtered_texts


def main():

    start = time.time()

    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"

    with open(file_test_input, "r") as file:
        input_texts = file.read().splitlines()

    # train_disease_100
    trained_xtree = Skeleton.load(
        "test/test_data/saved_trees/Skeleton_2025-07-19_15-20-54"
    )

    # exit()

    # print(trained_xtree)
    
    si = SkeletonInference(
        trained_xtree,
        trained_xtree.labels
    )
    
    gold_labels = read_codes_file("test/test_data/labels_bc5cdr_disease_medic.txt") # Need to filter out the ones that werent used.
    
    filtered_labels, filtered_texts = filter_labels_and_inputs(gold_labels, input_texts, trained_xtree.labels)
    
    # print(filtered_labels[0], filtered_texts[0])
    # print(filtered_labels[0], filtered_texts[0])
    # print(trained_xtree.dict_data[filtered_labels[0][0]])
    # for idx, _ in enumerate(filtered_labels):
    #     train_data_filtered_label_texts = [trained_xtree.train_data[idx] for idx in trained_xtree.dict_data[filtered_labels[idx][0]]]
    #     print(train_data_filtered_label_texts)
    # print(train_data_filtered_label_texts)
    
    print(filtered_labels)
    
    input_embs = si.generate_input_embeddigns(filtered_texts)
    
    
    
    # print(filtered_labels)
    # print(filtered_labels[0][0]) # 
    
    # print(input_embs[0])
    # print(trained_xtree.entity_centroids[filtered_labels[0][0]]) 
    
    for idx, _ in enumerate(filtered_labels):
        sim = cosine_similarity(input_embs[idx].reshape(1, -1), trained_xtree.entity_centroids[filtered_labels[idx][0]].reshape(1, -1))[0][0]
        print(sim)
    # exit()
    
    # print(sim)
    
    # exit()
    
    predicted_labels, hits = si.batch_inference(input_embs, filtered_labels, k=10)

    print(predicted_labels)
    
    print(Counter(hits))

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()