# import pytest  # noqa: F401; pylint: disable=unused-variable
# from pytest import approx
import numpy as np
from collections import Counter
import time

from xmr4el.xmr.model import XModel


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

    # train_disease_100 # more open cluster better,
    #. 3 flag better than, more depth more score
    trained_xtree = XModel.load( # better 5
        "test/test_data/saved_trees/xmodel_depth4_emb_flag_3_lgbm" # 5 excluded
    )
    
    # print(trained_xtree.hierarchical_model.hmodel[0])
    
    gold_labels = read_codes_file("test/test_data/labels_bc5cdr_disease_medic.txt") # Need to filter out the ones that werent used.
    
    
    # print(trained_xtree.initial_labels)
    # exit()
    
    filtered_labels, filtered_texts = filter_labels_and_inputs(gold_labels, input_texts, trained_xtree.initial_labels)
    
    # print(filtered_labels)
    
    # input_embs = sp.generate_input_embeddigns(filtered_texts)
    
    # print(filtered_labels)
    # print(filtered_labels[0][0]) # 
    
    # print(input_embs[0])
    # print(trained_xtree.entity_centroids[filtered_labels[0][0]]) 
    
    # topk50 - 6 - topk100 - 8
    # topk50 - 6 - topk100 - 6, topk200 - 38
    
    # flag 1 - topk10 - 0, topk100 - 9, topk200 - 42, after normalizing, topk100 - 32, topk200 - 40, 500 negatives
    # flag 2 - topk10 - 0, topk100 - 10, topk200 - 29
    # flag 3 - topk10 - 0, topk100 - 8, topk200 - 36
    
    predicted_labels, hits = trained_xtree.predict(filtered_texts, filtered_labels, topk=50, beam_size=10)

    print(predicted_labels)
    
    print(hits)
    
    found_ratio = []
    matcher_found_ratio = []
    for found, _, matcher_found, _ in hits:
        if found == 1:
            found_ratio.append(1)
        else:
            found_ratio.append(0)
            
        if matcher_found:
            matcher_found_ratio.append(1)
        else:
            matcher_found_ratio.append(0)
            
    print("Found Ratio", Counter(found_ratio), "Matcher Ratio", Counter(matcher_found_ratio))

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()