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

# debug_tables: list of DataFrames returned from predict(debug=True)
def save_debug_tables(debug_tables, filename_prefix="debug_folder/debug_file"):
    """
    Save all debug tables to separate CSV files.
    """
    for i, df in enumerate(debug_tables):
        # build filename with mention/layer index
        fname = f"{filename_prefix}_{i}.csv"
        # reset index to keep mention/layer info as a column
        df_reset = df.reset_index()
        df_reset.rename(columns={"index": "Mention_Layer"}, inplace=True)
        df_reset.to_csv(fname, index=False)
        # print(f"Saved debug table to {fname}")

def main():

    start = time.time()

    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"

    with open(file_test_input, "r") as file:
        input_texts = file.read().splitlines()

    # train_disease_100 # more open cluster better,
    #. 3 flag better than, more depth more score
    trained_xtree = XModel.load( # better 5
        "test/test_data/saved_trees/sgdclassifier_transformers" # 5 excluded
    )
    
    # print(trained_xtree.hierarchical_model.hmodel[0])
    
    gold_labels = read_codes_file("test/test_data/labels_bc5cdr_disease_medic.txt") # Need to filter out the ones that werent used.
    
    filtered_labels, filtered_texts = filter_labels_and_inputs(gold_labels, input_texts, trained_xtree.initial_labels)
    
    # print(filtered_texts[0])
    # print(filtered_labels[0]) # 25
    # exit()
    
    print(filtered_texts)
    
    score_matrix = trained_xtree.predict(filtered_texts)
    
    # print(score_matrix[0]["leaf_global_labels"])
    
    trained_labels = np.array(trained_xtree.initial_labels)
    
    # Get global label ids array from the score_matrix
    # global_labels = score_matrix.global_labels  # shape (n_labels,)

    hit_counts = []
    for idx in range(len(score_matrix)):
        
        # map to global label ids
        pred_labels = trained_labels[score_matrix[idx]["leaf_global_labels"]]

        # gold labels for this query
        gold = set(filtered_labels[idx])
        
        # print(f"Predicted Labels: {pred_labels}, with lenght: {len(pred_labels)}")
        # print(f"Golden Label: {gold}")
        
        hits = len(set(pred_labels).intersection(gold))
        hit_counts.append(hits)

    print("Hit counts per query:", Counter(hit_counts))
    print("Average hits:", np.mean(hit_counts))

    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()