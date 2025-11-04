from argparse import ArgumentParser
import numpy as np
from collections import Counter
import time

from xmr4el.featurization.preprocessor import Preprocessor
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
        if label_list in allowed_set:
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

    parser = ArgumentParser()
    parser.add_argument("-xmodel_path", type=str, required=True)
    parser.add_argument("-test_path", type=str, required=True)
    
    args = parser.parse_args()

    start = time.time()

    load_path = args.xmodel_path
    
    print(load_path)
    
    trained_xtree = XModel.load(load_path)
    
    print(trained_xtree)
    
    test_set = Preprocessor.load_pubtator_file(args.test_path)
    
    corpus = test_set["corpus"]
    labels = test_set["labels"]
    
    # print(corpus[:1], len(corpus), type(corpus))
    # print(labels[:1], len(labels))
    
    # print(trained_xtree.initial_labels)
    
    golden_labels, input_texts = filter_labels_and_inputs(labels, corpus, trained_xtree.initial_labels)

    print(golden_labels[0], len(golden_labels))
    print(np.unique(np.array(golden_labels)).shape)
    # print(input_texts[0], len(input_texts))

    # exit()

    # Counter({0: 18131, 1: 1103})
    routes, score_csr = trained_xtree.predict(input_texts, 
                                              beam_size=10, 
                                              topk=100, 
                                              fusion="lp_fusion", 
                                              topk_mode="global", 
                                              topk_inside_global=100)
    
    # print(routes)
    print(score_csr)
    
    trained_labels = np.array(trained_xtree.initial_labels)
    
    hit_counts = []
    for r in routes:
        qi = r["query_index"]
        cand = set()
        cand.update(trained_labels[r.get("final_path").get("leaf_global_labels", [])])
        gold = golden_labels[qi]
        hit_counts.append(1 if gold in cand else 0)
        
    print("Hit counts per query:", Counter(hit_counts))
    print("Average hits:", np.mean(hit_counts))


    end = time.time()

    print(f"{end - start} secs of running")


if __name__ == "__main__":
    main()