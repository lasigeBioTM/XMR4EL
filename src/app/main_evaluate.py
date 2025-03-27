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
    
    transformer_config = {'type': 'biobert', 'kwargs': {'batch_size': 500, 'onnx_directory': onnx_directory}}
    
    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    tree_location = "data/saved_trees"
    
    # Read the file and extract unique names
    with open(file_test_input, "r") as file:
        unique_names = set(file.read().splitlines())
        
    name_list = sorted(unique_names)
    
    xtree = XMRTree.load()
    
    
    
    # predicted_labels = XMRPipeline.inference(xtree, name_list[0:50], transformer_config, n_features, k=k)
    
    # print(predicted_labels)
    
    # Para gerar ground truth o modelo tem ver o test data, tem de ser processado um novo modelo
    
    # save_predicted_labels(predicted_labels, filename="predicted_labels.txt")
    
    end = time.time()
    
    print(f"{end - start} secs of running")

if __name__ == "__main__":
    main()