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
    
    transformer_config = {'type': 'biobert', 'kwargs': {'batch_size': 400, 'onnx_directory': onnx_directory}}
    
    file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    
    # Read the file and extract unique names
    with open(file_test_input, "r") as file:
        unique_names = set(file.read().splitlines())
        
    name_list = list(unique_names)
    
    xtree = XMRTree.load()
    
    print(xtree)
    
    exit()
    
    predicted_labels = XMRPipeline.inference(xtree, name_list, transformer_config, n_features)
    
    save_predicted_labels(predicted_labels, filename="/xmr4el/predicted_labels.txt")
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    

def save_predicted_labels(predicted_labels, filename="predicted_labels.txt"):
    with open(filename, "w") as file:
        for labels in predicted_labels:
            if isinstance(labels, list):  # If nested list, join with commas
                file.write(",".join(map(str, labels)) + "\n")
            else:
                file.write(str(labels) + "\n")
    print(f"Predicted labels saved to {filename}")

if __name__ == "__main__":
    main()