import os
import time

import numpy as np

from src.app.commandhelper import MainCommand
from src.app.utils import create_bio_bert_vectorizer, create_hierarchical_clustering, create_hierarchical_linear_model, load_bio_bert_vectorizer, load_hierarchical_clustering_model, load_hierarchical_linear_model, load_train_and_labels_file, predict_labels_hierarchical_clustering_model, predict_labels_hierarchical_linear_model

"""

    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,

"""
def main():
    args = MainCommand().run()
    kb_type = "medic"
    
    label_filepath_bc5cdr = "data/raw/mesh_data/bc5cdr/test_Disease.txt"
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/biobert_onnx_dense.npz"
    
    test_input_embeddings_filepath = "data/processed/vectorizer/text_input_embeddings.npy"

    hierarchical_clustering_model_filepath = "data/processed/clustering/hierarchical_clustering_model.pkl"
    hierarchical_linear_model_filepath = "data/processed/regression/hierarchical_linear_model.pkl"
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    
    start = time.time()
    
    with open(test_input_filepath, 'r') as test_input_file:
        test_input = [line.strip() for line in test_input_file]
    
    test_input = create_bio_bert_vectorizer(corpus=test_input, 
                                                    output_embeddings_file=test_input_embeddings_filepath,
                                                    directory_onnx_model=onnx_directory)
    
    hierarchical_clustering_model = load_hierarchical_clustering_model(hierarchical_clustering_model_filepath)
    
    predicted_labels = predict_labels_hierarchical_clustering_model(hierarchical_clustering_model, test_input)
    
    hierarchical_linear_model = load_hierarchical_linear_model(hierarchical_linear_model_filepath)
    
    k = 3
    
    top_k_score  = predict_labels_hierarchical_linear_model(hierarchical_linear_model, test_input, predicted_labels, k)
    
    # Overall Mean Top-1 Score: 0.6162762641906738
    print(f"Overall Mean Top-{k} Score: {top_k_score}")
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    

if __name__ == "__main__":
    main()