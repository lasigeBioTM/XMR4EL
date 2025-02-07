import os

import numpy as np

from src.app.commandhelper import MainCommand
from src.app.utils import create_bio_bert_vectorizer, create_hierarchical_clustering, create_hierarchical_linear_model, load_bio_bert_vectorizer, load_train_and_labels_file

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
    kb_location = "data/raw/mesh_data/medic/CTD_diseases.tsv"

    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"
    
    vectorizer_directory = "data/processed/vectorizer"
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_embeddigns = "data/processed/vectorizer/biobert_onnx_cpu_embeddings.npy"
    vectorizer_filepath = "data/processed/vectorizer/vectorizer.pkl"

    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # Dense Matrix
    # Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
    X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]
    
    if not os.path.exists(vectorizer_filepath):
        X_train_feat = create_bio_bert_vectorizer(X_train, vectorizer_directory, onnx_directory, onnx_embeddigns)
    else:
        X_train_feat = load_bio_bert_vectorizer(vectorizer_filepath)
        
        
    Y_train_feat = create_hierarchical_clustering(X_train_feat)
    
    top_k, top_k_score = create_hierarchical_linear_model(X_train_feat, Y_train_feat, 3)
    
    print(f"Top-{top_k} Score {top_k_score}")

if __name__ == "__main__":
    main()

