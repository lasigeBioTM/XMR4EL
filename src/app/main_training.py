import os
import time

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
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/biobert_onnx_dense.npz"
    vectorizer_filepath = "data/processed/vectorizer/vectorizer.pkl"

    start = time.time()

    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # Dense Matrix
    # Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
    X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]
    
    # See paths
    if os.path.exists(onnx_cpu_embeddigns_filepath):
        print(f"Path {onnx_cpu_embeddigns_filepath} does exists")
        X_train_feat = load_bio_bert_vectorizer(onnx_cpu_embeddigns_filepath)
    elif os.path.exists(onnx_gpu_embeddigns_filepath):
        print(f"Path {onnx_gpu_embeddigns_filepath} does exists")
        X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)
    else:
        print(f"Path does NOT exists")
        X_train_feat = create_bio_bert_vectorizer(corpus=X_train, 
                                                    output_embeddings_file=onnx_gpu_prefix_filepath,
                                                    directory_onnx_model=onnx_directory)
 
    
    "Get directorys"
    Y_train_feat = create_hierarchical_clustering(X_train_feat, save_directory="")
    
    top_k, top_k_score = create_hierarchical_linear_model(X_train_feat, Y_train_feat, 3, save_directory="")
    
    print(f"Top-{top_k} Score {top_k_score}")
    
    end = time.time()
    
    print(f"{end - start} secs of running")

if __name__ == "__main__":
    main()

