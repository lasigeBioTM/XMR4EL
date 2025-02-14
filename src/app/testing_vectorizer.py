import os

from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_train_and_labels_file


def main():
    
    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_500.txt"
    
    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # Dense Matrix
    # Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
    X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease500.npz"
    
    
    X_train_feat = create_bio_bert_vectorizer(corpus=X_train, 
                                                output_embeddings_file=onnx_gpu_prefix_filepath,
                                                directory_onnx_model=onnx_directory)


if __name__ == "__main__":
    main()
