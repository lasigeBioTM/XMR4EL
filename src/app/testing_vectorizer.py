import os

from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_train_and_labels_file


def main():
    
    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"
    
    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # Dense Matrix
    # Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
    X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/biobert_onnx_dense.npz"
    
    
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

    print(X_train_feat)

if __name__ == "__main__":
    main()
