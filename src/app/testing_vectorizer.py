import os
import unicodedata

from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_train_and_labels_file


def main():
    
    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"
    
    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # print(parsed_train_data['corpus'])
    
    def normalize_text(text):
        return unicodedata.normalize("NFKC", text).strip()
    
    X_train = [" | ".join([normalize_text(str(text)) for text in sublist]) 
            for sublist in parsed_train_data["corpus"]]
    
    print(X_train)

    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100.npz"
    
    
    X_train_feat = create_bio_bert_vectorizer(corpus=X_train, 
                                                output_embeddings_file=onnx_gpu_prefix_filepath,
                                                directory_onnx_model=onnx_directory)


if __name__ == "__main__":
    main()
