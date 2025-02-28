import os
import unicodedata

from src.featurization.bert_vectorizer import BioBert
from src.app.utils import load_train_and_labels_file


def main():
    
    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # print(parsed_train_data['corpus'])
    
    def normalize_text(text):
        return unicodedata.normalize("NFKC", text).strip()
    
    with open(test_input_filepath, 'r') as test_input_file:
        test_input = [normalize_text(line) for line in test_input_file]

    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/testing_same_words.npz"
    
    biobert = BioBert.predict_cpu(trn_corpus=test_input, config={})


if __name__ == "__main__":
    main()
