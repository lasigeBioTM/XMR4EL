import os
import unicodedata

from src.featurization.vectorizer import BioBertVectorizer
from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_train_and_labels_file


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
    
    corpus, sorted_embeddings = BioBertVectorizer.predict_cpu_original_index(corpus=test_input, 
                                                output_prefix=onnx_gpu_prefix_filepath,
                                                directory=onnx_directory)
    
    for text, emb in zip(corpus, sorted_embeddings):
        print(f"Text: {text}\nEmbedding: {emb[:5]}...")  # Print first 5 values for readability
        print("-" * 50)


if __name__ == "__main__":
    main()
