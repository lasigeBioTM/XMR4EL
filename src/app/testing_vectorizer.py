import os
import unicodedata

from src.featurization.bert_vectorizers import BertVectorizer
from src.featurization.vectorizers import Vectorizer
from src.featurization.preprocessor import Preprocessor


def main():
    
    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    parsed_train_data = Preprocessor.load_data_from_file(training_filepath, label_filepath)

    # print(parsed_train_data['corpus'])
    
    def normalize_text(text):
        return unicodedata.normalize("NFKC", text).strip()
    
    with open(test_input_filepath, 'r') as test_input_file:
        test_input = [normalize_text(line) for line in test_input_file]

    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
    onnx_cpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_cpu.npy"
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_gpu.npy"
    onnx_gpu_prefix_filepath = "data/processed/vectorizer/testing_same_words.npz"
    
    config = {'type': 'biobert', 'kwargs':{'onnx_directory': onnx_directory}}
    
    bert_vectorizer = BertVectorizer.train(trn_corpus=test_input[0:5], config=config)
    bert_vectorizer.save("test_vec")
    bert_vectorizer.load("test_vec")
    
    """
    config = {'type': 'tfidf', 'kwargs':{'max_features': 10}}
    
    vectorizer = Vectorizer.train(trn_corpus=test_input[0:5], config=config)
    vectorizer.save("test_vec")
    vectorizer.load("test_vec")
    """

if __name__ == "__main__":
    main()
