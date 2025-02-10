import numpy as np

from scipy.sparse import csr_matrix

from src.app.utils import load_bio_bert_vectorizer, load_train_and_labels_file, load_tdidf_vectorizer


kb_type = "medic"
kb_location = "data/raw/mesh_data/medic/CTD_diseases.tsv"

label_filepath = "data/raw/mesh_data/medic/labels.txt"
training_filepath = "data/train/disease/train_Disease_100.txt"
    
vectorizer_directory = "data/processed/vectorizer"
onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"
onnx_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_cpu_sparse.npz"
vectorizer_old__filepath = "data/processed/vectorizer/vectorizer_old.pkl"

parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

# Dense Matrix
# Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]

tfidf_model = load_tdidf_vectorizer(vectorizer_old__filepath).model

X_train_feat = tfidf_model.transform(X_train)

print(X_train_feat)

onnx_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_cpu_sparse.npz"

X_train_feat = np.load(onnx_embeddigns_filepath)['embeddings']

print(csr_matrix(X_train_feat))