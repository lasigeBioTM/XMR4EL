import numpy as np

from scipy.sparse import csr_matrix

from src.app.utils import load_bio_bert_vectorizer

onnx_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_cpu_sparse.npz"

X_train_feat = np.load(onnx_embeddigns_filepath)['embeddings']

print(csr_matrix(X_train_feat))