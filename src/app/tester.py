from src.app.utils import load_bio_bert_vectorizer


onnx_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_cpu_embeddings.npy"

X_train_feat = load_bio_bert_vectorizer(onnx_embeddigns_filepath)

print(X_train_feat)