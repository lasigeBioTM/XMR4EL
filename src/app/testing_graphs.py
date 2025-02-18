import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.app.utils import load_bio_bert_vectorizer


def main():
    
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100_gpu.npy"
    
    # numpy.array (13240, 768)
    X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)
    
    pca = PCA(n_components=2)
    X_train_feat = pca.fit_transform(X_train_feat)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_feat[:, 0], X_train_feat[:, 1], c='blue', alpha=0.6)
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Save the plot to a file
    plt.savefig("plot.png")
    
if __name__ == "__main__":
    main()
