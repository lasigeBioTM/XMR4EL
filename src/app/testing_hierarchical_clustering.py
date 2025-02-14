from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.app.utils import load_bio_bert_vectorizer
from src.machine_learning.cpu.ml import KMeansCPU
from src.machine_learning.hierarchical_clustering import DivisiveHierarchicalClustering


def main():
    
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100_gpu.npy"
    
    # numpy.array (13240, 768)
    X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)
    
    # Retain 95% variance
    # 4 clusters, 0.85, Silhouette Score: 0.17800777, (13240, 85)
    # 4 clusters, 0.90, Silhouette Score: 0.16433217, (13240, 140)
    # 4 clusters, 0.95, Silhouette Score: 0.15200047, (13240, 252)
    pca = PCA(n_components=2)
    X_train_feat = pca.fit_transform(X_train_feat)
    
    print("Shape of the embeddings:", X_train_feat.shape)
    
    # Depth: 1 -> 7 clusters: 0.5758, 16 clusters: 0.5403 
    divisiveModel = DivisiveHierarchicalClustering.fit(
        X=X_train_feat,
        clustering_model_factory=KMeansCPU.create_model(),
        config={
            'n_splits': 7,
            'max_iter': 500,
            'depth': 1,
            'min_leaf_size':10,
            'max_leaf_size': 20,
            'init': 'k-means++',
            'random_state': 0,
            'spherical': True,
            'prefix': ''
            }
    )
    
    labels = divisiveModel.labels
    
    # print(labels)
    
    divisiveModel.save("data/processed/clustering/test_hierarchical_clustering_model.pkl")
    
    """
    # Dimensionality reduction (t-SNE)
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(X_train_feat)

    # Plot results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title("Clustering Results (2D t-SNE Visualization)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Save the plot to a file
    plt.savefig("y_plots/plot.png")
    """
    
if __name__ == "__main__":
    main()
