from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np

from src.app.utils import load_bio_bert_vectorizer
from src.machine_learning.cpu.ml import KMeansCPU
from src.machine_learning.divisive_hierarchical_clustering import DivisiveHierarchicalClustering


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
    
    # Depth: 1 -> 7 clusters: 0.5758, 16 clusters: 0.5403, 
    
    # 16 splits, disease100, Top-1 Accuracy (sklearn): 0.4586397058823529
    # 16 splits, disease500, Top-1 Accuracy (sklearn): 0.40487132352941174 
    # 16 splits, disease1000, Top-1 Accuracy (sklearn): 0.3766084558823529
    

    """
    k_range = range(2, 15)
    wcss = [KMeansCPU.create_model({'n_clusters': k, 'random_state':0}).fit(X_train_feat).inertia_ for k in k_range]

    knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
    optimal_clusters = knee.knee
    print(f"Optimal number of clusters: {optimal_clusters}")

    exit()
    """
    
    # 4 splits init, depth2, diease100, Top-1 Accuracy (sklearn): 0.31893382352941174
    # 4 splits, depth1, disease100, Top-1 Accuracy (sklearn): 0.6617647058823529
    
    """n_splits = 0 <- splits will be formed in an dynamic way"""
    divisiveModel = DivisiveHierarchicalClustering.fit(
        X=X_train_feat,
        clustering_model_factory=KMeansCPU.create_model(),
        config={
            'n_splits': 0,
            'max_iter': 500,
            'depth': 3,
            'min_leaf_size':20,
            'init': 'k-means++',
            'random_state': 0,
            'spherical': True,
            }
    )
    
    # labels = divisiveModel.labels
    
    # print(np.unique(labels))
    
    print(divisiveModel.tree_node)
    
    divisiveModel.save("data/processed/clustering/test_hierarchical_clustering_model.pkl")
    
    # Dimensionality reduction (t-SNE)

    """
    # Generate a custom color palette for up to 100 clusters
    custom_colors = sns.color_palette('husl', 100)  # Up to 100 unique colors
    cmap = ListedColormap(custom_colors)

    # Plot with the custom colormap
    plt.scatter(X_train_feat[:, 0], X_train_feat[:, 1], c=labels, cmap=cmap)
    plt.colorbar()
    
    # Save the plot to a file
    plt.savefig("y_plots/disease100_depth2_dynamic_clusters.png")
    """
    
if __name__ == "__main__":
    main()
