import numpy as np

from sklearn.decomposition import PCA

from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU
from src.machine_learning.hierarchical_linear_model import HierarchicalLinearModel
from src.machine_learning.divisive_hierarchical_clustering import DivisiveHierarchicalClustering
from src.app.utils import load_bio_bert_vectorizer


def main():
    
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100_gpu.npy"
    divisive_model_filepath = "data/processed/clustering/test_hierarchical_clustering_model.pkl"
    
    # numpy.array (13240, 768)
    X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)
    
    pca = PCA(n_components=2)
    X_train_feat = pca.fit_transform(X_train_feat)
    
    divisiveModel = DivisiveHierarchicalClustering.load(divisive_model_filepath)
    
    tree_node = divisiveModel.tree_node
    
    top_k = 1
    
    hierarchical_linear_model = HierarchicalLinearModel.fit(
        tree_node=tree_node,
        linear_model_factory=LogisticRegressionCPU.create_model(
            {'max_iter': 1000,
             'solver': 'newton-cg',
             'penalty': 'l2',
             'random_state': 0
             }),
        config= {
            'min_leaf_size': 20,
            'max_leaf_size': 40,
            'top_k': top_k,
            'top_k_threshold': 0.15,
            'gpu_usage': False,
        }
    )
    
    print(hierarchical_linear_model.tree_node)
    
    hierarchical_linear_model.save("data/processed/regression/test_hierarchical_linear_model.pkl")
    
    # print(hierarchical_linear_model.tree_node.print_tree(cluster=False))
    
if __name__ == "__main__":
    main()