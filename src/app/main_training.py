import time

from sklearn.decomposition import PCA

from models.classifier_wrapper.hierarchical_linear_model import HierarchicalLinearModel
from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU
from src.app.utils import create_bio_bert_vectorizer, create_hierarchical_clustering, create_hierarchical_linear_model, load_bio_bert_vectorizer, load_train_and_labels_file

"""
    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,
    * labels.txt -> 13292 labels,
"""
def main():
   
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease500_gpu.npy"
   
    start = time.time()

    # numpy.array (13240, 768)
    X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)

    # pca = PCA(n_components=2)
    # X_train_feat = pca.fit_transform(X_train_feat)
    
    print(f"Starting the Hierarchical Clustering Model")
    
    divisiveModel = DivisiveHierarchicalClustering.fit(
        X=X_train_feat,
        clustering_model_factory=KMeansCPU.create_model(),
        config={
            'n_splits': 0,
            'max_iter': 500,
            'depth': 2,
            'min_leaf_size':10,
            'min_clusters': 3,
            'init': 'k-means++',
            'random_state': 0,
            'spherical': True,
            }
    )
    
    print(f"Finished the Hierarchical Clustering Model\n")
    
    divisiveModel.save("data/processed/clustering/test_hierarchical_clustering_model.pkl")
    
    tree_node = divisiveModel.tree_node
    
    top_k = 3
    
    print(f"Starting Hierarchical Linear Model")
    
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
    
    print(f"Finished Hierarchical Linear Model\n")
    
    hierarchical_linear_model.save("data/processed/regression/test_hierarchical_linear_model.pkl")
    
    end = time.time()
    
    print(f"{end - start} secs of running")

if __name__ == "__main__":
    main()

