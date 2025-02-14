import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU
from src.machine_learning.hierarchical_linear_model import HierarchicalLinearModel
from src.machine_learning.hierarchical_clustering import DivisiveHierarchicalClustering
from src.app.utils import load_bio_bert_vectorizer


def main():
    
    onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100_gpu.npy"
    divisive_model_filepath = "data/processed/clustering/test_hierarchical_clustering_model.pkl"
    
    # numpy.array (13240, 768)
    X_train_feat = load_bio_bert_vectorizer(onnx_gpu_embeddigns_filepath)
    
    pca = PCA(n_components=2)
    X_train_feat = pca.fit_transform(X_train_feat)
    
    divisiveModel = DivisiveHierarchicalClustering.load(divisive_model_filepath)
    
    y_train_feat = divisiveModel.labels
    
    print(y_train_feat.shape)
    
    top_k = 1
    
    hierarchical_linear_model = HierarchicalLinearModel.fit(
        X=X_train_feat,
        Y=y_train_feat,
        linear_model_factory=LogisticRegressionCPU.create_model(),
        clustering_model_factory=KMeansCPU.create_model(),
        config= {
            'max_iter': 1000,
            'min_leaf_size': 10,
            'max_leaf_size': 20,
            'random_state': 0,
            'top_k': top_k,
            'top_k_threshold': 0.9,
            'gpu_usage': False,
        }
    )
    
    hierarchical_linear_model.save("data/processed/regression/test_hierarchical_linear_model.pkl")
    
    X_test = hierarchical_linear_model.x_test
    y_test = hierarchical_linear_model.y_test
    
    linear_model = hierarchical_linear_model.linear_model
    
    # Step 5: Predict on the test set
    # cluster_labels_test = divisiveModel.predict(KMeansCPU.create_model(), X_test)  # Cluster test data
    # X_test_augmented = np.hstack((X_test, cluster_labels_test.reshape(-1, 1)))  # Combine cluster labels with features
    y_pred = linear_model.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')    
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1-Score: {f1}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    
    
if __name__ == "__main__":
    main()