import os
import pickle

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min


class HierarchicalLinearModel:
    """
    A model that combines clustering and logistic regression for hierarchical learning.
    It iteratively refines clusters and predicts probabilities with a logistic model.
    """

    def __init__(self, clustering_model_type=None, linear_model_type=None, labels=None, top_k_score=None, top_k=None, gpu_usage=False):
        """
        Initializes the hierarchical model with clustering, logistic regression, and top-k accuracy score.

        Parameters:
        clustering_model (str): Type of clustering model (e.g., 'KMeansCPU')
        linear_model_type (str): Type of linear model (e.g., 'LogisticRegressionCPU')
        labels (array): Initial cluster labels
        top_k_score (float): Top-k accuracy score
        top_k (int): Number of top labels to consider
        """
        self.clustering_model_type = clustering_model_type
        self.linear_model_type = linear_model_type
        self.labels = labels
        self.top_k_score = top_k_score
        self.top_k = top_k
        self.gpu_usage = False

    def save(self, directory):
        """
        Saves the trained model (clustering and linear models) to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        model_data = {
            'clustering_model_type': self.clustering_model_type,
            'linear_model_type': self.linear_model_type,
            'labels': self.labels,
            'top_k_score': self.top_k_score,
            'top_k': self.top_k
        }
        with open(os.path.join(directory, 'hierarchical_linear_model.pkl'), 'wb') as fout:
            pickle.dump(model_data, fout)

    @classmethod
    def load(cls, model_path):
        """
        Loads a previously saved model from the specified path.
        """
        assert os.path.exists(model_path), f"{model_path} does not exist"
        with open(model_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(
            clustering_model_type=data['clustering_model_type'], 
            linear_model_type=data['linear_model_type'],
            labels=data['labels'],
            top_k_score=data['top_k_score'],
            top_k=data['top_k']
        )
    
    @classmethod
    def fit(cls, X, Y, LINEAR_MODEL, CLUSTERING_MODEL, top_k=3, top_k_threshold=0.9, min_cluster_size=10, max_cluster_size=50):
        """
        Fits the model using clustering and logistic regression, and iteratively refines the cluster labels.
        """
        def merge_small_clusters(X, labels, min_cluster_size=10):
            """
            Merges small clusters with larger clusters.
            """
            unique_labels = np.unique(labels)
            valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
            centroids = np.array([
                X[labels == label].mean(axis=0) 
                for label in unique_labels
            ])
            
            updated_labels = labels.copy()
            for label in unique_labels:                
                cluster_indices = np.where(labels == label)[0]
                if len(cluster_indices) <= min_cluster_size:
                    cluster_points = X[cluster_indices]
                    distances = pairwise_distances_argmin_min(cluster_points, centroids)[1]
                    closest_centroid_idx = np.argmin(distances)
                    updated_labels[labels == label] = valid_labels[closest_centroid_idx]
                
            return updated_labels
    
        def recursive_top_k_ranking(X, Y):
            """
            Runs the model training pipeline (clustering and logistic regression).
            """
            def get_top_k_indices(y_proba, k, top_k_threshold):
                """
                Returns the top-k indices for each sample based on predicted probabilities.
                """
                filtered_proba = np.where(y_proba >= top_k_threshold, y_proba, -np.inf)
                top_k_indices = np.argsort(filtered_proba, axis=1)[:, -k:][:, ::-1]
                return top_k_indices.tolist()

            def top_k_accuracy(y_test, y_proba, k):
                """
                Calculates the top-k accuracy score.
                """
                return top_k_accuracy_score(y_test, y_proba, k=k, normalize=True)
            
            def get_embeddings_by_label(X, Y, label):
                """
                Retrieves embeddings of a specific cluster label.
                """
                return X[Y == label], np.where(Y == label)[0]
            
            def calculate_iteration_count(Y):
                """
                Calculates the number of iterations based on the number of unique labels.
                """
                num_classes = len(np.unique(Y))
                return 100 + (10 * num_classes)
            
            def encode_labels(labels_list):
                """Convert string labels into numeric labels"""
                label_to_idx = {}
                return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            n_iter = calculate_iteration_count(Y)
            
            # Train Logistic Regression Model AUMENTAR O NUMERO DE INTERAÇÕES
            # Need find a way to calculate interactions, test "saga" and "newton-cg", get rid of lbfgs
            if cls.gpu_usage:
                linear_model = LINEAR_MODEL.create_model({'max_iter': 100}).fit(X_train, y_train)
            else:
                linear_model = LINEAR_MODEL.create_model({'max_iter': 1000, 'solver':'saga', 'penalty': 'l2'}).fit(X_train, y_train)
            
            # Predict probabilities
            y_proba = linear_model.predict_proba(X_test)
                
            # Get top-k predictions
            top_k_indices = get_top_k_indices(y_proba, top_k, top_k_threshold)
            
            # Compute top-k accuracy
            top_k_score = top_k_accuracy(y_test, y_proba, top_k)
            
            # Update cluster labels
            new_labels = [None] * X.shape[0]
            for label in np.unique(Y):
                embeddings, indices = get_embeddings_by_label(X, Y, label)
                n_emb = embeddings.shape[0]

                if ((n_emb >= min_cluster_size and n_emb <= max_cluster_size) or n_emb >= max_cluster_size) and label in np.unique(top_k_indices):
                    n_iter = calculate_iteration_count(np.array([range(2)]))
                    clustering_model = CLUSTERING_MODEL.create_model({'n_clusters': 2, 'max_iter': n_iter, 'random_state': 0}).fit(embeddings)
                
                    kmeans_labels = clustering_model.labels_

                    for idx, label in zip(indices, kmeans_labels):
                        new_labels[idx] = f"{label}{chr(65 + int(label))}"
                else:
                    for i in indices:
                        new_labels[i] = f"{label}"
            
            new_labels_encoded = encode_labels(new_labels)
            
            return np.array(new_labels_encoded), top_k_score
        
        # Initial training
        labels, top_k_score = recursive_top_k_ranking(X, Y)

        # Refine labels and improve top-k score
        best_labels, best_top_k_score = labels, top_k_score
        labels = merge_small_clusters(X, labels, min_cluster_size)

        while True:
            new_labels, new_top_k_score = recursive_top_k_ranking(X, labels)
            if new_top_k_score > best_top_k_score:
                best_top_k_score = new_top_k_score
                best_labels = new_labels
                labels = merge_small_clusters(X, best_labels, min_cluster_size)
            else:
                break

        return cls("KMeansCPU", "LogisticRegressionCPU", best_labels, best_top_k_score, top_k)