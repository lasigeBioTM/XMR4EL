import os
import pickle

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min

# ONE HOT ENCODINGS

class HierarchicalLinearModel:
    """
    A model that combines clustering and logistic regression for hierarchical learning.
    It iteratively refines clusters and predicts probabilities with a logistic model.
    """

    def __init__(self, linear_model=None, linear_model_type=None, X_test=None, y_test=None, labels=None, top_k_score=None, top_k=None, gpu_usage=False):
        """
        Initializes the hierarchical model with clustering, logistic regression, and top-k accuracy score.

        Parameters:
        clustering_model (str): Type of clustering model (e.g., 'KMeansCPU')
        linear_model_type (str): Type of linear model (e.g., 'LogisticRegressionCPU')
        labels (array): Initial cluster labels
        top_k_score (float): Top-k accuracy score
        top_k (int): Number of top labels to consider
        """
        self.linear_model = linear_model
        self.linear_model_type = linear_model_type
        self.X_test = X_test
        self.y_test = y_test
        self.labels = labels
        self.top_k_score = top_k_score
        self.top_k = top_k
        self.gpu_usage = gpu_usage

    def save(self, directory):
        """
        Saves the trained model (clustering and linear models) to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        model_data = {
            'linear_model': self.linear_model,
            'linear_model_type': self.linear_model_type,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'labels': self.labels,
            'top_k_score': self.top_k_score,
            'top_k': self.top_k,
            'gpu_usage': self.gpu_usage
        }
        with open(directory, 'wb') as fout:
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
            linear_model=data['linear_model'],
            linear_model_type=data['linear_model_type'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            labels=data['labels'],
            top_k_score=data['top_k_score'],
            top_k=data['top_k'],
            gpu_usage=data['gpu_usage']
        )
    
    @classmethod
    def fit(cls, X, Y, LINEAR_MODEL, CLUSTERING_MODEL, top_k=3, top_k_threshold=0.9, min_cluster_size=10, max_cluster_size=50, gpu_usage=False):
        """
        Fits the model using clustering and logistic regression, and iteratively refines the cluster labels.
        """
        
         # Initialize one-hot encoder
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        Y_one_hot = one_hot_encoder.fit_transform(Y.reshape(-1, 1))  # Convert to one-hot encoding
        
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
        
        def encode_labels(labels_list):
            """
            Convert string labels into numeric labels
            """
            label_to_idx = {}
            return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
        
        def get_embeddings_by_label(X, Y, label):
            """
            Retrieves embeddings of a specific cluster label.
            """
            return X[Y == label], np.where(Y == label)[0]
        
        def calculate_iteration_count(num_classes):
            """
            Calculates the number of iterations based on the number of unique labels.
            """
            return 100 + (10 * num_classes)

        # Embeddings, Labels
        def linear_model_training(X, Y):
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

            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            # n_iter = calculate_iteration_count(Y)
            
            # Train Logistic Regression Model AUMENTAR O NUMERO DE INTERAÇÕES
            # Need find a way to calculate interactions, test "saga" and "newton-cg", get rid of lbfgs
            if gpu_usage:
                linear_model = LINEAR_MODEL.create_model({'max_iter': 100}).fit(X_train, y_train)
            else:
                linear_model = LINEAR_MODEL.create_model({'max_iter': 100, 'solver':'saga', 'penalty': 'l2'}).fit(X_train, y_train)
            
            # Predict probabilities
            y_proba = linear_model.predict_proba(X_test)
                
            # Get top-k predictions
            top_k_indices = get_top_k_indices(y_proba, top_k, top_k_threshold)
            
            # Compute top-k accuracy
            top_k_score = top_k_accuracy(y_test, y_proba, top_k)
            
            return linear_model, Y, X_test, y_test, top_k_indices, top_k_score
            
        def clustering_model_refining(X, Y, top_k_indices):
            # Update cluster labels
            new_labels = [None] * X.shape[0]
            for label in np.unique(Y):
                embeddings, indices = get_embeddings_by_label(X, Y, label)
                n_emb = embeddings.shape[0]

                if ((n_emb >= min_cluster_size and n_emb <= max_cluster_size) or n_emb >= max_cluster_size) and label in np.unique(top_k_indices):
                    n_iter = calculate_iteration_count(n_emb)
                    clustering_model = CLUSTERING_MODEL.create_model({'n_clusters': 2, 'max_iter': n_iter, 'random_state': 0}).fit(embeddings)
                
                    kmeans_labels = clustering_model.labels_

                    for idx, label in zip(indices, kmeans_labels):
                        new_labels[idx] = f"{label}{chr(65 + int(label))}"
                else:
                    for i in indices:
                        new_labels[i] = f"{label}"
            
            new_labels_encoded = np.array(encode_labels(new_labels))
            
            merged_labels = merge_small_clusters(X, new_labels_encoded, min_cluster_size)
            
            return merged_labels
        
        # Initial training
        linear_model, labels, X_test, y_test, top_k_indices, top_k_score = linear_model_training(X, Y)

        # Refine labels and improve top-k score        
        best_linear_model, best_labels, best_X_test, best_y_test, best_top_k_indices, best_top_k_score = linear_model, labels, X_test, y_test, top_k_indices, top_k_score
        
        while True:
            new_labels_encoded = clustering_model_refining(X, best_labels, best_top_k_indices)
            
            new_linear_model, new_labels, new_X_test, new_y_test, new_top_k_indices, new_top_k_score = linear_model_training(X, new_labels_encoded)
            
            if new_top_k_score > best_top_k_score:
                best_linear_model = new_linear_model
                best_labels = new_labels
                best_X_test = new_X_test
                best_y_test = new_y_test
                best_top_k_indices = new_top_k_indices
                best_top_k_score = new_top_k_score
            else:
                break

        return cls(
            linear_model=best_linear_model, 
            linear_model_type=LINEAR_MODEL.model_type, 
            labels=best_labels, 
            X_test=best_X_test,
            y_test=best_y_test,
            top_k_score=best_top_k_score, 
            top_k=top_k,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, labels, top_k):
        
        def top_k_score_sklearn(pred_probs, true_labels, k=3):
            """
            Calculate top-k score using sklearn's top_k_accuracy_score as a helper.
            
            Parameters:
            - pred_probs: Numpy array of predicted probabilities, shape (n_samples, n_classes)
            - true_labels: Array of true labels, shape (n_samples,)
            - k: Top-k predictions to consider for scoring
            
            Returns:
            - top_k_accuracy: The top-k accuracy using sklearn
            - average_top_k_score: The average sum of the top-k probabilities across all instances
            """
            # Calculate sklearn's top-k accuracy score
            top_k_accuracy = top_k_accuracy_score(true_labels, pred_probs, k=k, labels=np.arange(pred_probs.shape[1]))
            
            # Calculate the average top-k score (sum of probabilities for the top-k predictions)
            top_k_scores = []
            for probs in pred_probs:
                # Sort probabilities in descending order and sum the top-k
                top_k_scores.append(np.sum(np.sort(probs)[::-1][:k]))
            
            average_top_k_score = np.mean(top_k_scores)
            
            return top_k_accuracy, average_top_k_score
        
        preds = self.linear_model.predict_proba(embeddings)
        
        return top_k_score_sklearn(preds, labels, top_k)