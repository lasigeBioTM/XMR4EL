import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min


class HierarchicalLinearModel:
    """
    A hierarchical learning model that combines clustering and linear models.
    Iteratively refines clusters and predicts probabilities using a linear model.
    """

    def __init__(self, linear_model=None, linear_model_type=None, labels=None,
                 top_k_score=None, top_k=3, gpu_usage=False):
        """Initialization"""
        self.linear_model = linear_model
        self.linear_model_type = linear_model_type
        self.labels = labels
        self.top_k_score = top_k_score
        self.top_k = top_k
        self.gpu_usage = gpu_usage

    def save(self, hierarchical_folder):
        """Save the model and its parameters to a file """
        os.makedirs(os.path.dirname(hierarchical_folder), exist_ok=True)
        with open(hierarchical_folder, 'wb') as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, hierarchical_folder):
        """Load a saved model from a file."""
        if not os.path.exists(hierarchical_folder):
            raise FileNotFoundError(f"Hierarchical folder {hierarchical_folder} does not exist.")
        
        with open(hierarchical_folder, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def fit(cls, X, Y, linear_model_factory, clustering_model_factory, 
            top_k=3, top_k_threshold=0.9, min_cluster_size=10, max_cluster_size=50, gpu_usage=False):
        """
        Train the model using clustering and logistic regression with iterative label refinement.
        """
        # Initial training
        linear_model, labels, top_k_indices, top_k_score = cls.__train_linear_model(X, Y, linear_model_factory,
                                                                                    top_k, top_k_threshold, gpu_usage=gpu_usage)

        # Refine labels and improve top-k score        
        best_linear_model, best_labels, best_top_k_indices, best_top_k_score = linear_model, labels, top_k_indices, top_k_score
        
        while True:
            new_labels_encoded = cls.__refine_clusters(X, best_labels, clustering_model_factory, best_top_k_indices,
                                                       min_cluster_size, max_cluster_size, gpu_usage=gpu_usage)
            
            new_linear_model, new_labels, new_top_k_indices, new_top_k_score = cls.__train_linear_model(X, new_labels_encoded, linear_model_factory, 
                                                                                                        top_k, top_k_threshold, gpu_usage=gpu_usage)
            
            if new_top_k_score > best_top_k_score:
                best_linear_model = new_linear_model
                best_labels = new_labels
                best_top_k_indices = new_top_k_indices
                best_top_k_score = new_top_k_score
            else:
                break

        return cls(
            linear_model=best_linear_model, 
            linear_model_type=linear_model_factory.model_type, 
            labels=best_labels, 
            top_k_score=best_top_k_score, 
            top_k=top_k,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, labels, top_k):        
        preds = self.linear_model.predict_proba(embeddings)
        return self.__top_k_score_sklearn(preds, labels, top_k)

    # Embeddings, Labels
    @classmethod
    def __train_linear_model(cls, X, Y, linear_model_factory, top_k, top_k_threshold, gpu_usage=False):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            
        # Need find a way to calculate interactions, test "saga" and "newton-cg", get rid of lbfgs
        if gpu_usage:
            linear_model = linear_model_factory.create_model({'max_iter': 1000}).fit(X_train, y_train)
        else:
            linear_model = linear_model_factory.create_model({'max_iter': 100, 'solver':'saga', 'penalty': 'l2'}).fit(X_train, y_train)
            
        # Predict probabilities
        y_proba = linear_model.predict_proba(X_test)
                
        # Get top-k predictions
        top_k_indices = cls.__get_top_k_indices(y_proba, top_k, top_k_threshold)
            
        # Compute top-k accuracy
        top_k_score = top_k_accuracy_score(y_test, y_proba, k=top_k, normalize=True)
            
        return linear_model, Y, top_k_indices, top_k_score
    
    @classmethod
    def __refine_clusters(cls, X, labels, clustering_model_factory, 
                                    top_k_indices, min_cluster_size, max_cluster_size, 
                                    gpu_usage=False):
        # Update cluster labels
        new_labels = np.empty_like(labels, dtype=object)
            
        for label in np.unique(labels):
                
            embeddings =  X[labels == label]
            n_emb = embeddings.shape[0]
            indices = np.where(labels == label)[0]

            if ((n_emb >= min_cluster_size and n_emb <= max_cluster_size) or n_emb >= max_cluster_size) and label in np.unique(top_k_indices):
                    
                n_iter = 100 + (10 * n_emb)
                clustering_model = clustering_model_factory.create_model({'n_clusters': 2, 'max_iter': n_iter, 'random_state': 0}).fit(embeddings)
                
                kmeans_labels = clustering_model.labels_

                for idx, label in zip(indices, kmeans_labels):
                    new_labels[idx] = f"{label}{chr(65 + int(label))}"
                else:
                    for i in indices:
                        new_labels[i] = f"{label}"
            
        new_labels_encoded = np.array(cls.__encode_labels(new_labels))
            
        merged_labels = cls.__merge_small_clusters(X, new_labels_encoded, min_cluster_size)
            
        return merged_labels

    @staticmethod
    def __merge_small_clusters(X, labels, min_cluster_size=10):
        """Merges small clusters with larger clusters."""
        unique_labels = np.unique(labels)
        valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
            
        updated_labels = labels.copy()
        for label in unique_labels:                
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) <= min_cluster_size:
                cluster_points = X[cluster_indices]
                _, closest_centroid_idx = pairwise_distances_argmin_min(cluster_points, centroids)
                updated_labels[labels == label] = valid_labels[closest_centroid_idx[0]]
                
            return updated_labels
        
    @staticmethod    
    def __encode_labels(labels_list):
        """Convert string labels into numeric labels."""
        label_to_idx = {}
        return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
    
    @staticmethod
    def __get_top_k_indices(y_proba, k, top_k_threshold):
        """Returns the top-k indices for each sample based on predicted probabilities."""
        filtered_proba = np.where(y_proba >= top_k_threshold, y_proba, -np.inf)
        top_k_indices = np.argsort(filtered_proba, axis=1)[:, -k:][:, ::-1]
        return top_k_indices.tolist()
    
    @staticmethod
    def __top_k_score_sklearn(pred_probs, true_labels, k=3):
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
                
        
        print(top_k_scores)
        
        average_top_k_score = np.mean(top_k_scores)
            
        return top_k_accuracy, average_top_k_score