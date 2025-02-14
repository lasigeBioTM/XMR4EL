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
                 top_k_score=None, top_k=3, x_test=None, y_test=None, gpu_usage=False):
        """Initialization"""
        self.linear_model = linear_model
        self.linear_model_type = linear_model_type
        self.labels = labels
        self.top_k_score = top_k_score
        self.top_k = top_k
        self.x_test = x_test
        self.y_test = y_test
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
    def fit(cls, X, Y, linear_model_factory, clustering_model_factory, config={}):
        """
        Train the model using clustering and logistic regression with iterative label refinement.
        """
        
        DEFAULTS = {
            'max_iter': 100,
            'min_leaf_size': 10,
            'max_leaf_size': 20,
            'random_state': 0,
            'top_k': 3,
            'top_k_threshold': 0.9,
            'gpu_usage': False,
        }
        
                
        config = {**DEFAULTS, **config}
        
        max_iter = config['max_iter']
        min_leaf_size = config['min_leaf_size']
        max_leaf_size = config['max_leaf_size']
        random_state = config['random_state']
        top_k = config['top_k']
        top_k_threshold = config['top_k_threshold']
        gpu_usage = config['gpu_usage']
        
        
        # Initial training
        linear_model, labels, top_k_indices, top_k_score, x_test, y_test = cls.__train_linear_model(
            X, 
            Y, 
            linear_model_factory, 
            max_iter, 
            random_state,
            top_k, 
            top_k_threshold, 
            gpu_usage=gpu_usage
        )

        # Refine labels and improve top-k score        
        best_linear_model, best_labels, best_top_k_indices, best_top_k_score, best_x_test, best_y_test = linear_model, labels, top_k_indices, top_k_score, x_test, y_test
        
        print(f"BEST LABELS: {best_labels}, LEN: {len(np.unique(best_labels))}")
        
        while True:
            new_labels_encoded = cls.__refine_clusters(X, best_labels, clustering_model_factory, best_top_k_indices,
                                                       min_leaf_size, max_leaf_size, random_state, gpu_usage=gpu_usage)
            
            print(f"LABELS ENCODED: {new_labels_encoded}, LEN: {len(np.unique(new_labels_encoded))}")
            
            new_linear_model, new_labels, new_top_k_indices, new_top_k_score, new_x_test, new_y_test = cls.__train_linear_model(
                X, 
                new_labels_encoded, 
                linear_model_factory, 
                max_iter,
                random_state,
                top_k, 
                top_k_threshold, 
                gpu_usage=gpu_usage
            )
            
            if new_top_k_score > best_top_k_score:
                best_linear_model = new_linear_model
                best_labels = new_labels
                best_top_k_indices = new_top_k_indices
                best_top_k_score = new_top_k_score
                best_x_test = new_x_test
                best_y_test = new_y_test
            else:
                break

        return cls(
            linear_model=best_linear_model, 
            linear_model_type=linear_model_factory.model_type, 
            labels=best_labels, 
            top_k_score=best_top_k_score, 
            top_k=top_k,
            x_test = best_x_test,
            y_test = best_y_test,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, predicted_labels, class_labels, top_k):        
        preds = self.linear_model.predict_proba(embeddings)
        return self.__top_k_score_sklearn(preds, predicted_labels, class_labels, top_k)

    # Embeddings, Labels
    @classmethod
    def __train_linear_model(cls, X, Y, linear_model_factory, max_iter, random_state, top_k, top_k_threshold, gpu_usage):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            
        print("Inside trainign linear model")
        print(Y, np.unique(Y))    
            
        # Need find a way to calculate interactions, test "saga" and "newton-cg", get rid of lbfgs
        if gpu_usage:
            linear_model = linear_model_factory.create_model({'max_iter': max_iter, 'random_state': random_state}).fit(X_train, y_train)
        else:
            linear_model = linear_model_factory.create_model({'max_iter': max_iter, 'solver':'newton-cg', 'penalty': 'l2', 'random_state': random_state}).fit(X_train, y_train)
            
        # Predict probabilities
        y_proba = linear_model.predict_proba(X_test)
        
        # Compute top-k accuracy
        top_k_score = top_k_accuracy_score(y_test, y_proba, k=top_k, normalize=True)       

        # Get top-k predictions
        top_k_indices = cls.__get_top_k_indices(y_proba, top_k, top_k_threshold)
        
            
        return linear_model, Y, top_k_indices, top_k_score, X_test, y_test
    
    @classmethod
    def __refine_clusters(cls, X, labels, clustering_model_factory, 
                      top_k_indices, min_cluster_size, max_cluster_size, 
                      random_state, gpu_usage=False):
        # Initialize new labels with the original labels
        new_labels = np.array(labels, dtype=object)
        
        for label in np.unique(labels):
            # print("LABEL:", label)
            
            indices = np.where(labels == label)[0]
            embeddings = X[indices]
            n_emb = embeddings.shape[0]
            # print(f"Number of embeddings for label {label}: {n_emb}")
            
            if ((n_emb >= min_cluster_size and n_emb <= max_cluster_size) or n_emb >= max_cluster_size) and label in np.unique(top_k_indices):
                # print("INSIDE CLUSTERING LOGIC")
                
                n_iter = 100 + (10 * n_emb)
                n_clusters = min(2, n_emb)  # Ensure at least 1 cluster
                clustering_model = clustering_model_factory.create_model({'n_clusters': n_clusters, 'max_iter': n_iter, 'random_state': random_state}).fit(embeddings)
                
                kmeans_labels = clustering_model.labels_
                for idx, cluster_label in zip(indices, kmeans_labels):
                    new_labels[idx] = f"{label}{chr(65 + cluster_label)}"
            else:
                for idx in indices:
                    new_labels[idx] = f"{label}"
        
        # print(f"LABELS ENCODED: {new_labels[:10]}... (showing 10 out of {len(new_labels)})")  
        
        new_labels_encoded = np.array(cls.__encode_labels(new_labels))
        # print(f"LABELS ENCODED: {new_labels_encoded[:10]}... (showing 10 out of {len(new_labels_encoded)})")
        
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
                
                updated_labels[labels == label] = valid_labels[int(closest_centroid_idx[0])]
                
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
    def __top_k_score_sklearn(pred_probs, true_labels, class_labels, k=3):
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
            
        print(len(top_k_scores))
        
        average_top_k_score = np.mean(top_k_scores)
            
        return top_k_accuracy, average_top_k_score