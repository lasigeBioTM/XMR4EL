import os
import pickle
import numpy as np

from collections import Counter
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min, silhouette_score

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
            'top_k_threshold': 0.15,
            'gpu_usage': False,
        }
        
                
        config = {**DEFAULTS, **config}
        
        min_leaf_size = config['min_leaf_size']
        random_state = config['random_state']
        top_k = config['top_k']
        top_k_threshold = config['top_k_threshold']
        gpu_usage = config['gpu_usage']
        
        
        linear_model_dict = cls.__train_linear_model(
            X, 
            Y, 
            linear_model_factory, 
            top_k, 
            top_k_threshold
        )
        
        best_linear_model = linear_model_dict['linear_model']
        best_labels = linear_model_dict['labels']
        best_top_k_indices = linear_model_dict['top_k_indices']
        best_top_k_score = linear_model_dict['top_k_score']
        best_X_test = linear_model_dict['X_test']
        best_y_test = linear_model_dict['y_test'] 
        
        # ok
        while True:
            
            new_labels_encoded = cls.__refine_top_k_clusters(
                X, 
                best_labels, 
                best_linear_model, 
                clustering_model_factory, 
                min_leaf_size, 
                random_state=random_state,
                k=top_k, 
                threshold=0.8
            )
            
            print(f"\nOld Silhouette Score: {silhouette_score(X, best_labels)}")
            print(f"New Silhouette Score: {silhouette_score(X, new_labels_encoded)}")
            
            # print(f"__fit, new_labels_encoded, Clusters: {np.unique(new_labels_encoded)}, Number of Clusters: {np.unique(new_labels_encoded).shape}")
            
            new_linear_model_dict = cls.__train_linear_model(
            X, 
            new_labels_encoded, 
            linear_model_factory, 
            top_k, 
            top_k_threshold
            )
            
            new_linear_model = new_linear_model_dict['linear_model']
            new_labels = new_linear_model_dict['labels']
            new_top_k_indices = new_linear_model_dict['top_k_indices']
            new_top_k_score = new_linear_model_dict['top_k_score']
            new_X_test = new_linear_model_dict['X_test']
            new_y_test = new_linear_model_dict['y_test'] 
            
            print(f"Best top-k score: {best_top_k_score} - New top-k score: {new_top_k_score}")
              
            if new_top_k_score > best_top_k_score:
                best_linear_model = new_linear_model
                best_labels = new_labels
                best_top_k_indices = new_top_k_indices
                best_top_k_score = new_top_k_score
                best_X_test = new_X_test
                best_y_test = new_y_test
            else:
                break
            
            """
            best_linear_model = new_linear_model
            best_labels = new_labels
            best_top_k_indices = new_top_k_indices
            best_top_k_score = new_top_k_score
            best_X_test = new_X_test
            best_y_test = new_y_test
            break
            """
            
        return cls(
            linear_model=best_linear_model, 
            linear_model_type=linear_model_factory.model_type, 
            labels=best_labels, 
            top_k_score=best_top_k_score, 
            top_k=top_k,
            x_test = best_X_test,
            y_test = best_y_test,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, predicted_labels, top_k):        
        preds = self.linear_model.predict_proba(embeddings)
        return self.__top_k_score_sklearn(preds, predicted_labels, top_k)

    # Embeddings, Labels
    @classmethod
    def __train_linear_model(cls, X, Y, linear_model_factory, top_k, top_k_threshold):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        linear_model = linear_model_factory.fit(X_train, y_train)
            
        # Predict probabilities
        y_proba = linear_model.predict_proba(X_test)
        
        # Get top-k predictions
        top_k_indices = cls.__get_top_k_indices(y_proba, top_k, top_k_threshold)
        
        # Compute top-k accuracy
        top_k_score = top_k_accuracy_score(y_test, y_proba, k=top_k, normalize=True)       
        
        # print(f"__train_linear_model Labels: {np.unique(top_k_indices)}")
        
        return {'linear_model': linear_model, 
                'labels': Y, 
                'top_k_indices': top_k_indices, 
                'top_k_score': top_k_score, 
                'X_test': X_test, 
                'y_test': y_test}
    
        
    @staticmethod
    def __calculate_cluster_margin(X, cluster_labels, linear_model, k=3):
        """
        For each cluster in `cluster_labels`, compute the average margin between 
        the highest predicted probability and the kth highest predicted probability.
        
        Parameters:
        - X: Input features (numpy array).
        - cluster_labels: Array of cluster assignments for each data point.
        - model: A trained classifier with a predict_proba method.
        - k: The top-k value (default is 3).
        
        Returns:
        A dictionary mapping cluster IDs to the average margin.
        """
        clusters = np.unique(cluster_labels)
        cluster_margins = {}
        
        for cluster in clusters:
            # Select the data points for the current cluster
            indices = (cluster_labels == cluster)
            cluster_points = X[indices]
            
            # print(cluster_points.shape)
            
            if cluster_points.shape[0] == 0:
                cluster_margins[cluster] = None
                continue
            
            # Get predicted probabilities for these points
            y_proba = linear_model.predict_proba(cluster_points)
            
            # Sort the probabilities for each sample in descending order
            sorted_probs = np.sort(y_proba, axis=1)[:, ::-1]
            
            # print(f"Top Probability: {sorted_probs[:, 0]}, Top-kth Probability: {sorted_probs[:, k-1]}")
            
            margins = sorted_probs[:, k-1]
            
            # Average the margin across all samples in the cluster 
            avg_margin = np.mean(margins)
            cluster_margins[int(cluster)] = avg_margin
        
        return cluster_margins
    
    @classmethod
    def __refine_top_k_clusters(cls, X, cluster_labels, linear_model, clustering_model_factory, min_leaf_size, random_state=0, k=3, threshold=0.7, max_refinement=1):
        """Dynamically refine clusters based on top-k accuracy"""
        refinements = 0
        while refinements < max_refinement:
            # Calculate the margins for earch cluster
            top_k_margins = cls.__calculate_cluster_margin(X, cluster_labels, linear_model, k=k).items()
            
            """
            print("__refine_top_k_clusters")
            for cluster, margin_score in top_k_margins:
                print(f"Cluster: {cluster}, Margin Score: {margin_score}")
            """
                        
            mean_top_k_margin = np.mean([margin_score for _, margin_score in top_k_margins])
            
            # Find clusters with top-k accuracy below the threshold
            clusters_to_refine = [cluster for cluster, margin_score in top_k_margins if margin_score < threshold]
            
            print(clusters_to_refine)
            
            if not clusters_to_refine:
                print("All clusters meet the top-k accuracy threshold")
                break
                
            # print(f"Clusters to refine {clusters_to_refine}")
            
            # Refine each low-performing cluster
            for cluster in clusters_to_refine:
                print(f"Cluster: {cluster}")
                
                # Save the state of cluster_labels before refinement
                previous_cluster_labels = cluster_labels.copy()
                previous_mean_top_k_margin = mean_top_k_margin
            
                indices = (cluster_labels == cluster)
                cluster_points = X[indices]
                
                n_clusters_points = cluster_points.shape[0]
                
                if n_clusters_points > min_leaf_size:
                    # Calculate optimal number of clusters
                    n_splits = cls.__calculate_optimal_clusters(cluster_points, clustering_model_factory, 16, random_state=random_state)
                    
                    # print(f"Number of splits: {n_splits}")
                    
                    # Apply KMeans to further refine this cluster 
                    clustering_model = clustering_model_factory.create_model({'n_clusters': n_splits}).fit(cluster_points)
                    sub_cluster_labels = clustering_model.labels_
                    
                    print(f"Sub cluster labels: {sub_cluster_labels}, Length: {sub_cluster_labels.shape[0]}")        
                    
                    # Reassign sub-cluster labels to the original cluster
                    unique_label = np.max(cluster_labels) + 1
                    
                    cluster_labels[indices] = unique_label + sub_cluster_labels
                    
                    # print(f"Cluster n: {cluster}")
                    
                    # Recalculate the margins after refinement
                    new_top_k_margin = cls.__calculate_cluster_margin(X, cluster_labels, linear_model, k=k).items()
                    
                    new_mean_top_k_margin = np.mean([margin_score for _, margin_score in new_top_k_margin])
                    
                    """
                    print("__refine_top_k_clusters")
                    for cluster, margin_score in new_margins:
                        print(f"Cluster: {cluster}, Margin Score: {margin_score}")
                    """

                    print(f"Old score: {previous_mean_top_k_margin}, New score: {new_mean_top_k_margin}")
                    # Check if the new top_k margin score is better
                    if new_mean_top_k_margin < previous_mean_top_k_margin:
                        print(f"New score for cluster {cluster} is worse. Reverting changes.")
                        cluster_labels = previous_cluster_labels  # Revert to previous state
                    else:
                        print(f"New score for cluster {cluster} is better. Keeping changes.")
                        mean_top_k_margin = new_mean_top_k_margin
                
                else:
                    print(f"Cluster {cluster} has too few points for further refinement.")

            refinements += 1
            print(f"Refinement {refinements} completed.")
        
        cluster_labels = cls.__merge_small_clusters(X, cluster_labels, min_leaf_size)
        
        cluster_labels = cls.__encode_labels(cluster_labels)
        
        print(f"Count Label Occurrences: \n{cls.__print_count_label_occurrences(cluster_labels)}")
        
        return cluster_labels
    
    @staticmethod
    def __calculate_optimal_clusters(X, clustering_model_factory, cluster_range=16, random_state=0):
        k_range = range(2, cluster_range)
        wcss = [clustering_model_factory.create_model(
            {'n_clusters': k, 
            'random_state':random_state}
            ).fit(X).inertia_ for k in k_range]

        knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing', online=True)
        return knee.knee
    
    @staticmethod
    def __get_top_k_indices(y_proba, k, top_k_threshold):
        """Returns the top-k indices for each sample based on predicted probabilities."""
        filtered_proba = np.where(y_proba >= top_k_threshold, y_proba, -np.inf)
        top_k_indices = np.argsort(filtered_proba, axis=1)[:, -k:][:, ::-1]
        return top_k_indices.tolist()
    
    @staticmethod
    def __encode_labels(labels_list):
        """Convert string labels into numeric labels"""
        label_to_idx = {}
        return np.array([label_to_idx.setdefault(label, len(label_to_idx)) for label in labels_list])
    
    @staticmethod    
    def __print_count_label_occurrences(labels_list):
        """Count occurrences of each label in the list"""
        labels_list = list(Counter(labels_list).items())
        sorted_data = sorted(labels_list, key=lambda x: x[1], reverse=True)
        out = ""
        for label, n_values in sorted_data:
            out += f"Cluster: {label} -> {n_values}\n"
        return out
    
    @staticmethod    
    def __count_label_occurrences(labels_list):
        """Count occurrences of each label in the list"""
        labels_list = list(Counter(labels_list).items())
        sorted_data = sorted(labels_list, key=lambda x: x[1], reverse=True)
        
        labels_occurrences = {} 
        for label, n_values in sorted_data:
            labels_occurrences[label] = n_values
            
        return labels_occurrences
    
    @staticmethod
    def __top_k_score_sklearn(pred_probs, true_labels, k=3):
        """Calculate top-k score using sklearn's top_k_accuracy_score as a helper."""
        return top_k_accuracy_score(true_labels, pred_probs, k=k, labels=np.arange(pred_probs.shape[1]))
    
    @staticmethod
    def __merge_small_clusters(X, labels, min_cluster_size=10):
        """
        Merges small clusters with larger clusters based on proximity to valid clusters.
        Ensures that all small clusters are merged.
        """
        unique_labels = np.unique(labels)
        
        # Find valid clusters and their centroids
        valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
        valid_centroids = np.array([X[labels == label].mean(axis=0) for label in valid_labels])
        
        updated_labels = labels.copy()
        
        # Process small clusters
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < min_cluster_size:
                cluster_points = X[cluster_indices]
                
                # Find the closest valid centroid
                closest_valid_idx, _ = pairwise_distances_argmin_min(cluster_points, valid_centroids)
                
                # Assign all points in the small cluster to the closest valid cluster
                closest_label = valid_labels[int(closest_valid_idx[0])]
                updated_labels[cluster_indices] = closest_label
        
        return updated_labels