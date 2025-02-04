import os
import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min

from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU

class HieararchicalLinearModel():

    def __init__(self, model_clustering_type=None, model_linear_type=None, labels_=None, top_k_score_=None, top_k_=None):
        self.model_clustering_type = model_clustering_type
        self.model_linear_type = model_linear_type
        self.labels_ = labels_
        self.top_k_score_ = top_k_score_
        self.top_k_ = top_k_

    def save(self, linear_model_folder):
        os.makedirs(linear_model_folder, exist_ok=True)
        with open(os.path.join(linear_model_folder, 'hierarchical_linear_model.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_clustering_type': self.model_clustering_type, 'model_linear_type': self.model_linear_type}, fout)

    @classmethod
    def load(cls, linear_model_folder):
        linear_model_path = os.path.join(linear_model_folder, 'vectorizer.pkl')
        assert os.path.exists(linear_model_path), f"{linear_model_path} does not exist"
        with open(linear_model_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])
    
    
    @classmethod
    def fit(cls, X, Y, top_k=3, top_k_threshold=0.9, min_leaf_size=10, max_leaf_size=50):
    
        def merge_small_clusters(X, labels, min_cluster_size=10):
            # Get unique cluster labels
            unique_labels = np.unique(labels)
                
            # Exclude centroids of clusters with fewer than `min_cluster_size`
            valid_labels = [label for label in unique_labels if np.sum(labels == label) >= min_cluster_size]
                
            # Compute centroids of each cluster
            centroids = np.array([
                X[labels == label].mean(axis=0).A.flatten()  # Convert mean to dense
                for label in unique_labels
            ])
            
            # Create a copy of the labels to update
            updated_labels = labels.copy()

            for label in unique_labels:                
                cluster_indices = np.where(labels == label)[0]
                    
                if len(cluster_indices) <= min_cluster_size:
                    # Get the points in the small cluster
                    cluster_points = X[cluster_indices]
                        
                    # Compute distances of the small cluster points to all valid centroids
                    distances = pairwise_distances_argmin_min(cluster_points, centroids)[1]
                        
                    # Assign the points in the small cluster to the nearest valid centroid
                    closest_centroid_idx = np.argmin(distances)
                        
                    # Assign the small cluster's points to the closest cluster
                    updated_labels[labels == label] = valid_labels[closest_centroid_idx]
                
            return updated_labels
    
        def execute_pipeline(X, Y):
            # Gets an List with all the embeddings and the labels with the score of more probable
            def get_top_k_indices(y_proba, k, top_k_threshold):
                filtered_proba = np.where(y_proba >= top_k_threshold, y_proba, -np.inf)
                # print(filtered_proba)
                top_k_indices = np.argsort(filtered_proba, axis=1)[:, -k:][:, ::-1]
                return top_k_indices.tolist()

            # Final top_k_accuracy_score
            def top_k_accuracy_method(y_test, y_proba, k):            
                return top_k_accuracy_score(y_test, y_proba, k=k, normalize=True)
            
            # Embeddings, Label To Filter
            def get_embeddings_from_cluster_label(X, Y, label):
                return X[Y == label], np.where(Y == label)[0]
            
            def n_iter_labels(Y):
                num_classes = len(np.unique(Y))  # Count unique labels
                base_iter = 100  # Base iterations for small label sets
                max_iter_scaled = base_iter + (10 * num_classes)  # Scale with labels
                return max_iter_scaled
            
            def checkHowManyTimesALabelIsChosenTopKAccuracy(top_k_indices):
                unique_list = np.unique(top_k_indices)
                times_list = []
                
                for label in unique_list:
                    label_counter = 0
                    for lst in top_k_indices:
                        for x in lst:    
                            if x == label:
                                label_counter += 1
                                break
                            
                    times_list.append((label.item(), label_counter))
                
                return times_list
            
            # Splitting Data
            X_train, X_test, y_train, y_test = train_test_split(
            X, # Embeddings
            Y, # ClusterLabels
            test_size=0.2, 
            random_state=42,
            stratify=Y # Ensures Class Balance
            )    
            
            # Number of iter to not converge
            n_iter = n_iter_labels(Y)
            
            # Use of an linear model
            linear_model = LogisticRegressionCPU.train(X, Y, defaults={'solver': 'lbfgs', 'max_iter': n_iter, 'random_state': 0}).model 
            
            # Generates probabilities for each cluster
            y_proba = linear_model.predict_proba(X_test)
                
            # Gets the k most likely clusters for each test sample
            top_k_indices = get_top_k_indices(y_proba, top_k, top_k_threshold)
            
            # print(np.unique(top_k_indices))
            # print(checkHowManyTimesALabelIsChosenTopKAccuracy(top_k_indices))
            
            top_k_score = top_k_accuracy_method(y_test, y_proba, k=3)
            
            new_combined_labels = [None] * X.shape[0]
        
            for indice in np.unique(Y):
                embeddings, indices_in_cluster = get_embeddings_from_cluster_label(X, Y, indice)
                
                n_emb = embeddings.shape[0]
                
                if ((n_emb >= min_leaf_size and n_emb <= max_leaf_size) or n_emb >= max_leaf_size) and indice in np.unique(top_k_indices):
                    
                    # K-Means Clustering model
                    clustering_model = KMeansCPU.fit(embeddings, defaults={'n_clusters':2, 'max_iter':100, 'random_state':0}).model
                    
                    # New Cluster Labels
                    kmeans_labels = clustering_model.labels_
                    
                    # Combine the original cluster label (indice) with the new sub-cluster label (a, b, ...)                    
                    for idx, label in zip(indices_in_cluster, kmeans_labels):
                        new_combined_labels[idx] = f"{indice}{chr(65 + int(label))}"  # 'A', 'B', etc.
                    
                else:
                    # If the cluster does not undergo clustering, keep the original label
                    for i in indices_in_cluster:
                        new_combined_labels[i] = f"{indice}"  # Keep original label  
            
            
            return (np.array(new_combined_labels), top_k_score)
        
        # Execute the initial pipeline to get combined_labels and top_k_score
        combined_labels, top_k_score = execute_pipeline(X, Y)
        
        # Store the best top-k labels initially
        best_labels = combined_labels
        best_top_k_score = top_k_score
        
        top_k_better_results = True
        
        # First merge of small clusters
        merged_labels = merge_small_clusters(X, combined_labels, min_cluster_size=10)
        print(f"First Top-{top_k} Score: {top_k_score}, Number of Labels: {np.unique(merged_labels).shape[0]}")
        
        while top_k_better_results:
            # Execute the pipeline again with merged labels
            child_labels, child_top_k_score = execute_pipeline(X, merged_labels)
            
            print(f"Child Top-{top_k} Score: {child_top_k_score}, Number of Labels: {np.unique(child_labels).shape[0]}")
            
            # If the new top-k score is better (greater), update the labels and top-k score
            if child_top_k_score > best_top_k_score:
                best_top_k_score = child_top_k_score
                best_labels = child_labels
                print(f"New Best Top-{top_k} Score: {best_top_k_score}, Number of Labels: {np.unique(best_labels).shape[0]}")
                
                # Merge small clusters again for the child labels
                merged_labels = merge_small_clusters(X, best_labels, min_cluster_size=10)
            else:
                # Stop if the score doesn't improve
                top_k_better_results = False
                    
        # model_clustering_type=None, model_linear_type=None, labels_=None, top_k_score_=None, k_=None
        return cls("KMeansCPU", "LogisticRegressionCPU", best_labels, top_k_score, top_k)
        
        