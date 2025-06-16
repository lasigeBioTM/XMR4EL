import numpy as np

from joblib import Parallel, delayed

from sklearn.metrics import silhouette_score, davies_bouldin_score

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class XMRTuner:
    """
    A tuner class for optimizing the number of clusters (k) in clustering algorithms.
    Uses multiple metrics and parallel processing to efficiently find the optimal k.
    """
    
    @staticmethod
    def tune_k(
        trn_corpus,
        config,
        dtype,
        k_range,
        weight_silhouette=0.6,
        weight_db=0.2,
        weight_elbow=0.2,
    ):
        """
        Optimizes the number of clusters (k) using multiple evaluation metrics with parallel processing.
        
        Args:
            trn_corpus (np.array): Training data to cluster
            config (dict): Configuration dictionary for the clustering model
            dtype: Data type for computations
            k_range (tuple): Range of k values to evaluate (min_k, max_k)
            weight_silhouette (float): Weight for silhouette score in combined metric
            weight_db (float): Weight for Davies-Bouldin score in combined metric
            weight_elbow (float): Weight for elbow method (inertia) in combined metric
            
        Returns:
            tuple: (best_k, combined_scores) where:
                - best_k (int): Optimal number of clusters
                - combined_scores (np.array): Array of combined scores for each k
        """

        def evaluate_k(k):
            """
            Helper function to evaluate clustering quality for a specific k value.
            
            Args:
                k (int): Number of clusters to evaluate
                
            Returns:
                tuple: (k, inertia, silhouette_score, davies_bouldin_score)
            """
            # print(f"In evaluate_k with k={k}")
            # Train clustering model with current k
            model = ClusteringModel.train(
                trn_corpus,
                {**config, "kwargs": {**config["kwargs"], "n_clusters": k}},
                dtype=dtype,
            ).model.model
            
            # Get cluster assignments
            labels = model.labels_
            unique_labels = set(labels)
            num_clusters = len(unique_labels) # Count unique clusters (may be < k)
            
            # Handle case where clustering failed (fewer than 2 clusters)
            if num_clusters < 2:
                print(
                    f"Warning: Only {num_clusters} clusters found for k={k}, skipping..."
                )
                return k, np.inf, 0, np.inf  # Return sentinel scores
            
            # Compute evaluation metrics
            inertia = model.inertia_ # Within-cluster sum of squares
            
            
            sil_score = 0
            
            try:
                sil_score = silhouette_score(trn_corpus, labels, metric="cosine", random_state=0) if num_clusters >= 2 else 0
            except ValueError:
                sil_score = 0
                
            # Compute Davies-Bouldin score (skip if invalid)
            db_score = np.inf
            try:
                db_score = davies_bouldin_score(trn_corpus, labels) if num_clusters >= 2 else np.inf
            except ValueError:
                db_score = np.inf
            
            return k, inertia, sil_score, db_score

        # Adjust k_range if max_k exceeds number of data points 
        corpus_len = len(trn_corpus)
        k_range = (k_range[0], min(k_range[1], corpus_len))
        
        # Parallel evaluation of all k values in range
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_k)(k) for k in range(k_range[0], k_range[1] + 1)
        )

        # Filter out invalid results (where clustering failed)
        results = [res for res in results if res[1] != np.inf]
        if not results:  # If all k values failed, return default
            # print("No valid clustering found, returning k=2")
            return k_range[0], np.zeros(k_range[1] - k_range[0] + 1)  # Fallback

        # Normalize scores and compute combined metric
        ks, inertia_scores, sil_scores, db_scores = zip(*results)
        inertia_norm = 1 - (inertia_scores - np.min(inertia_scores)) / (np.max(inertia_scores) - np.min(inertia_scores) + 1e-10)
        sil_norm = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-10)
        db_norm = 1 - (db_scores - np.min(db_scores)) / (np.max(db_scores) - np.min(db_scores) + 1e-10)
        
        combined_score = (weight_elbow * inertia_norm) + (weight_silhouette * sil_norm) + (weight_db * db_norm)
        best_k = ks[np.argmax(combined_score)]
        return int(best_k), combined_score
