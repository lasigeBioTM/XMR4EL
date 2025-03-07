import os
import logging
import pickle
import numpy as np

from kneed import KneeLocator

from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from src.models.dtree import DTree
from src.models.cluster_wrapper.clustering_model import ClusteringModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DivisiveClustering(ClusteringModel):
    
    class Config():
        
        def __init__(self, n_clusters=0, max_iter=300, depth=1, 
                     min_leaf_size=10, max_n_clusters=16, min_n_clusters=3,
                     max_merge_attempts=5, spherical=True, model=None):
            
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.depth = depth
            self.min_leaf_size = min_leaf_size
            self.max_n_clusters = max_n_clusters
            self.min_n_clusters = min_n_clusters
            self.max_merge_attempts = max_merge_attempts
            self.spherical = spherical
            
            if model is None:
                self.model = {
                'type': 'sklearnkmeans', 
                'kwargs': {}
            }
            else: self.model = model
        
        def __update_mode_config(self):
            self.model['kwargs'] = self.__to_dict
        
        def to_dict(self):
            return {
                'n_clusters': self.n_clusters,
                'max_iter': self.max_iter,
                'depth': self.depth,
                'min_leaf_size': self.min_leaf_size,
                'max_n_clusters': self.max_n_clusters,
                'min_n_clusters': self.min_n_clusters,
                'max_merge_attempts': self.max_merge_attempts,
                'spherical': self.spherical,
                'model': self.model
            }
            
        def set_model_n_clusters(self, n_clusters):
            assert n_clusters is not None and n_clusters > 1, f"Value of n_clusters is not valid"
            self.model['kwargs']['n_clusters'] = n_clusters
            
        def __str__(self):
            return (f"Config(n_clusters={self.n_clusters}, "
                    f"max_iter={self.max_iter}, "
                    f"depth={self.depth}, "
                    f"min_leaf_size={self.min_leaf_size}, "
                    f"max_n_clusters={self.max_n_clusters}, "
                    f"min_n_clusters={self.min_n_clusters}, "
                    f"max_merge_attempts={self.max_merge_attempts}, "
                    f"spherical={self.spherical}, "
                    f"model={self.model})")
            
            
    def __init__(self, config=None, dtree=None):
        self.config = config
        self.dtree: DTree = dtree
        
    def save(self, save_dir):
        """Save trained Hierarchical Clustering model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved Hierarchical Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """
        
        LOGGER.info(f"Loading Agglomerative Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        
        defaults = cls.Config().to_dict()
        
        try:
            config = {**defaults, **config}
            model = DivisiveClustering(config=cls.Config(**config))
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for DivisiveClustering"
            )
        dtree = model.fit(trn_corpus, dtype=dtype)
        return cls(config, dtree)
    
    def fit(self, trn_corpus, dtype):
        
        depth = self.config.depth
        min_leaf_size = self.config.min_leaf_size
        max_n_clusters = self.config.max_n_clusters
        min_n_clusters = self.config.min_n_clusters
        max_merge_attempts = self.config.max_merge_attempts
        spherical = self.config.spherical
        
        def split_recursively(trn_corpus, dtree, depth):
            """Recursively applies divisive clustering while handling cases where only one cluster remains."""
            
            # LOGGER.info(f"Dtree: {dtree}")
            
            config = self.config
            
            # Assures that wont be created an cluster algorithm with less than min_leaf_size
            if trn_corpus.shape[0] <= min_leaf_size:
                return dtree
            
            n_clusters = self.__compute_elbow(trn_corpus, dtype=dtype) or max_n_clusters
            
            config.set_model_n_clusters(int(n_clusters)) # Sets the n_clusters inside the config class
            
            merge_attempts = 0
            
            LOGGER.info(f"Starting to merge")
            while merge_attempts < max_merge_attempts:
                cluster_model = ClusteringModel.train(trn_corpus, config.model, dtype=dtype).model # Type ClusteringModel
                cluster_labels = cluster_model.model.labels_
                centroids = cluster_model.model.cluster_centers_
                
                # Stop to cluster if the number of labels is less than the min_n_clusters permits
                unique_labels = np.unique(cluster_labels)
                if len(unique_labels) <= min_n_clusters:
                    break

                # Merge small clusters
                LOGGER.info("Compact Clusters")
                compacted_labels = self.__compact_clusters(trn_corpus, cluster_labels, centroids)
                
                if np.array_equal(cluster_labels, compacted_labels):
                    break   # Stop merging if labels don't change
                
                n_clusters = len(np.unique(compacted_labels))
                config.set_model_n_clusters(n_clusters)
                merge_attempts += 1
            
            dtree.node.set_cluster_node(cluster_model, trn_corpus, config.model)

            # overall_silhouette_score, silhouette_scores = cls.__compute_silhouette_scores(X, cluster_labels)          

            unique_labels = np.unique(cluster_labels)
            
            LOGGER.info(f"Spliting")
            for cl in unique_labels:
                idx = cluster_labels == cl
                c_points = trn_corpus[idx]
                
                if c_points.shape[0] <= min_leaf_size or depth - 1 == 0: # Stop clustring if datapoints are less than required
                    continue
                
                child_dtree = DTree(depth=dtree.depth + 1)
                new_child_dtree = split_recursively(c_points, child_dtree, depth=depth - 1)
                dtree.set_child_dtree(cl, new_child_dtree)
            
            return dtree
        
        if spherical:
            trn_corpus = normalize(trn_corpus, norm='l2', axis=1)  
            
        return split_recursively(trn_corpus, DTree(), depth)
                
    def __compact_clusters(self, trn_corpus, cluster_labels, centroids):
        """Merges Clusters that are smaller than the specified size"""
        
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        
        if len(unique) <= self.config.min_n_clusters:
            return cluster_labels
        
        # c == cluster
        small_clusters = {c for c, size in cluster_sizes.items() if size < self.config.min_leaf_size}
        if len(small_clusters) == 0:
            return cluster_labels # No small clusters, return original labels
        
        updated_labels = cluster_labels.copy()
        
        large_clusters = [c for c in np.unique(updated_labels) if c not in small_clusters]
        if len(large_clusters) == 0:
            return np.zeros_like(cluster_labels) # Assign all points to cluster 0 

        large_centroids = centroids[large_clusters]
        
        for cluster in small_clusters:
            # Find the indices of the points belonging to the small cluster
            indices = np.where(updated_labels == cluster)[0]
            
            if len(indices) == 0:
                continue
            
            # Use cosine similarity
            distances = cosine_distances(trn_corpus[indices], large_centroids)
            
            # Assign each data point to the closest large cluster centroid
            nearest_cluster = np.argmin(distances, axis=1)
            updated_labels[indices] = np.array(large_clusters)[nearest_cluster]
            
        return updated_labels

    def __compute_elbow(self, trn_corpus, dtype):
        """Computes the best number of clusters to a specific trn_corpus"""
        
        cluster_range = min(len(trn_corpus), self.config.max_n_clusters)
        k_range = range(2, cluster_range)
        
        wcss = [
            ClusteringModel.train(
                trn_corpus, 
                {
                    **self.config.model,  # Keep the type and kwargs
                    'kwargs': {**self.config.model['kwargs'], 'n_clusters': k}  # Update or add n_clusters to kwargs
                }, 
                dtype=dtype
            ).model.model.inertia_
            for k in k_range
        ]

        knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing', online=True)
        optimal_clusters = knee.knee
        return optimal_clusters
        
    def predict(self, batch_embeddings):
        """Predicts the cluster for input data X by traversing the hierarchical tree."""
        
        predictions = []
        for input_embedding in batch_embeddings:
            
            LOGGER.debug(input_embedding)
            
            current_dtree = self.dtree  # Start at the root
            current_cluster_node = current_dtree.node.cluster_node
            pred = []
            
            while current_cluster_node.is_populated():
                input_embedding = input_embedding.reshape(1, -1)
                
                LOGGER.debug(current_cluster_node.model.model.get_params())
                
                cluster_predictions = current_cluster_node.model.predict(input_embedding)[0]
                
                LOGGER.debug(cluster_predictions)
                pred.append(int(cluster_predictions))
                
                if cluster_predictions not in current_dtree.children:
                    LOGGER.debug("No Children")
                    break
                
                current_dtree = current_dtree.children[int(cluster_predictions)]
            
            predictions.append(pred)
        
        return predictions