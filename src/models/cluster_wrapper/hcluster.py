import os
import logging
import pickle
import numpy as np

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.models.cluster_wrapper.htuner import KTuner, SimilarityMetric
from src.models.dtree import DTree
from src.models.cluster_wrapper.clustering_model import ClusteringModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DivisiveClustering(ClusteringModel):
    
    class Config():
        
        def __init__(self, n_clusters=0, max_iter=300, depth=1, 
                     min_leaf_size=10, max_n_clusters=16, min_n_clusters=3,
                     max_merge_attempts=5, weight_silhouette=0.5, weight_db=0.3,
                     weight_elbow=0.2, random_state=0, spherical=True, model=None):
            
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.depth = depth
            self.min_leaf_size = min_leaf_size
            self.max_n_clusters = max_n_clusters
            self.min_n_clusters = min_n_clusters
            self.max_merge_attempts = max_merge_attempts
            
            # Weights 
            self.weight_silhouette = weight_silhouette
            self.weight_db = weight_db
            self.weight_elbow = weight_elbow
            
            self.random_state = random_state
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
                'random_state': self.random_state,
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
                    f"random_state={self.random_state}, "
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
        with open(os.path.join(save_dir, "hierarchical_clustering.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved Hierarchical Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """
        
        LOGGER.info(f"Loading Divisive Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "hierarchical_clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        
        model.config = cls.Config(**model.config)
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
        
        min_n_clusters = self.config.min_n_clusters
        max_n_clusters = self.config.max_n_clusters
        
        weight_silhouette = self.config.weight_silhouette
        weight_db = self.config.weight_db 
        weight_elbow = self.config.weight_elbow
        
        spherical = self.config.spherical
        
        random_state = self.config.random_state

        def split_recursively(trn_corpus, parent_dtree, depth):
            """Recursively applies divisive clustering while handling cases where only one cluster remains."""
            
            config = self.config
            
            k_range = (min_n_clusters, max_n_clusters)
            k, combined_score = KTuner.tune(trn_corpus,
                                                 config,
                                                 dtype,
                                                 k_range,
                                                 weight_silhouette,
                                                 weight_db,
                                                 weight_elbow
                                                 )
            
            config.set_model_n_clusters(k) # Sets the n_clusters inside the config class
            
            cluster_model = ClusteringModel.train(trn_corpus, config.model, dtype=dtype).model # Type ClusteringModel
            cluster_labels = cluster_model.model.labels_
            cluster_centroids = cluster_model.model.cluster_centers_
            
            allow_split = self.__allow_split(trn_corpus, 
                                             cluster_labels,
                                             cluster_centroids,
                                             min_leaf_size=min_leaf_size
                                             )
            
            
            if min(Counter(cluster_labels).values()) < min_leaf_size:
                return parent_dtree
            
            if not allow_split:
                return parent_dtree
            
            parent_dtree.node.set_cluster_node(cluster_model, trn_corpus, config.model)       

            unique_labels = np.unique(cluster_labels)

            for cl in unique_labels:
                idx = cluster_labels == cl
                cluster_points = trn_corpus[idx]
                
                new_dtree_instance = DTree(depth=parent_dtree.depth + 1)
                new_child_dtree = split_recursively(cluster_points, new_dtree_instance, depth=depth - 1)
                
                if new_child_dtree is not None:
                    parent_dtree.set_child_dtree(int(cl), new_child_dtree)
            
            return parent_dtree
        
        
        assert trn_corpus.shape[0] >= min_leaf_size, f"Training corpus is less than min_leaf_size"
        
        if spherical:
            trn_corpus = normalize(trn_corpus, norm='l2', axis=1) 
        
        init_dtree = DTree(depth=0)
        final_dtree = split_recursively(trn_corpus, init_dtree, depth)
        
        return final_dtree
        
    def __allow_split_intra_sim(self, trn_corpus, cluster_labels, cluster_centroids, intra_thresh=0.85):
        
        intra_sims = SimilarityMetric.compute_intra_cluster_similarity(trn_corpus, cluster_labels, cluster_centroids)
        
        for cluster, intra_sim in intra_sims.items():
            if intra_sim > intra_thresh:
                print(f"Cluster {cluster} is already tight (intra-cluster sim: {intra_sim:.2f}), stopping split.")
                return False
                        
        return True   
    
    def __allow_split_inter_sim(self, cluster_centroids, inter_thresh=0.75):
        inter_sims = SimilarityMetric.compute_inter_cluster_similarity(cluster_centroids)
        
        for i, j, inter_sim in inter_sims:
            if inter_sim > inter_thresh:
                print(f"Clusters {i} and {j} are too similar (inter-cluster sim: {inter_sim:.2f}), stopping further split.")
                return False
            
        # Stop if most clusters are highly similar internally or not well-separated externally
        # stop_splitting = all(intra_sim > intra_thresh for intra_sim in intra_sims.values()) or \
        #                 any(inter_sim > inter_thresh for _, _, inter_sim in inter_sims)
        
        return True
    
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
                
                cluster_predictions = int(current_cluster_node.model.predict(input_embedding)[0])
                pred.append(cluster_predictions)
                
                if cluster_predictions not in current_dtree.children:
                    break
                
                current_dtree = current_dtree.children[cluster_predictions]
                current_cluster_node = current_dtree.node.cluster_node
                
            LOGGER.info(pred)
            
            predictions.append(pred)
        
        return predictions