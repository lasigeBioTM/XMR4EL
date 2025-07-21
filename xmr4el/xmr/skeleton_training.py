import logging
import gc
import random

import numpy as np

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.ranker.reranker import Reranker


# LOGGER = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


class SkeletonTraining():
    """
    Hierarchical classifier trainer for XMR tree nodes.
    
    This class handles the training of classifiers at each node of the hierarchical tree,
    using combined transformer and text embeddings as features. Key features:
    - Recursive training through the tree structure
    - Combined feature space creation
    - Stratified train-test splitting
    - Memory-efficient processing
    
    Attributes:
        init_tfr_emb (np.ndarray): Initial transformer embeddings for all samples
        classifier_config (dict): Configuration for classifier models
        test_size (float): Proportion of data for testing (0.0-1.0)
        random_state (int): Random seed for reproducibility
        dtype (np.dtype): Data type for numerical operations
    """
    
    def __init__(self, label_classes, classifier_config, reranker_config, num_negatives=5, test_size=0.2, random_state=42):
        """
        Initializes the SkeletonTraining with training parameters.
        
        Args:
            init_tfr_emb (np.ndarray): Transformer embeddings matrix (n_samples, n_features)
            classifier_config (dict): Classifier configuration parameters
            test_size (float): Proportion of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            dtype (np.dtype): Data type for computations (default: np.float32)
        """
        
        self.label_classes = label_classes
        
        self.classifier_config = classifier_config
        self.reranker_config = reranker_config
        self.num_negatives = num_negatives
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _train_classifier(X_corpus, y_corpus, config, dtype=np.float32):
        """Trains the classifier model with the training data

        Args:
            trn_corpus (np.array): Trainign data as a Dense Array
            config (dict): Configurations of the clustering model
            dtype (np.float): Type of the data inside the array

        Return:
            RankingModel (ClassifierModel): Trained Classifier Model
        """
        # Delegate training to ClassifierModel class
        return ClassifierModel.train(X_corpus, y_corpus, config, dtype)
    
    def _gen_cluster_matrix(self, features, n_clusters, cluster_ids):
        K = n_clusters
        C = np.zeros((features, K), dtype=int)
        for label_idx, cluster_idx in enumerate(cluster_ids):
            C[label_idx, cluster_idx] = 1
        return C
    
    def _train_node(self, htree):
        """
        Train a node-level classifier to route to its children.

        Args:
            X_node (np.ndarray): Embeddings of samples at this node, shape (N,d).
            true_ids (List[Any]): Corresponding gold entity IDs, length N.
            children (List[XMRTree]): Node's immediate children.
            htree: Current tree node (for setters).
        """
        model = self._train_classifier(X_tr, Y_tr, self.classifier_config)
        # Store model and splits on the tree node
        htree.set_tree_classifier(model)

    def _train_routing_nodes(self, htree):
        children = list(htree.children.values())
        
        # Prune trees that cant generate classifiers
        if len(children) < 2:
            htree.children = None
            return 

        cluster_labels = htree.clustering_model.labels() # Get cluster labels to create C matrix
        n_clusters = len(np.unique(cluster_labels)) # Calculate n of clusters
        Y = htree.text_embeddings
        
        print(Y.shape)
        exit()
        
        C = self._gen_cluster_matrix(Y.shape[1], n_clusters, cluster_labels) # Produce cluster_matrix
        M_raw = Y @ C
        M = (M_raw > 0).astype(int)
        
        # print(C)
        # print(M_raw)
        print(M)
        exit()
        
        self._train_node(X, true_ids, children, htree, self.classifier_config)
        
        for child in children:
            self._train_routing_nodes(child, all_kb_ids, comb_emb_idx)

    def _train_leaf_rerankers(self, htree, comb_emb_idx, all_embeddings):
        """
            all_kb_ids (list): has all ids
            comb_emb_idx (dict): idx: mean embeddings
            all_embeddings (list list): [[individual synonyms embedddings]]
        """
        if not htree.children:
            print(f"Preparing Reranker Child: {htree}")
            mention_indices = htree.kb_indices # All the indices of the cluster
            # print(mention_indices)
            # kb_ids = [all_kb_ids[idx] for idx in mention_indices] # each mention row maps to one KB index
            if not mention_indices:
                return
            m_embs = [all_embeddings[idx] for idx in mention_indices]
            centroid_emb = [comb_emb_idx[idx] for idx in mention_indices]

            # Map from global KB index to local index in centroid_emb
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(mention_indices)}
            local_mention_indices = [global_to_local[idx] for idx in mention_indices]

            reranker = Reranker(self.reranker_config, num_negatives=self.num_negatives)
            print("Training Reranker")
            reranker.train(m_embs, centroid_emb, local_mention_indices)
            # print(reranker)
            htree.set_reranker(reranker)
        else:
            for child in htree.children.values():
                self._train_leaf_rerankers(child, comb_emb_idx, all_embeddings)

    def execute(self, htree):
        """
        Recursively train routing classifiers and leaf rerankers.

        Args:
            htree: current XMRTree node.
            all_kb_ids: list mapping embedding rows to KB indices.
            all_embeddings: np.ndarray of mention embeddings (N,d).
            entity_embs_dict: mapping KB index -> centroid embedding.
        """
        print("Starting classifier routing nodes")
        # First, train routing classifiers as before
        self._train_routing_nodes(htree)
        print(htree)
        # Then, train rerankers at leaf nodes
        print("Starting Reranker nodes")
        # self._train_leaf_rerankers(htree, comb_emb_idx, all_embeddings)
        
