import numpy as np

from scipy.sparse import csr_matrix

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


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
    
    def __init__(self, X, Y, label_classes, htree, classifier_config, reranker_config, num_negatives=5, test_size=0.2, random_state=42):
        """
        Initializes the SkeletonTraining with training parameters.
        
        Args:
            init_tfr_emb (np.ndarray): Transformer embeddings matrix (n_samples, n_features)
            classifier_config (dict): Classifier configuration parameters
            test_size (float): Proportion of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            dtype (np.dtype): Data type for computations (default: np.float32)
        """
        
        self.X = X
        self.Y = Y
        self.label_classes = label_classes
        self.htree = htree
        
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
        return ClassifierModel.train(X_corpus, y_corpus, config, dtype, onevsrest=True)
    
    def _gen_cluster_matrix(self, features, n_clusters, cluster_ids):
        K = n_clusters
        C = np.zeros((features, K), dtype=int)
        for label_idx, cluster_idx in enumerate(cluster_ids):
            C[label_idx, cluster_idx] = 1
        return C
    
    def _train_node(self, X, M):
        """
        Train a node-level classifier to route to its children.

        Args:
            X_node (np.ndarray): Embeddings of samples at this node, shape (N,d).
            true_ids (List[Any]): Corresponding gold entity IDs, length N.
            children (List[XMRTree]): Node's immediate children.
            htree: Current tree node (for setters).
        """
        """One VS Rest Classifier"""
        model = self._train_classifier(X, M, self.classifier_config)
        return model

    # Its working correctly, maybe optmize the retrieval of X_node, need to understnad how to train reranker
    def _train_routing_nodes(self, htree):
        C = htree.C
        label_indices = htree.kb_indices
        if not C.any():
            return 
        
        Y_sub = self.Y[:, label_indices]
        
        keep_mask = np.asarray(Y_sub.sum(axis=1)).flatten() > 0
        X_node = self.X[keep_mask]
        Y_node = Y_sub[keep_mask]
        
        htree.set_X(X_node)
        htree.set_Y(Y_node)
        
        M_raw = Y_node @ C
        M = (M_raw > 0).astype(int)
        
        htree.set_M(M)
        
        print(X_node.shape, Y_node.shape, M.shape)
        
        X_node = csr_matrix(X_node)
        
        model = self._train_node(X_node, M)
        htree.set_classifier(model)
        
        for child in htree.children.values():
            self._train_routing_nodes(child)


    def execute(self):
        """
        Recursively train routing classifiers and leaf rerankers.

        Args:
            htree: current XMRTree node.
            all_kb_ids: list mapping embedding rows to KB indices.
            all_embeddings: np.ndarray of mention embeddings (N,d).
            entity_embs_dict: mapping KB index -> centroid embedding.
        """
        print("Starting classifier routing nodes")
        self._train_routing_nodes(self.htree)
        
