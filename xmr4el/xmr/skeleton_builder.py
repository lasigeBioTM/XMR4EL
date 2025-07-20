import numpy as np

from sklearn.model_selection import train_test_split

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.ranker.reranker import Reranker


class SkeletonTraining():
    """Hierarchical classifier trainer for XMR tree nodes.
    
    Handles training of classifiers at each node of the hierarchical tree,
    using combined embeddings as features. Features include:
    - Recursive training through tree structure
    - Combined feature space creation
    - Memory-efficient processing
    - Leaf node reranker training
    
    Attributes:
        classifier_config (dict): Configuration for classifier models
        reranker_config (dict): Configuration for reranker models
        num_negatives (int): Number of negative samples for reranker
        test_size (float): Proportion of data for testing (0.0-1.0)
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, classifier_config, reranker_config, num_negatives=5, test_size=0.2, random_state=42):
        """Initializes the SkeletonTraining with training parameters.
        
        Args:
            classifier_config (dict): Classifier configuration parameters
            reranker_config (dict): Reranker configuration parameters
            num_negatives (int): Number of negative samples (default: 5)
            test_size (float): Test data proportion (default: 0.2)
            random_state (int): Random seed (default: 42)
        """
        self.classifier_config = classifier_config
        self.reranker_config = reranker_config
        self.num_negatives = num_negatives
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _train_classifier(X_corpus, y_corpus, config, dtype=np.float32):
        """Trains the classifier model with training data.

        Args:
            X_corpus (np.array): Training data features
            y_corpus (np.array): Training data labels
            config (dict): Model configurations
            dtype (np.dtype): Data type (default: np.float32)

        Returns:
            ClassifierModel: Trained classification model
        """
        return ClassifierModel.train(X_corpus, y_corpus, config, dtype)
    
    def _build_targets(self, true_ids, children_kb_sets):
        """Build routing targets for node's children.
        
        Args:
            true_ids (List[Any]): Gold entity IDs for samples
            children_kb_sets (List[Set]): Each child's set of leaf IDs
        
        Returns:
            np.ndarray: Array where Y[i] = j if true_ids[i] in children_kb_sets[j]
        """
        Y = np.zeros(len(true_ids), dtype=np.int64)
        for i, eid in enumerate(true_ids):
            for j, kb_set in enumerate(children_kb_sets):
                if eid in kb_set:
                    Y[i] = j
                    break  # First match wins
        return Y
    
    def _train_node(self, X_node, true_ids, children, htree):
        """Train a node-level classifier to route to its children.

        Args:
            X_node (np.ndarray): Sample embeddings (N,d)
            true_ids (List[Any]): Corresponding gold entity IDs
            children (List[XMRTree]): Node's immediate children
            htree: Current tree node
        """
        children_kb_sets = [set(child.kb_indices) for child in children]
        Y_node = self._build_targets(true_ids, children_kb_sets)
        
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_node, Y_node,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        self.classifier_config["kwargs"]["onevsrest"] = False
        model = self._train_classifier(X_tr, Y_tr, self.classifier_config)
        htree.set_tree_classifier(model)

    def _train_routing_nodes(self, htree, all_kb_ids, comb_emb_idx):
        """Recursively train routing classifiers for each node.
        
        Args:
            htree: Current tree node
            all_kb_ids: List of all KB indices
            comb_emb_idx: Dict of index to mean embeddings
        """
        children = list(htree.children.values())
        
        # Base cases
        if len(children) < 2:
            htree.children = None
            return
        if not children or not htree.kb_indices:
            return
            
        # Prepare node data
        node_indices = htree.kb_indices
        X_node = np.array([comb_emb_idx[idx] for idx in node_indices])
        
        # Train current node and recurse
        self._train_node(X_node, node_indices, children, htree)
        for child in children:
            self._train_routing_nodes(child, all_kb_ids, comb_emb_idx)

    def _train_leaf_rerankers(self, htree, comb_emb_idx, all_embeddings):
        """Train rerankers at leaf nodes.
        
        Args:
            htree: Current tree node
            comb_emb_idx: Dict of index to mean embeddings
            all_embeddings: List of individual synonym embeddings
        """
        if not htree.children:  # Leaf node
            mention_indices = htree.kb_indices
            if not mention_indices:
                return
                
            # Prepare reranker data
            m_embs = [all_embeddings[idx] for idx in mention_indices]
            centroid_emb = [comb_emb_idx[idx] for idx in mention_indices]
            global_to_local = {g: l for l, g in enumerate(mention_indices)}
            local_indices = [global_to_local[idx] for idx in mention_indices]

            # Train and set reranker
            reranker = Reranker(self.reranker_config, self.num_negatives)
            reranker.train(m_embs, centroid_emb, local_indices)
            htree.set_reranker(reranker)
        else:  # Non-leaf node
            for child in htree.children.values():
                self._train_leaf_rerankers(child, comb_emb_idx, all_embeddings)

    def execute(self, htree, all_kb_ids, comb_emb_idx, all_embeddings):
        """Execute complete training pipeline.
        
        Args:
            htree: Root tree node
            all_kb_ids: List mapping embedding rows to KB indices
            comb_emb_idx: Dict of index to mean embeddings
            all_embeddings: List of individual synonym embeddings
        """
        print("Training classifier routing nodes")
        self._train_routing_nodes(htree, all_kb_ids, comb_emb_idx)
        
        print("Training reranker nodes")
        self._train_leaf_rerankers(htree, comb_emb_idx, all_embeddings)