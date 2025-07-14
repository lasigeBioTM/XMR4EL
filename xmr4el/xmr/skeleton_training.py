import logging
import gc
import random

import numpy as np

from collections import defaultdict

from sklearn.model_selection import train_test_split

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    
    def __init__(self, init_tfr_emb, classifier_config, test_size=0.2, random_state=42, dtype=np.float32):
        """
        Initializes the SkeletonTraining with training parameters.
        
        Args:
            init_tfr_emb (np.ndarray): Transformer embeddings matrix (n_samples, n_features)
            classifier_config (dict): Classifier configuration parameters
            test_size (float): Proportion of data to use for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            dtype (np.dtype): Data type for computations (default: np.float32)
        """
        
        self.init_tfr_emb = init_tfr_emb
        
        self.classifier_config = classifier_config
        
        self.test_size = test_size
        self.random_state = random_state
        self.dtype = dtype

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

    def execute(self, htree, all_kb_ids):
        """
        Recursively trains classifiers throughout the hierarchical tree.
        
        For each node:
        1. Retrieves embeddings and cluster assignments
        2. Creates combined feature space
        3. Performs stratified train-test split
        4. Trains classifier
        5. Stores results in tree node
        6. Recurses to child nodes
        
        Args:
            htree (XMRTree): Current tree node being processed
            
        Note:
            - Automatically handles memory cleanup via gc.collect()
            - Maintains consistent feature space across tree levels
            - Preserves cluster proportions in train-test split
        """

        gc.collect()

        # print(all_kb_ids)

        LOGGER.info(f"Using Classifier at depth {htree.depth}")

        # Get embeddings and cluster assignments from current node
        text_emb_idx = htree.text_embeddings 
        cluster_labels = htree.clustering_model.labels()
        
        # Convert indexed embeddings to array (sorted)
        match_index = sorted(text_emb_idx.keys()) # Got the ids location
        text_emb_array = np.array([text_emb_idx[idx] for idx in match_index]) # Return only the emb that match the index
        
        # Get corresponding transformer embeddings
        trans_emb = self.init_tfr_emb[match_index]
        kb_ids = [all_kb_ids[idx] for idx in match_index] # Get the ids

        # Create combined feature space
        conc_array = np.hstack((trans_emb, text_emb_array)) # The transformer embedings with text embeddings

        # Spit data for classifier training
        X_train, X_test, y_train, y_test = train_test_split(
            conc_array, # trans_emb
            cluster_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=cluster_labels,
        )

        # Train classifier for this node
        classifier_model = self._train_classifier(
            X_train, 
            y_train, 
            self.classifier_config, 
            self.dtype
        )

        # Save the results to tree node
        test_split = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        
        # -- Step 2: Create positive and negative (x, e) pairs --        

        entity_embs = defaultdict(list)
        for eid, emb in zip(kb_ids, conc_array):
            entity_embs[eid].append(emb)
            
        entity_emb_dict = {eid: np.mean(embs, axis=0) for eid, embs in entity_embs.items()}
        
        X_pairs, y_labels = [], []
        
        for i, m_emb in enumerate(conc_array):
            true_eid = kb_ids[i]

            # Positive pair
            pos_pair = (m_emb, entity_emb_dict[true_eid])
            X_pairs.append(np.hstack(pos_pair))
            y_labels.append(1)

            # Sample 5 random *incorrect* entity IDs
            negatives = random.sample([eid for eid in entity_emb_dict if eid != true_eid], k=5)
            for neg_eid in negatives:
                neg_pair = (m_emb, entity_emb_dict[neg_eid])
                X_pairs.append(np.hstack(neg_pair))
                y_labels.append(0)
        
        # Train Reranker with X_pairs and y_labels, postive or negative
        
        reranker_model = self._train_classifier(
            X_pairs, 
            y_labels,
            self.classifier_config,
            self.dtype
        )
        
        # Setters
        htree.set_transformer_embeddings(trans_emb)
        htree.set_kb_indices(match_index)
        htree.set_concatenated_embeddings(conc_array)
        
        htree.set_classifier_model(classifier_model)
        htree.set_test_split(test_split)
        
        htree.set_reranker(reranker_model)

        # Recurse to child nodes
        for children_htree in htree.children.values():
            self.execute(
                children_htree,
                all_kb_ids
            )
