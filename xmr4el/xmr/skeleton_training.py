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
    
    def _train_tree_classifier(self, X_corpus, y_corpus, config, dtype=np.float32):
        
        # Spit data for classifier training
        X_train, X_test, y_train, y_test = train_test_split(
            X_corpus,
            y_corpus,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_corpus,
        )
        
        # Save the results to tree node
        test_split = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        
        model = self._train_classifier(X_train, y_train, config)
        
        return test_split, model
    
    def _train_flat_classifier(self, conc_syn_list, config, dtype=np.float32):
        
        for syn_list in conc_syn_list:
            print(syn_list.keys())
            exit()
    
    def train_reranker(self):
        pass
        

    def execute(self, htree, all_kb_ids, embeddings_dict):
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

        tree_split, tree_model = self._train_tree_classifier(conc_array, cluster_labels, self.classifier_config)

        conc_syn_list = [embeddings_dict[ids] for ids in kb_ids]

        flat_split, flat_model = self._train_flat_classifier(conc_syn_list, self.classifier_config)

        self.classifier_config["kwargs"]["objective"] = "multiclass"
        
        # -- Step 2: Create positive and negative (x, e) pairs --        

        entity_embs = defaultdict(list)
        for eid, emb in zip(kb_ids, conc_array):
            entity_embs[eid].append(emb)
            
        X_pairs, y_labels = [], []
        
        entity_centroids = {eid: np.mean(embs, axis=0) for eid, embs in entity_embs.items()}
        
        for i, mention_emb in enumerate(conc_array):
            true_ied = kb_ids[i]
            
            pos_pair = np.hstack((mention_emb, entity_centroids[true_ied]))
            X_pairs.append(pos_pair)
            y_labels.append(1)
            
            # Sample negatives
            negatives = random.sample([eid for eid in entity_centroids if eid != true_ied], k=5)
            for neg_eid in negatives:
                neg_pair = np.hstack((mention_emb, entity_centroids[neg_eid]))
                X_pairs.append(neg_pair)
                y_labels.append(0)
        
        # Hardoced for lightgbm
        # self.classifier_config["kwargs"]["objective"] = "binary"
        
        reranker_model = self._train_classifier(
            np.array(X_pairs), 
            np.array(y_labels),
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
        htree.set_entity_centroids(entity_centroids)

        # Recurse to child nodes
        for children_htree in htree.children.values():
            self.execute(
                children_htree,
                all_kb_ids
            )
