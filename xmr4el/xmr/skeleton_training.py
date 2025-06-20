import logging
import gc

import numpy as np

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

    def execute(self, htree):
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

        # Get embeddings and cluster assignments from current node
        text_emb_idx = htree.text_embeddings 
        cluster_labels = htree.clustering_model.labels()
        
        # Convert indexed embeddings to array (sorted)
        match_index = sorted(text_emb_idx.keys())
        text_emb_array = np.array([text_emb_idx[idx] for idx in match_index]) # Return only the emb that match the index
        
        # Get corresponding transformer embeddings
        trans_emb = self.init_tfr_emb[match_index]

        # Create combined feature space
        conc_array = np.hstack((trans_emb, text_emb_array))

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
        
        # Setters
        htree.set_transformer_embeddings(trans_emb)
        htree.set_kb_indices(match_index)
        htree.set_concatenated_embeddings(conc_array) # trans_emb
        htree.set_classifier_model(classifier_model)
        htree.set_test_split(test_split)

        # Recurse to child nodes
        for children_htree in htree.children.values():
            self.execute(
                children_htree,
            )
