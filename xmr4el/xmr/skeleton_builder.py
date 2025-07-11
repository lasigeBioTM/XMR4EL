import gc
import logging
import torch

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from umap import UMAP

from scipy.sparse import csr_matrix

from xmr4el.featurization.featurization import PIFAEmbeddingFactory
from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer

from xmr4el.featurization.attention_fusion import AttentionFusion
from xmr4el.xmr.skeleton_construction import SkeletonConstruction
from xmr4el.xmr.skeleton_training import SkeletonTraining
from xmr4el.xmr.skeleton import Skeleton


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class SkeletonBuilder():
    """
    End-to-end pipeline for constructing an Extreme Multi-label Ranking (XMR) hierarchical model.
    
    The builder combines multiple stages:
    1. Text feature extraction (vectorization + transformer embeddings)
    2. Label embedding creation (PIFA methodology)
    3. Hierarchical skeleton construction
    4. Classifier training at each tree level
    
    The pipeline produces a complete XMRTree ready for prediction.
    
    Attributes:
        vectorizer_config (dict): Configuration for text vectorization
        transformer_config (dict): Configuration for transformer embeddings
        clustering_config (dict): Configuration for hierarchical clustering
        classifier_config (dict): Configuration for node classifiers
        n_features (int): Target dimensionality for reduced embeddings
        max_n_clusters (int): Maximum clusters per tree node
        min_n_clusters (int): Minimum clusters per tree node  
        min_leaf_size (int): Minimum samples per leaf node
        depth (int): Maximum tree depth (1000 if -1)
        dtype (np.dtype): Data type for numerical operations
    """

    def __init__(self,
                 vectorizer_config,
                 transformer_config,
                 clustering_config,
                 classifier_config, 
                 n_features=1000,  # Number of Features
                 max_n_clusters=16,
                 min_n_clusters=6,
                 min_leaf_size=10,
                 depth=3,
                 dtype=np.float32):
        """
        Initializes the SkeletonBuilder with complete pipeline configuration.
        
        Args:
            vectorizer_config (dict): Text vectorizer parameters
            transformer_config (dict): Transformer model parameters
            clustering_config (dict): Clustering algorithm parameters
            classifier_config (dict): Classifier training parameters
            n_features (int): Target feature dimension after reduction. Defaults to 1000.
            max_n_clusters (int): Maximum clusters per node. Defaults to 16.
            min_n_clusters (int): Minimum clusters per node. Defaults to 6.
            min_leaf_size (int): Minimum samples per leaf cluster. Defaults to 10.
            depth (int): Maximum tree depth (-1 for unlimited). Defaults to 3.
            dtype (np.dtype): Data type for embeddings. Defaults to np.float32.
        """
        
        # Configs
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.clustering_config = clustering_config
        self.classifier_config = classifier_config
        
        # Params
        self.n_features = n_features
        self.max_n_clusters = max_n_clusters
        self.min_n_clusters = min_n_clusters
        self.min_leaf_size = min_leaf_size
        if depth == -1:
            self.depth = 1000
        else:
            self.depth = depth
        
        self.dtype = dtype

    @staticmethod
    def _train_vectorizer(trn_corpus, config, dtype=np.float32):
        """Trains the vectorizer model with the training data

        Args:
            trn_corpus (np.array): Training Data, sparse or dense array
            config (dict): Configurations of the vectorizer model
            dtype (np.float): Type of the data inside the array

        Return:
            TextVectorizer (Vectorizer): Trained Vectorizer
        """
        # Delegate training to Vectorizer class with given configuration
        return Vectorizer.train(trn_corpus, config, dtype)

    @staticmethod
    def _predict_vectorizer(text_vec, corpus):
        """Predicts the training data with the Vectorizer model

        Args:
            text_vec (Vectorizer): The Vectorizer Model
            corpus (np.array or sparse ?): The data array to be predicted

        Return:
            Transformed text embeddings (nd.array or scipy.sparse)
        """
        # Use trained vectorizer to transform input corpus
        return text_vec.predict(corpus)

    @staticmethod
    def _predict_transformer(trn_corpus, config, dtype=np.float32):
        """Predicts the training data with the transformer model

        Args:
            trn_corpus (np.array): Training Data, sparse or dense array
            config (dict): Configurations of the vectorizer model
            dtype (np.float): Type of the data inside the array

        Return:
            Transformed Embeddings (np.array): Predicted Embeddings
        """
        # Delegate training to Transformer class with given configuration
        return Transformer.train(trn_corpus, config, dtype)

    @staticmethod
    def _reduce_dimensionality(emb, n_features, random_state=0):
        """Reduces the dimensionality of embeddings

        Args:
            emb (np.array): Embeddings to reduce
            n_features (int): Target number of features
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.array: Reduced embeddings with shape (n_samples, n_features)
                     or original if reduction not possible
        """
        # n_samples, n_features_emb = emb.shape
        # effective_dim = min(n_features, emb.shape[0])
        
        # Perform PCA with effective dimension
        svd = TruncatedSVD(n_components=n_features, random_state=random_state)
        dense_emb = svd.fit_transform(emb) # turns it into dense auto
        
        return dense_emb, svd

    @staticmethod
    def _compute_pifa(X, n_features=2**17):
        """
        Compute PIFA (Positive Instance Feature Aggregation) embeddings for labels.
        This creates label embeddings by averaging the feature of positive instances.
        
        Args:
            X_tfidf: (n_samples, tfidf_dim) Tfidf Matrix, (sparse or dense) Must be sparse, will be sparse
            Y_train: (n_samples, n_labels) Binary label matrix, (sparse)
            
        Return:
            pifa_embeddings: (n_labels, tfidf_dim) PIFA Embeddings # Will be dense ig
        """
        pifa_factory = PIFAEmbeddingFactory(n_features=n_features)
        X_csr = pifa_factory.transform(X)
        return X_csr

    @staticmethod
    def _fused_emb(X, Y, fusion_model, batch_size=1536):
        """
        Compute fused embeddings for X and Y in batches using a trained fusion_model.
        Args:
            X, Y: scipy.sparse matrices (e.g., csr_matrix)
            fusion_model: trained AttentionFusion instance
            batch_size: size of each batch
        Returns:
            numpy array of fused embeddings
        """
        fusion_model.eval()
        fusion_model.to(fusion_model.device)

        fused_all = []

        # LOGGER.info(f"Fusing embeddings, total batches: {X.shape[0] / batch_size}")
        counter = 0
        with torch.no_grad():
            
            for i in range(0, X.shape[0], batch_size):
                # LOGGER.info(f"Batch: {counter}")
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                X_tensor = torch.tensor(X_batch, dtype=torch.float).to(fusion_model.device)
                Y_tensor = torch.tensor(Y_batch, dtype=torch.float).to(fusion_model.device)

                fused_batch = fusion_model(X_tensor, Y_tensor).detach().cpu()
                fused_all.append(fused_batch)
                counter += 1

        fused_all_tensor = torch.cat(fused_all, dim=0)
        return fused_all_tensor.numpy()
            
    def execute(self, labels, x_cross_train, trn_corpus):
        """
        Executes the complete XMR model building pipeline.
        
        Pipeline stages:
        1. Text vectorization and embedding generation
        2. PIFA label embedding creation
        3. Hierarchical skeleton construction
        4. Transformer embedding generation
        5. Classifier training throughout hierarchy
        
        Args:
            labels (iterable): Label identifiers
            x_cross_train (iterable): Training text data
            trn_corpus (iterable): Processed training corpus
            
        Returns:
            Skeleton: Fully trained hierarchical XMR model
        """

        # Clean up memory before starting
        gc.collect()
        
        train_data = {}
        
        for label, trn in zip(labels, x_cross_train):
            train_data[label] = trn
        
        # Initialize tree structure 
        htree = Skeleton(depth=0, train_data=train_data)

        # Step 1: Text vectorization
        LOGGER.info(f"Started to train Vectorizer -> {self.vectorizer_config}")
        vec_model = self._train_vectorizer(trn_corpus, self.vectorizer_config, self.dtype)
        vec_emb = self._predict_vectorizer(vec_model, trn_corpus)
        htree.set_vectorizer(vec_model)
        
        # Reduce dimensions, and save the model to use in predict method
        vec_emb, svd = self._reduce_dimensionality(vec_emb, self.n_features)
        htree.set_dimension_model(svd)
        
        # Step 2: PIFA embeddings
        LOGGER.info("Computing PIFA")
        pifa_emb = self._compute_pifa(trn_corpus)
        
        # Reduce dimensions, no need to save the model
        pifa_emb, _ = self._reduce_dimensionality(pifa_emb, self.n_features)
        htree.set_pifa_embeddings(pifa_emb) # Set pifa embeddings

        # Attention Model to form conc_emb Unsupervised
        fusion_model = AttentionFusion(vec_emb.shape[1], pifa_emb.shape[1])        
        conc_emb = self._fused_emb(vec_emb, pifa_emb, fusion_model)
        
        htree.text_features = conc_emb.shape[1]

        # Normalize PIFA embeddings
        dense_conc_emb = normalize(conc_emb, norm="l2", axis=1) # Need to cap features in kwargs
        dense_vec_emb = normalize(vec_emb, norm="l2", axis=1)
        
        # Create indexed versions for hierarchical processing
        conc_emb_index = {idx: emb for idx, emb in enumerate(dense_conc_emb)}
        vec_emb_idx = {idx: emb for idx, emb in enumerate(dense_vec_emb)}

        # Step 3: Build hierarchical clustering structure 
        LOGGER.info("Initializing SkeletonConstruction")
        skl_form = SkeletonConstruction(
            max_n_clusters=self.max_n_clusters, 
            min_n_clusters=self.min_n_clusters,
            min_leaf_size=self.min_leaf_size,
            dtype=self.dtype)
        
        LOGGER.info(f"Executing Constructor -> {self.clustering_config}")
        htree = skl_form.execute(
            htree, 
            conc_emb_index, 
            vec_emb_idx, 
            depth=self.depth,
            clustering_config=self.clustering_config,
            root=True
        )
        
        LOGGER.info(htree)

        # Step 4: Transformer Embeddings
        LOGGER.info(f"Creating Transformer Embeddings -> {self.transformer_config}")
        transformer_model = self._predict_transformer(
            trn_corpus, self.transformer_config, self.dtype
        )
        transformer_emb = transformer_model.embeddings()
        del transformer_model  # Clean up memory

        # Normalize and reduce transformer embeddings
        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)

        # Step 5: Train classifiers throughout hierarchy  
        LOGGER.info(f"Initializing SkeletonTraining")      
        skl_train = SkeletonTraining(transformer_emb,
                                     self.classifier_config, 
                                     dtype=self.dtype)
        
        LOGGER.info(f"Executing Trainer -> {self.classifier_config}")
        skl_train.execute(htree)

        return htree
