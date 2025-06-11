import gc
import logging
import torch

import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

from umap import UMAP

from scipy.sparse import csr_matrix, vstack

from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer

from xmr4el.xmr.attention_fusion import AttentionFusion
from xmr4el.xmr.skeleton_construction import SkeletonConstruction
from xmr4el.xmr.skeleton_training import SkeletonTraining
from xmr4el.xmr.skeleton import Skeleton


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SkeletonBuilder():
    """
    Pipeline for Extreme Multi-label Ranking (XMR) system that combines:
    - Text vectorization
    - Dimensionality reduction
    - Hierarchical clustering
    - Classification
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
        Init
            vectorizer_config (dict): Text vectorizer config
            transformer_config (dict): Transformer config
            clustering_config (dict): Clustering config
            classifier_config (dict): Classifier config
            n_features (int): Target feature dimension
            max_n_clusters (int): Max clusters per node
            min_n_clusters (int): Min clusters per node
            min_leaf_size (int): Min points per cluster
            depth (int): Max tree depth
            dtype (np.type): Data type
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
        n_samples, n_features_emb = emb.shape
        # PCA cannot have more components than min(n_samples, n_features)
        max_possible = min(n_samples-1, n_features_emb)  # PCA limitation
        effective_dim = min(n_features, max_possible)
        
        if effective_dim <= 0:
            LOGGER.info("Maintaining original number of features," 
                        f"impossible to reduce, min({n_samples}, {n_features})={max_possible}")
            return emb  # Return original if reduction not possible
        
        # Perform PCA with effective dimension
        pca = PCA(n_components=effective_dim, random_state=random_state)
        return pca.fit_transform(emb)

    @staticmethod
    def _compute_pifa(X_tfidf, Y_train):
        """
        Compute PIFA (Positive Instance Feature Aggregation) embeddings for labels.
        This creates label embeddings by averaging the feature of positive instances.
        
        Args:
            X_tfidf: (n_samples, tfidf_dim) Tfidf Matrix, (sparse or dense) Must be sparse, will be sparse
            Y_train: (n_samples, n_labels) Binary label matrix, (sparse)
            
        Return:
            pifa_embeddings: (n_labels, tfidf_dim) PIFA Embeddings # Will be dense ig
        """
        # Convert matrices to efficient sparse formats for computation
        labels_matrix = Y_train.tocsc() # CSC for efficient column operations
        
        # Matrix multiplications: sum features of positive instances per label
        pifa_emb = labels_matrix @ X_tfidf
        
        # Normalize by number of positive instances per label
        label_counts = np.array(labels_matrix.sum(axis=0)).flatten()  # (n_labels,)
        label_counts = np.maximum(label_counts, 1.0)  # Prevent division by zero
        pifa_emb = pifa_emb / label_counts[:, None]  # Broadcasting division
        
        return pifa_emb

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

        LOGGER.info(f"Fusing embeddings, total batches: {X.shape[0] / batch_size}")
        counter = 0
        with torch.no_grad():
            
            for i in range(0, X.shape[0], batch_size):
                LOGGER.info(f"Batch: {counter}")
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                X_tensor = torch.tensor(X_batch.toarray(), dtype=torch.float).to(fusion_model.device)
                Y_tensor = torch.tensor(Y_batch.toarray(), dtype=torch.float).to(fusion_model.device)

                fused_batch = fusion_model(X_tensor, Y_tensor).detach().cpu()
                fused_all.append(fused_batch)
                counter += 1

        fused_all_tensor = torch.cat(fused_all, dim=0)
        return fused_all_tensor.numpy()
            
    def execute(
        self,
        labels,
        x_cross_train,
        trn_corpus,
        labels_matrix
    ):
        """
        Full XMR pipeline execution:
        1. Text vectorization
        2. PIFA label embedding creation
        3. Hierarchical clustering
        4. Classifier training at each level
        
        Args:
            labels
            x_cross_train: Training text data
            trn_corpus: Binary label matrix
            labels_matrix: Label encoder
            
        Returns:
            XMRTree: Fully trained hierarchical model
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
        htree.set_vectorizer(vec_model)
        
        # Predict the embeddings
        vec_emb = self._predict_vectorizer(vec_model, trn_corpus)

        # Step 2: PIFA embeddings
        LOGGER.info("Computing PIFA")
        pifa_emb = self._compute_pifa(vec_emb, labels_matrix).tocsr()
        
        # Store pifa embeddings information
        htree.set_pifa_embeddings(pifa_emb) # Set pifa embeddings

        # Attention Model to form conc_emb
        fusion_model = AttentionFusion(vec_emb.shape[1], pifa_emb.shape[1])        
        conc_emb = self._fused_emb(vec_emb, pifa_emb, fusion_model)
        sparse_conc_emb = csr_matrix(conc_emb)
        
        print(conc_emb.shape, type(conc_emb))
        
        # Truncate to use UMAP next
        svd = TruncatedSVD(n_components=10000, random_state=0)
        dense_conc_emb = svd.fit_transform(sparse_conc_emb) # turns it into dense auto
        
        # Prepare UMAP
        n_samples = conc_emb.shape[0]
        n_neighbors = min(15, max(2, n_samples - 1))
        n_components = min(self.n_features, n_samples - 2) # Got to be -2, cause of spectral decomposition
        
        umap = UMAP(n_components=n_components, metric='cosine', n_neighbors=n_neighbors, n_jobs=-1)
        conc_emb = umap.fit_transform(dense_conc_emb)

        # Normalize PIFA embeddings
        dense_conc_emb = normalize(conc_emb, norm="l2", axis=1) # Need to cap features in kwargs
        vec_emb = normalize(conc_emb, norm="l2", axis=1)
        
        # Create indexed versions for hierarchical processing
        conc_emb_index = {idx: emb for idx, emb in enumerate(conc_emb)}
        vec_emb_idx = {idx: emb for idx, emb in enumerate(vec_emb)}

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
            clustering_config=self.clustering_config
        )
        
        # Final Embeddigns = [Transfomer] * [PIFA] * [TF-IDF] now is [Transfomer] * [TF-IDF] 

        # Step 4: Transformer Embeddings
        LOGGER.info(f"Creating Transformer Embeddings -> {self.transformer_config}")
        transformer_model = self._predict_transformer(
            trn_corpus, self.transformer_config, self.dtype
        )
        transformer_emb = transformer_model.embeddings()
        del transformer_model  # Clean up memory

        # Normalize and reduce transformer embeddings
        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)
        transformer_emb = self._reduce_dimensionality(transformer_emb, self.n_features)

        # Step 5: Train classifiers throughout hierarchy  
        LOGGER.info(f"Initializing SkeletonTraining")      
        skl_train = SkeletonTraining(transformer_emb,
                                     self.classifier_config, 
                                     dtype=self.dtype)
        
        LOGGER.info(f"Executing Trainer -> {self.classifier_config}")
        skl_train.execute(htree)

        return htree
