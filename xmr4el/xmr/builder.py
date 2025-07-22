import gc
import torch

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from collections import defaultdict

from xmr4el.featurization.featurization import PIFAEmbeddingFactory
from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer

from xmr4el.xmr.clustering import SkeletonConstruction
from xmr4el.xmr.classifier import SkeletonTraining
from xmr4el.xmr.reranker import SkeletonReranker
from xmr4el.xmr.skeleton import Skeleton


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
                 reranker_config,
                 n_features=1000,  # Number of Features
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
        self.reranker_config = reranker_config
        
        # Params
        self.n_features = n_features
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
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                X_tensor = torch.tensor(X_batch, dtype=torch.float).to(fusion_model.device)
                Y_tensor = torch.tensor(Y_batch, dtype=torch.float).to(fusion_model.device)

                fused_batch = fusion_model(X_tensor, Y_tensor).detach().cpu()
                fused_all.append(fused_batch)
                counter += 1

        fused_all_tensor = torch.cat(fused_all, dim=0)
        return fused_all_tensor.numpy()
    
    def _gen_label_matrix(self, label_to_indices):
        label_to_matrix = []
        for key in list(label_to_indices.keys()):
            labels_ids = label_to_indices[key]
            for _ in labels_ids:
                label_to_matrix.append([key])
        
        return label_to_matrix
    
    def execute(self, labels, x_cross_train):
        """Executes the complete XMR model building pipeline.
        
        Pipeline stages:
        1. Text vectorization and embedding generation
        2. PIFA label embedding creation
        3. Hierarchical skeleton construction
        4. Transformer embedding generation
        5. Classifier training throughout hierarchy
        
        Args:
            labels (iterable): Label identifiers
            x_cross_train (iterable): Training text data (list of synonym lists)
            device (str): Computation device (default: "cpu")
            
        Returns:
            Skeleton: Fully trained hierarchical XMR model
        """
        # Clean up memory before starting
        gc.collect()
        
        # Initialize tree structure
        htree = Skeleton(depth=0)
        
        # Step 1: Flatten synonyms into corpus and build reverse mapping
        trn_corpus = []
        label_to_indices = defaultdict(list)  # label -> indices of its synonyms in trn_corpus

        for label, synonyms in zip(labels, x_cross_train):
            for synonym in synonyms:
                idx = len(trn_corpus)
                trn_corpus.append(synonym)
                label_to_indices[label].append(idx)

        htree.set_labels(labels)
        htree.set_train_data(trn_corpus)
        htree.set_dict_data(label_to_indices)
        
        label_to_matrix = self._gen_label_matrix(label_to_indices)    
        
        label_fact = LabelEmbeddingFactory(trn_corpus, label_to_matrix, self.n_features, self.vectorizer_config)
        fact_dict = label_fact.gen_pifa()
        
        X = fact_dict["mention_embeddings"] 
        Y = fact_dict["label_to_mention_matrix"]
        Z = fact_dict["label_matrix"]
        true_label_classes = fact_dict["label_classes"]
        vec_model = fact_dict["model"]
        dim_model = fact_dict["dim_model"]
        
        print(X.shape, "X") # X
        print(Y.shape, "Y") # Y
        print(Z.shape, "Z") # Z
        print(true_label_classes.shape, "Y classes") # Y.classes_
        
        # exit()
        
        htree.set_vectorizer(vec_model)
        htree.set_dimension_model(dim_model)

        # Step 4: Build hierarchical clustering structure
        skl_form = SkeletonConstruction(
            htree,
            Z,
            clustering_config=self.clustering_config,
            min_leaf_size=self.min_leaf_size,
            depth=self.depth,
            dtype=self.dtype
        )

        print("Starting Clustering")
        htree = skl_form.execute()
        
        print(htree)
        
        # Step 5: Train classifiers throughout hierarchy
        print("Starting Training")
        skl_train = SkeletonTraining(
            X, 
            Y,
            true_label_classes,
            htree, 
            self.classifier_config, 
            self.reranker_config,
            num_negatives=5
        )
        
        skl_train.execute()
        
        skl_reranker = SkeletonReranker(
            labels,
            label_to_indices, 
            self.reranker_config
        )
        
        skl_reranker.execute(htree)
        
        return htree
