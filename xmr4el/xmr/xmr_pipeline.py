import os
import gc
import logging

import numpy as np

from typing import Counter

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix

from xmr4el.featurization.transformers import Transformer
from xmr4el.featurization.vectorizers import Vectorizer

from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel

from xmr4el.xmr.xmr_tuner import XMRTuner
from xmr4el.xmr.xmr_tree import XMRTree


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class XMRPipeline:
    """
    Pipeline for Extreme Multi-label Ranking (XMR) system that combines:
    - Text vectorization
    - Dimensionality reduction
    - Hierarchical clustering
    - Classification
    """

    @staticmethod
    def __train_vectorizer(trn_corpus, config, dtype=np.float32):
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
    def __predict_vectorizer(text_vec, corpus):
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
    def __predict_transformer(trn_corpus, config, dtype=np.float32):
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
    def __reduce_dimensionality(emb, n_features, random_state=0):
        """Reduces the dimensionality of embeddings

        Args:
            emb (np.array): Embeddings to reduce
            n_features (int): Target number of features
            random_state (int): Random seed for reproducibility
            
        Returns:
            Reduced dimensionality embeddings (nd.array)
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
    def __train_clustering(trn_corpus, config, dtype=np.float32):
        """Trains the clustering model with the training data

        Args:
            trn_corpus (np.array): Trainign data as a Dense Array
            config (dict): Configurations of the clustering model
            dtype (np.float): Type of the data inside the array

        Return:
            ClusteringModel (ClusteringModel): Trained Clustering Model
        """
        # Delegate training to ClusteringModel class
        return ClusteringModel.train(trn_corpus, config, dtype)

    @staticmethod
    def __train_classifier(X_corpus, y_corpus, config, dtype=np.float32):
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

    @staticmethod
    def __compute_pifa(X_tfidf, Y_train):
        """
        Compute PIFA (Positive Instance Feature Aggregation) embeddings for labels.
        This creates label embeddings by averaging the feature of positive instances.
        
        Args:
            X_tfidf: (n_samples, tfidf_dim) Tfidf Matrix, (sparse or dense)
            Y_train: (n_samples, n_labels) Binary label matrix, (sparse)
            
        Return:
            pifa_embeddings: (n_labels, tfidf_dim) PIFA Embeddings
        """
        # Convert matrices to efficient sparse formats for computation
        labels_matrix = Y_train.tocsc() # CSC for efficient column operations
        X_tfidf = csr_matrix(X_tfidf) # CSR for efficient row operations
        
        # Matrix multiplications: sum features of positive instances per label
        pifa_emb = labels_matrix @ X_tfidf
        
        # Normalize by number of positive instances per label
        label_counts = np.array(labels_matrix.sum(axis=0)).flatten()  # (n_labels,)
        label_counts = np.maximum(label_counts, 1.0)  # Prevent division by zero
        pifa_emb = pifa_emb / label_counts[:, None]  # Broadcasting division
        
        return pifa_emb

    # Tested, Working
    @classmethod
    def __execute_first_pipeline(cls, 
                                 htree, 
                                 combined_emb_idx, 
                                 text_emb_idx,
                                 clustering_config, 
                                 max_n_clusters, 
                                 min_n_clusters,
                                 min_leaf_size, 
                                 depth, 
                                 dtype=np.float32
                                 ):
        
        """
        Recursively builds hierarchical clustering tree structure using text embeddings
        
        Args:
            htree (XMRTree): Current tree node
            combined_emb_idx (dict): Indexed combined embeddings (text + PIFA)
            text_emb_idx (dict): Indexed text embeddings
            clustering_config (dict): Configuration for clustering
            max_n_clusters (int): Maximum number of clusters to consider
            min_n_clusters (int): Minimum number of clusters to consider
            min_leaf_size (int): Minimum size for a cluster to be valid
            depth (int): Remaining recursion depth
            dtype: Data type for computations
            
        Returns:
            XMRTree: Constructed hierarchical tree
        """
        
        # Convert indexed embeddings to array (sorted by index)
        indices = sorted(combined_emb_idx.keys())
        text_emb_array = np.array([combined_emb_idx[idx] for idx in indices])
        
        # Base case: stop recursion if depth exhausted
        if depth < 0:
            return htree

        # Base case: stop if too few points to cluster meaningfully
        if len(text_emb_array) <= min_n_clusters:
            return htree

        # Determine optimal number of clusters using inertia, silhouette and davies bouldin score  
        k_range = (min_n_clusters, max_n_clusters)
        optimal_k, _ = XMRTuner.tune_k(text_emb_array, clustering_config, dtype, k_range=k_range)
        
        n_clusters = optimal_k
        clustering_config["kwargs"]["n_clusters"] = n_clusters

        # Try clustering with decreasing k until valid clusters found
        while True:
            # Train clustering model with current configuration
            clustering_model = cls.__train_clustering(
                text_emb_array, clustering_config, dtype
            )  
            cluster_labels = clustering_model.model.labels()

            # Check if any cluster is to small
            if min(Counter(cluster_labels).values()) <= min_leaf_size: # if the depth is 0, create the clustering anyway
                LOGGER.warning("Skipping: Cluster size is too small.")
                
                # If we can't reduce k further, either break (root) or return (because of small X_data)
                if n_clusters == min_n_clusters:
                    LOGGER.warning("Skipping: No more clusters to reduce.")
                    if htree.depth == 0:
                        break
                    return htree
                
                # Try with fewer clusters
                n_clusters -= 1
                clustering_config["kwargs"]["n_clusters"] = n_clusters
                continue

            break  # Valid clustering found

        # Save model and embeddings to current tree node
        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(text_emb_idx)

        # Process each cluster recursively
        unique_labels = np.unique(cluster_labels)
        for cluster in unique_labels:
            # Get indices of points in this cluster
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) 
                            if label == cluster]
            
            # Create filtered embeddings for this cluster
            filt_combined_dict = {idx: combined_emb_idx[idx] for idx in cluster_indices}
            filt_text_dict = {idx: text_emb_idx[idx] for idx in cluster_indices}
            
            # Create child node and recurse
            new_child_htree_instance = XMRTree(depth=htree.depth + 1)
            new_child_htree = cls.__execute_first_pipeline(
                new_child_htree_instance,
                filt_combined_dict,
                filt_text_dict,
                clustering_config,
                max_n_clusters,
                min_n_clusters,
                min_leaf_size,
                depth - 1,
                dtype,
            )

            # Only add child if it contains meaningful structure
            if not new_child_htree.is_empty():
                htree.set_children(int(cluster), new_child_htree)

        return htree

    # Tested Working
    @classmethod
    def __execute_second_pipeline(
        cls,
        htree,
        classifier_config,
        initial_text_emb,
        initial_transformer_emb,
        n_features,
        dtype=np.float32,
    ):
        """
        Second phase: Trains classifiers at each node of the hierarchical tree
        
        Args:
            htree (XMRTree): Current tree node
            classifier_config (dict): Configuration for classifiers
            initial_text_emb (dict): Original text embeddings (indexed)
            initial_transformer_emb (np.array): Transformer embeddings
            n_features (int): Target feature dimension
            dtype: Data type for computations
        """

        # Get embeddings and cluster assignments from current node
        text_emb_idx = htree.text_embeddings 
        cluster_labels = htree.clustering_model.labels()
        
        # Convert indexed embeddings to array (sorted)
        match_index = sorted(text_emb_idx.keys())
        text_emb_array = np.array([text_emb_idx[idx] for idx in match_index])
        
        # Get corresponding transformer embeddings
        trans_emb = initial_transformer_emb[match_index]

        # Create combined feature space
        concatenated_array = np.hstack((trans_emb, text_emb_array))

        # Spit data for classifier training
        X_train, X_test, y_train, y_test = train_test_split(
            concatenated_array,
            cluster_labels,
            test_size=0.2,
            random_state=42,
            stratify=cluster_labels,
        )

        # Train classifier for this node
        classifier_model = cls.__train_classifier(
            X_train, y_train, classifier_config, dtype
        )

        # Save the results to tree node
        test_split = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        htree.set_transformer_embeddings(trans_emb)
        htree.set_kb_indices(match_index)
        htree.set_concatenated_embeddings(concatenated_array)
        htree.set_classifier_model(classifier_model)
        htree.set_test_split(test_split)

        # Recurse to child nodes
        for children_htree in htree.children.values():
            cls.__execute_second_pipeline(
                children_htree,
                classifier_config,
                initial_text_emb,
                initial_transformer_emb,
                n_features,
                dtype,
            )

    @classmethod
    def execute_pipeline(
        cls,
        trn_corpus,
        labels_matrix,
        label_enconder,
        vectorizer_config,
        transformer_config,
        clustering_config,
        classifier_config,
        n_features,  # Number of Features
        max_n_clusters,
        min_n_clusters,
        min_leaf_size,
        depth,
        dtype=np.float32,
    ):
        """
        Full XMR pipeline execution:
        1. Text vectorization
        2. PIFA label embedding creation
        3. Hierarchical clustering
        4. Classifier training at each level
        
        Args:
            trn_corpus: Training text data
            labels_matrix: Binary label matrix
            label_enconder: Label encoder
            vectorizer_config: Text vectorizer config
            transformer_config: Transformer config
            clustering_config: Clustering config
            classifier_config: Classifier config
            n_features: Target feature dimension
            max_n_clusters: Max clusters per node
            min_n_clusters: Min clusters per node
            min_leaf_size: Min points per cluster
            depth: Max tree depth
            dtype: Data type
            
        Returns:
            XMRTree: Fully trained hierarchical model
        """

        # Clean up memory before starting
        gc.collect()
        
        # Initialize tree structure 
        htree = XMRTree(depth=0)

        # Step 1: Text vectorization
        vectorizer_model = cls.__train_vectorizer(trn_corpus, vectorizer_config, dtype)
        text_emb = cls.__predict_vectorizer(vectorizer_model, trn_corpus)
        htree.set_vectorizer(vectorizer_model)

        # Convert to dense array if sparse
        text_emb = text_emb.toarray() 

        # Step 2: PIFA embeddings
        pifa_emb = cls.__compute_pifa(text_emb, labels_matrix)
        
        # Store label information
        htree.set_label_matrix(labels_matrix) # Set label matrix
        htree.set_label_enconder(label_enconder) # Set label enconder
        htree.set_pifa_embeddings(pifa_emb) # Set pifa embeddings

        # Normalize PIFA embeddings
        text_emb = normalize(text_emb, norm="l2", axis=1) # Need to cap features in kwargs
        
        # Normalize PIFA embeddings
        pifa_emb = normalize(pifa_emb, norm="l2", axis=1)
        pifa_emb = pifa_emb.toarray() # Ensure dense
        
        # Combine text and PIFA embeddings
        combined_emb = np.hstack((text_emb, pifa_emb))
        combined_emb = normalize(combined_emb, norm="l2", axis=1)
        
        # Create indexed versions for hierarchical processing
        combined_emb_index = {idx: emb for idx, emb in enumerate(combined_emb)}
        text_emb_idx = {idx: emb for idx, emb in enumerate(text_emb)}

        # Step 3: Build hierarchical clustering structure 
        htree = cls.__execute_first_pipeline(
            htree, 
            combined_emb_index, 
            text_emb_idx,
            clustering_config, 
            max_n_clusters, 
            min_n_clusters, 
            min_leaf_size, 
            depth, 
            dtype
        )
        
        # Final Embeddigns = [Transfomer] * [PIFA] * [TF-IDF] now is [Transfomer] * [TF-IDF] 

        # Step 4: Transformer Embeddings
        transformer_model = cls.__predict_transformer(
            trn_corpus, transformer_config, dtype
        )
        transformer_emb = transformer_model.embeddings()
        del transformer_model  # Clean up memory

        # Normalize and reduce transformer embeddings
        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)
        transformer_emb = cls.__reduce_dimensionality(transformer_emb, n_features)

        # Step 5: Train classifiers throughout hierarchy
        cls.__execute_second_pipeline(
            htree, 
            classifier_config, 
            text_emb_idx, 
            transformer_emb, 
            n_features, 
            dtype
        )

        return htree
