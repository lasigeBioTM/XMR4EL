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

        return Vectorizer.train(trn_corpus, config, dtype)

    @staticmethod
    def __predict_vectorizer(text_vec, corpus):
        """Predicts the training data with the Vectorizer model

        Args:
            text_vec (Vectorizer): The Vectorizer Model
            corpus (np.array or sparse ?): The data array to be predicted

        Return:

        """

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

        return Transformer.train(trn_corpus, config, dtype)

    @staticmethod
    def __reduce_dimensionality(emb, n_features, random_state=0):
        """Reduces the dimensionality of embeddings

        Args:
            emb (np.array): Embeddings
            n_features (int): Number of features to have

        Returns:
            np.array
        """
        n_samples, n_features_emb = emb.shape
        max_possible = min(n_samples-1, n_features_emb)  # PCA limitation
        effective_dim = min(n_features, max_possible)
        
        if effective_dim <= 0:
            LOGGER.info("Maintaining original number of features," 
                        f"impossible to reduce, min({n_samples}, {n_features})={max_possible}")
            return emb  # Return original if reduction not possible
        
        pca = PCA(n_components=effective_dim)
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

        return ClassifierModel.train(X_corpus, y_corpus, config, dtype)

    @staticmethod
    def __compute_pifa(X_tfidf, Y_train):
        """
        Compute PIFA embeddings for labels.
        
        Args:
            X_tfidf: (n_samples, tfidf_dim) Tfidf Matrix, Array
            Y_train: (n_samples, n_labels) Binary label matrix, Sparse
            
        Return:
            pifa_embeddings: (n_labels, tfidf_dim) PIFA Embeddings
        """
        labels_matrix = Y_train.tocsc()
        X_tfidf = csr_matrix(X_tfidf)
        
        pifa_emb = labels_matrix @ X_tfidf
        
        label_counts = np.array(labels_matrix.sum(axis=0)).flatten()  # (n_labels,)
        label_counts = np.maximum(label_counts, 1.0)  # Prevent div/0
        pifa_emb = pifa_emb / label_counts[:, None]  # Broadcasting
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
        
        """Create an Tree Structure using the text embeddings"""

        """text_emb will be the tf-idf with pifa embeddings, and are indexed, dict(float, int)"""
        
        indices = sorted(combined_emb_idx.keys())
        text_emb_array = np.array([combined_emb_idx[idx] for idx in indices])
        
        """Check depth"""
        if depth < 0:
            return htree

        if len(text_emb_array) <= min_n_clusters: # So when the datapoints are less that the min_n_clusters
            return htree

        """Evaluating best K according to elbow method, and some more weighted statistics"""
        k_range = (min_n_clusters, max_n_clusters)
        optimal_k, _ = XMRTuner.tune_k(text_emb_array, clustering_config, dtype, k_range=k_range) #Return all the keys
        n_clusters = optimal_k
        clustering_config["kwargs"]["n_clusters"] = n_clusters

        while True:
            """Training Clustering Model"""
            clustering_model = cls.__train_clustering(
                text_emb_array, clustering_config, dtype
            )  # Returns the Model (ClusteringModel)

            # Changed this
            cluster_labels = clustering_model.model.labels()

            """If clustering model has enough data points"""
            if min(Counter(cluster_labels).values()) <= min_leaf_size: # if the depth is 0, create the clustering anyway
                LOGGER.warning("Skipping: Cluster size is too small.")
                
                if n_clusters == min_n_clusters:
                    LOGGER.warning("Skipping: No more clusters to reduce.")
                    if htree.depth == 0:
                        break
                    return htree
                
                n_clusters -= 1
                clustering_config["kwargs"]["n_clusters"] = n_clusters
                continue

            break # Valid clustering found, exit loop

        """Saving the model in the tree"""
        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(text_emb_idx)

        unique_labels = np.unique(cluster_labels)

        """Loop all the clusters processing the Transformer and concatining transformer emb with Text emb"""
        for cluster in unique_labels:
            # Get indices of points in this cluster
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) 
                            if label == cluster]
            
            # Create filtered dict
            filt_combined_dict = {idx: combined_emb_idx[idx] for idx in cluster_indices}
            filt_text_dict = {idx: text_emb_idx[idx] for idx in cluster_indices}
            
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
            Produce the transformer embeddings and classifiers

        Args:
            htree (XMRTtree): The tree structure with the clustering
            transformer_config (config): the config to the transformer model
            classifier_config (config): the config to the classifier model
            initial_text_embeddings (numpy.array): the initial vectorizer trn_corpus
            trn_corpus (list[str]): the training corpus
            n_features (int): the number of features/dimensions the embeddings can have
            dtype (numpy.dtype): the type of the results ?

        Return:
            htree (XMRTree): An trained tree with the classifiers, test_split,
            concantenated embeddings, transformer embeddings

        """

        """Initializing the htree attributes"""
        text_emb_idx = htree.text_embeddings 
        cluster_labels = htree.clustering_model.labels()
        
        """Get the values and the keys of the text embeddings"""
        match_index = sorted(text_emb_idx.keys())
        text_emb_array = np.array([text_emb_idx[idx] for idx in match_index])
        
        """transformers that equal to the index of the text_embeddings"""
        trans_emb = initial_transformer_emb[match_index]

        """Concatenate embeddings"""
        concatenated_array = np.hstack((trans_emb, text_emb_array))

        """Train the classifier with the concatenated embeddings with cluster labels"""
        X_train, X_test, y_train, y_test = train_test_split(
            concatenated_array,
            cluster_labels,
            test_size=0.2,
            random_state=42,
            stratify=cluster_labels,
        )

        classifier_model = cls.__train_classifier(
            X_train, y_train, classifier_config, dtype
        )

        """Save the train test split"""
        test_split = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        """Save the classifier model and test_split"""
        htree.set_transformer_embeddings(trans_emb)
        htree.set_kb_indices(match_index)
        htree.set_concatenated_embeddings(concatenated_array)
        htree.set_classifier_model(classifier_model)
        htree.set_test_split(test_split)

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
        """Executes the full pipelin, tree initialization, and classifier training

        Args:
            trn_corpus,
            vectorizer_config,
            transformer_config,
            clustering_config,
            classifier_config,
            n_features,
            max_n_clusters,
            min_n_clusters,
            min_leaf_size,
            depth,
            dtype

        Return:
            htree: The tree stucture with the classifiers at each level
        """

        gc.collect()
        
        # Force garbage collection
        """Initializing tree structure"""
        htree = XMRTree(depth=0)

        """Text Vectorizer Embeddings"""
        vectorizer_model = cls.__train_vectorizer(trn_corpus, vectorizer_config, dtype)
        text_emb = cls.__predict_vectorizer(vectorizer_model, trn_corpus)
        htree.set_vectorizer(vectorizer_model)

        """Turn to an dense array"""
        text_emb = text_emb.toarray() 

        """Create PIFA embeddings"""
        pifa_emb = cls.__compute_pifa(text_emb, labels_matrix)
        
        htree.set_label_matrix(labels_matrix) # Set label matrix
        htree.set_label_enconder(label_enconder) # Set label enconder
        htree.set_pifa_embeddings(pifa_emb) # Set pifa embeddings

        """Normalize, reduce the text embeddings"""
        text_emb = normalize(text_emb, norm="l2", axis=1)
        # text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        
        """Normalize and reduce the PIFA embeddings, already an dense array"""
        pifa_emb = normalize(pifa_emb, norm="l2", axis=1)
        
        # print("Text emb actual shape:", text_emb.shape)
        # print("PIFA emb actual shape:", pifa_emb.shape)
        # print("Text emb ndim:", text_emb.ndim)
        # print("PIFA emb ndim:", pifa_emb.ndim)
        # print(type(text_emb), type(pifa_emb))
        
        pifa_emb = pifa_emb.toarray()
        
        """Combine the embeddings"""
        combined_emb = np.hstack((text_emb, pifa_emb))
        combined_emb = normalize(combined_emb, norm="l2", axis=1)
        
        """Indexing the combined pifa tfidf embeddings"""
        combined_emb_index = {idx: emb for idx, emb in enumerate(combined_emb)}
        
        """Indexing the text embeddings"""
        text_emb_idx = {idx: emb for idx, emb in enumerate(text_emb)}

        """Executing the first pipeline"""
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
        
        # print(htree)
        
        # Final Embeddigns = [Transfomer] * [PIFA] * [TF-IDF] now is [Transfomer] * [TF-IDF] 
        
        """Normalizing and dimensionality ?"""

        """Predict embeddings using Transformer"""
        transformer_model = cls.__predict_transformer(
            trn_corpus, transformer_config, dtype
        )
        transformer_emb = transformer_model.embeddings()
        del transformer_model  # Delete the model when no longer needed

        """Normalize the transformer embeddings"""
        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)

        """Reduce the dimension of the transformer to the dimension of the vectorizer"""
        transformer_emb = cls.__reduce_dimensionality(transformer_emb, n_features)

        """Executing the second pipeline, Training the classifiers"""
        cls.__execute_second_pipeline(
            htree, 
            classifier_config, 
            text_emb_idx, 
            transformer_emb, 
            n_features, 
            dtype
        )

        return htree
