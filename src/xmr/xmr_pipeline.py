import gc
import torch

import numpy as np

from typing import Counter

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from src.featurization.transformers import Transformer
from src.featurization.vectorizers import Vectorizer

from src.models.classifier_wrapper.classifier_model import ClassifierModel
from src.models.cluster_wrapper.clustering_model import ClusteringModel

from src.xmr.xmr_tuner import XMRTuner
from src.xmr.xmr_tree import XMRTree


class XMRPipeline():
    
    def __init__(self,
                 text_vec: Vectorizer,
                 transformer: Transformer,
                 cluster_model: ClusteringModel,
                 ranker_model: ClassifierModel
                 ):
        """Initialization"""
        
        self.text_vec = text_vec
        self.transformer = transformer
        self.cluster_model = cluster_model
        self.ranker_model = ranker_model
        
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
    def __reduce_dimensionality(emb, n_features):
        """Reduces the dimensionality of embeddings
        
        Args: 
            emb (np.array): Embeddings
            n_features (int): Number of features to have
        
        Returns:
            np.array
        """
        
        pca = PCA(n_components=n_features)  # Limit to 300 dimensions
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
    def __predict_classifier(classifier_model, data_points):
        """
            Predicts what label it will correspond with the use of the 
            classifier model.
            
            classifier_model (ClassifierModel): The Classifier Model
            data_point (np.array or sparse matrix): Point to be predicted
            
            Return:
            predicted_label (): Predicted Label       
        """
        
        return classifier_model.predict(data_points)
    
    # Done, not tested
    @classmethod
    def __execute_first_pipeline(cls,
                                 htree,
                                 text_emb,
                                 clustering_config,
                                 min_leaf_size,
                                 depth, 
                                 dtype=np.float32 
                                 ):
        """Create an Tree Structure using the text embeddings"""
        
        """Check depth"""
        if depth < 0:
            return htree
        
        """Evaluating best K according to elbow method, and some more weighted statistics"""
        k_range = (2, 6) # Hardcode for now
        k, _ = XMRTuner.tune_k(text_emb, clustering_config, dtype, k_range=k_range)
        clustering_config['n_clusters'] = k
        
        """Training Clustering Model"""
        clustering_model = cls.__train_clustering(
            text_emb, 
            clustering_config, 
            dtype
            ).model # Returns the Model (SKlearnKmeans)
        
        cluster_labels = clustering_model.model.labels_
        
        if min(Counter(cluster_labels).values()) <= min_leaf_size:
            return htree
        
        """Saving the model in the tree"""
        htree.clustering_model = clustering_model
        
        unique_labels = np.unique(cluster_labels)
        
        """Loop all the clusters processing the Transformer and concatining transformer emb with Text emb"""
        for cluster in unique_labels:
            idx = cluster_labels == cluster
            text_points = text_emb[idx]
            
            new_child_htree_instance = XMRTree(depth=htree.depth + 1)
            new_child_htree = cls.__execute_first_pipeline(new_child_htree_instance, 
                                                            text_points, 
                                                            clustering_config, 
                                                            min_leaf_size, 
                                                            depth - 1,
                                                            dtype)
            
            if new_child_htree is not None:
                htree.children[int(cluster)] = new_child_htree
                
        return htree
    
    @classmethod
    def __execute_second_pipeline(cls, 
                                 htree, 
                                 transformer_config,
                                 classifier_config,
                                 initial_text_embeddings,
                                 trn_corpus, # Not yet vectorizer,
                                 n_features, 
                                 dtype=np.float32
                                 ):
        """
            Produce the transformer embeddings and classifiers
        """
        
        """Initializing the htree attributes"""
        text_emb = htree.text_embeddings
        cluster_labels = htree.clustering_model.model.labels_

        """Check depth because the depth 0, has all the text embeddings (root)"""
        if htree.depth > 0:
            idx = initial_text_embeddings == text_emb
            input_text = trn_corpus[idx]
        else:
            input_text = trn_corpus
        
        """Predict embeddings using Transformer"""
        transformer_emb = cls.__predict_transformer(input_text, transformer_config, dtype)
        
        """Reduce the dimension of the transformer to the dimension of the vectorizer"""
        transformer_emb = cls.__reduce_dimensionality(transformer_emb, n_features)
        
        """Normalize the transformer embeddings"""
        transformer_emb = normalize(transformer_emb, norm='l2', axis=1) 
            
        """Concatenates the transformer embeddings with the text embeddings"""
        concantenated_array = np.hstack((transformer_emb, text_emb))
            
        """Train the classifier with the concatenated embeddings with cluster labels"""
        X_train, X_test, y_train, y_test = train_test_split(
            concantenated_array, 
            cluster_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=cluster_labels)
        
        classifier_model = cls.__train_classifier(X_train, y_train, classifier_config, dtype)
            
        """Save the train test split"""
        test_split = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        """Save the classifier model and test_split"""
        htree.classifier_model = classifier_model
        htree.test_split = test_split
        
        for children_htree in htree.children.values():
            
            cls.__execute_second_pipeline(children_htree, 
                                           transformer_config,
                                           classifier_config,
                                           initial_text_embeddings, 
                                           trn_corpus,
                                           dtype
                                           )
            
    def __execute_final_pipeline(self,
                                 htree, 
                                 input_text, 
                                 ):

        """
            Testing pipeline
        """
        
        classifier_model = htree.classifier_model
        test_split = htree.test_split
        
        predicted_labels = classifier_model.predict(test_split['X_test'])
        
        """Store the predicted labels"""
        """TODO"""
    
    @classmethod
    def execute_pipeline(cls,
                         trn_corpus,
                         vectorizer_config,
                         transformer_config,
                         clustering_config,
                         classifier_config,
                         n_features, # Number of Features
                         min_leaf_size, 
                         depth,
                         dtype=np.float32
                         ):
        
        """Executes the full pipelin, tree initialization, and classifier training
            
        Args:
            trn_corpus,
            vectorizer_config,
            transformer_config,
            clustering_config, 
            classifier_config,
            n_features,
            min_leaf_size,
            depth,
            dtype
            
        Return:
            htree: The tree stucture with the classifiers at each level
        """
        
        """Force garbage collection"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        """Text Vectorizer Embeddings"""
        vectorizer_model = cls.__train_vectorizer(trn_corpus, vectorizer_config, dtype)
        text_emb = cls.__predict_vectorizer(vectorizer_model, trn_corpus)
        
        """Normalize the text embeddings"""
        text_emb = text_emb.toarray()
        text_emb = normalize(text_emb, norm='l2', axis=1) 
        
        text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        
        """Executing the first pipeline, Initializing the tree structure"""
        htree = XMRTree(depth=0)
        htree = cls.__execute_first_pipeline(htree, text_emb, clustering_config, min_leaf_size, depth, dtype)
        
        """Executing the second pipeline, Training the classifiers"""
        cls.__execute_second_pipeline(htree, 
                                       transformer_config, 
                                       classifier_config, 
                                       None, 
                                       trn_corpus, 
                                       n_features, 
                                       dtype
                                       )
        
        """Testing the pipeline"""
        # TODO
        
        return htree
        
    def inference():
        pass
    
        
    
