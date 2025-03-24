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
    
    # Tested, Working
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
        clustering_config['kwargs']['n_clusters'] = k
        
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
        htree.text_embeddings = text_emb
        
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
            
            if not new_child_htree.is_empty():
                htree.children[int(cluster)] = new_child_htree
                
        return htree
    
    # Tested Working
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
        text_emb = htree.text_embeddings
        cluster_labels = htree.clustering_model.model.labels_

        """Check depth because the depth 0, has all the text embeddings (root)"""
        if htree.depth > 0:
            # Create a dictionary to store the initial_text_embeddings by hash (tuple)
            embedding_dict = {tuple(initial_emb): idx for idx, initial_emb in enumerate(initial_text_embeddings)}

            # Initialize input_text as an empty list
            input_text = []

            # Iterate over each embedding in text_emb and find its exact match in embedding_dict
            for emb in text_emb:
                emb_tuple = tuple(emb)  # Convert the embedding to a tuple for hashing
                match_idx = embedding_dict.get(emb_tuple)  # Get the index of the matching embedding

                # If a match is found, add the corresponding text to input_text
                if match_idx is not None:
                    input_text.append(trn_corpus[match_idx])

        else:
            input_text = trn_corpus
        
        """Predict embeddings using Transformer"""
        transformer_model = cls.__predict_transformer(input_text, transformer_config, dtype).model
        transformer_emb = transformer_model.embeddings
        del transformer_model # Delete the model when no longer needed
        
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
        htree.transformer_embeddings = transformer_emb
        htree.concantened_embeddings = concantenated_array
        htree.test_split = test_split
        
        for children_htree in htree.children.values():
            
            cls.__execute_second_pipeline(children_htree, 
                                           transformer_config,
                                           classifier_config,
                                           initial_text_embeddings, 
                                           trn_corpus,
                                           n_features,
                                           dtype
                                           )
    
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
        
        # Force garbage collection
        """Initializing tree structure"""
        htree = XMRTree(depth=0)
        
        """Text Vectorizer Embeddings"""
        vectorizer_model = cls.__train_vectorizer(trn_corpus, vectorizer_config, dtype)
        text_emb = cls.__predict_vectorizer(vectorizer_model, trn_corpus)
        
        """Saving the vectorizer"""
        htree.vectorizer = vectorizer_model
        
        """Normalize the text embeddings"""
        text_emb = text_emb.toarray()
        text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        text_emb = normalize(text_emb, norm='l2', axis=1) 
        
        """Executing the first pipeline"""
        htree = cls.__execute_first_pipeline(htree, text_emb, clustering_config, min_leaf_size, depth, dtype)
        
        """Executing the second pipeline, Training the classifiers"""
        cls.__execute_second_pipeline(htree, 
                                       transformer_config, 
                                       classifier_config, 
                                       text_emb,
                                       trn_corpus, 
                                       n_features, 
                                       dtype
                                       )
        
        return htree
        
    @classmethod
    def inference(cls, htree, input_text, transformer_config, n_features, dtype):
        """Inference to know which cluster doest the inputs or input"""
        
        vectorizer = htree.vectorizer
        
        """Predict Embeddings using stored vectorizer"""
        text_emb = cls.__predict_vectorizer(vectorizer, input_text)
        text_emb = text_emb.toarray()
        
        """Reduce Dimensions"""
        text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        
        """Normalizing"""
        text_emb = normalize(text_emb, norm='l2', axis=1) 
        
        """Predict embeddings using Transformer"""
        transformer_model = cls.__predict_transformer(input_text, transformer_config, dtype).model
        transformer_emb = transformer_model.embeddings
        del transformer_model # Delete the model when no longer needed
        
        """Reduce the dimension of the transformer to the dimension of the vectorizer"""
        transformer_emb = cls.__reduce_dimensionality(transformer_emb, n_features)
        
        """Normalize the transformer embeddings"""
        transformer_emb = normalize(transformer_emb, norm='l2', axis=1) 
            
        """Concatenates the transformer embeddings with the text embeddings"""
        concantenated_array = np.hstack((transformer_emb, text_emb))
        
        transformer_emb = cls.__predict_transformer(transformer_config, input_text)
        
        pass
    
        
    
