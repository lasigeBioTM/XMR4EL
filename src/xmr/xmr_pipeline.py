import logging

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


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    def __reduce_dimensionality(emb, n_features, random_state=0):
        """Reduces the dimensionality of embeddings
        
        Args: 
            emb (np.array): Embeddings
            n_features (int): Number of features to have
        
        Returns:
            np.array
        """
        
        pca = PCA(n_components=n_features, random_state=random_state)  # Limit to 300 dimensions
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
    def __predict_proba_classifier(classifier_model, data_points):
        """
            Predicts what label it will correspond with the use of the 
            classifier model.
            
            classifier_model (ClassifierModel): The Classifier Model
            data_point (np.array or sparse matrix): Point to be predicted
            
            Return:
            predicted_label (): Predicted Label       
        """
        
        return classifier_model.predict_proba(data_points)
    
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
            )# Returns the Model (ClusteringModel)
        
        # Changed this
        cluster_labels = clustering_model.model.labels()
        
        if min(Counter(cluster_labels).values()) <= min_leaf_size:
            return htree
        
        """Saving the model in the tree"""
        htree.set_clustering_model(clustering_model)
        htree.set_text_embeddings(text_emb)
        
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
                htree.set_children(int(cluster), new_child_htree)
                
        return htree
    
    # Tested Working
    @classmethod
    def __execute_second_pipeline(cls, 
                                 htree, 
                                 classifier_config,
                                 initial_text_emb,
                                 initial_transformer_emb,
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
        cluster_labels = htree.clustering_model.labels()

        """Initialize a list for indexes of the embeddings"""
        match_idx = []

        """Check depth because the depth 0, has all the text embeddings (root)"""
        if htree.depth > 0:
            # Create a dictionary mapping text embeddings to indices
            embedding_dict = {tuple(initial_emb): idx for idx, initial_emb in enumerate(initial_text_emb)}

            # Iterate over each embedding in text_emb and find its exact match in embedding_dict
            for emb in text_emb:
                emb_tuple = tuple(emb)  # Convert the embedding to a tuple for hashing
                matched_idx = embedding_dict.get(emb_tuple)  # Get the index of the matching embedding
                if matched_idx is not None:
                    match_idx.append(matched_idx)
                    
        else:
            # Depth 0, root: Include all indices from the initial embeddings
            match_idx = list(range(len(initial_text_emb)))
            
        partial_transformer_emb = initial_transformer_emb[match_idx]
        partial_text_emb = initial_text_emb[match_idx]
        
        """Concatenates the transformer embeddings with the text embeddings"""
        concantenated_array = np.hstack((partial_transformer_emb, partial_text_emb))
            
        htree.set_transformer_embeddings(partial_transformer_emb)
        htree.set_concatenated_embeddings(concantenated_array)

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
        htree.set_classifier_model(classifier_model)
        htree.set_test_split(test_split)
        
        for children_htree in htree.children.values():
            
            cls.__execute_second_pipeline(children_htree, 
                                           classifier_config,
                                           initial_text_emb,
                                           initial_transformer_emb, 
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
        htree.set_vectorizer(vectorizer_model)
        del vectorizer_model
        
        """Normalize the text embeddings"""
        text_emb = text_emb.toarray()
        text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        text_emb = normalize(text_emb, norm='l2', axis=1) 
        
        """Executing the first pipeline"""
        htree = cls.__execute_first_pipeline(htree, text_emb, clustering_config, min_leaf_size, depth, dtype)
        
        """Predict embeddings using Transformer"""
        transformer_model = cls.__predict_transformer(trn_corpus, transformer_config, dtype)
        transformer_emb = transformer_model.embeddings
        del transformer_model # Delete the model when no longer needed
        
        """Reduce the dimension of the transformer to the dimension of the vectorizer"""
        transformer_emb = cls.__reduce_dimensionality(transformer_emb, n_features)
        
        """Normalize the transformer embeddings"""
        transformer_emb = normalize(transformer_emb, norm='l2', axis=1) 
        
        """Executing the second pipeline, Training the classifiers"""
        cls.__execute_second_pipeline(htree,  
                                      classifier_config, 
                                      text_emb,
                                      transformer_emb,
                                      n_features, 
                                      dtype
                                      )
        
        return htree
    
    @classmethod
    def __inference_predict_input(cls, htree, conc_input, k):
        """Inference of an single concatenated text throw out the tree
        
        Args:
            htree (XMRTree): Trained tree
            conc_input (np.array): Concatenated text embeddings and transformer embeddings
            
        Return:
            predicted_labels (lst): Predicted labels
        """
        current_htree = htree
        predicted_labels = []
        
        while True:
            current_classifier = current_htree.classifier_model
            
            n_labels = len(current_htree.clustering_model.labels())            
                        
            if n_labels <  k:
                k = n_labels
                LOGGER.warning("Children number is '< k', k value will be the number of children")
            
            # Get top-k predictions from the classifier model
            top_k_probs = cls.__predict_proba_classifier(current_classifier, conc_input.reshape(1, -1))[0]
            top_k_indices = np.argsort(top_k_probs)[-k:][::-1]  # Sort probabilities and get top-k indices
            top_k_labels = top_k_indices[:k]
            
            predicted_labels.append(top_k_labels.tolist())
            
            # Move to the best child node if possible
            best_label = top_k_labels[0]  # Select the label with the highest probability
            
            if best_label in current_htree.children:
                current_htree = current_htree.children[best_label]
            else:
                break  # Stop if there are no more children

        return predicted_labels
            
        
    @classmethod
    def inference(cls, htree, input_text, transformer_config, n_features, k=3, dtype=np.float32):
        """Inference to know which cluster doest the inputs or input
        
        Args:
            htree (XMRTree): Trained tree
            input_text (lst): Text to be predicted by the classifiers
            transfomrer_config (config): Configuration to run a transformer
            n_features (int): Number of features that the new embeddings should have n_features + n_features
            k (int): Number of predictions to make 
            dtype (np.dtype): type of the embeddings
            
        Return:
            predicted_labels: The predicted_labels by the classifier
        """
        
        vectorizer = htree.vectorizer
        
        """Predict Embeddings using stored vectorizer"""
        text_emb = cls.__predict_vectorizer(vectorizer, input_text)
        
        text_emb = text_emb.toarray()
        
        # print(text_emb)
        
        """Reduce Dimensions"""
        text_emb = cls.__reduce_dimensionality(text_emb, n_features)
        
        # print(text_emb)
        
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
        
        # Predict labels for each concatenated input
        predicted_labels = [cls.__inference_predict_input(htree, conc_input.reshape(1, -1), k=k) for conc_input in concantenated_array]
        
        return predicted_labels
    
    def compute_top_k_accuracy(true_labels, predicted_labels, k=5):
        """
        Compute the top-k accuracy for a set of predictions.

        Args:
            true_labels (list): List of true labels (ground truth) for each sample.
            predicted_labels (list): List of top-k predicted labels (for each sample).
            k (int): The number of top predictions to consider.

        Returns:
            float: The top-k accuracy score.
        """
        correct_predictions = 0
        total_samples = len(true_labels)

        # Iterate over each sample
        for true_label, pred_labels in zip(true_labels, predicted_labels):
            # Check if the true label is within the top-k predicted labels
            if true_label in pred_labels[:k]:
                correct_predictions += 1

        # Calculate top-k accuracy
        top_k_accuracy = correct_predictions / total_samples
        return top_k_accuracy
    
