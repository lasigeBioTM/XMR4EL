import os
import logging

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, parallel_backend


from xmr4el.featurization.transformers import Transformer
from xmr4el.ranker.reranker import XMRReranker


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class XMRPredict():
    
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
        
        pca = PCA(n_components=effective_dim, random_state=random_state)
        return pca.fit_transform(emb)
    
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
    def __predict_proba_classifier(classifier_model, data_points):
        """
        Predicts what label it will correspond with the use of the
        classifier model.
        
        Args:
            classifier_model: (ClassifierModel): The Classifier Model
            data_point: (np.array or sparse matrix): Point to be predicted

        Return:
            predicted_label (): Predicted Label
        """

        return classifier_model.predict_proba(data_points)
    
    @staticmethod
    def __reranker(input_vec, labels_vec, k=5, candidates=100):
        reranker = XMRReranker(embed_dim=labels_vec.shape[1])
        indices, scores = reranker.match(
            input_vec=input_vec,
            label_vecs=labels_vec,
            top_k=k,
            candidates=candidates,
        )
        return indices, scores
    
    @staticmethod
    def __convert_predictions_into_csr(data, num_labels=None):
        rows, cols, vals = [], [], []

        for row_idx, instance in enumerate(data):
            for col_idx, score in instance:
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(np.float32(score))

        if num_labels is None:
            num_labels = max(cols) + 1  # infer if not provided

        return csr_matrix((vals, (rows, cols)), shape=(len(data), num_labels), dtype=np.float32)
    
    @classmethod
    def __inference_predict_input(cls, htree, conc_input, k=10):
        """Inference of an single concatenated text throw out the tree

        Args:
            htree (XMRTree): Trained tree
            conc_input (np.array): Concatenated text embeddings and transformer embeddings

        Return:
            predicted_labels (lst): Predicted labels
        """
        LOGGER.info(f"[PID {os.getpid()}] Processing input with shape {conc_input.shape}")
        current_htree = htree

        while True:
            current_classifier = current_htree.classifier_model

            n_labels = len(current_htree.clustering_model.labels())

            if n_labels < k:
                k = n_labels
                LOGGER.warning(
                    "Children number is '< k', k value will be the number of children"
                )

            # Get top-k predictions from the classifier model
            top_k_probs = cls.__predict_proba_classifier(
                current_classifier, conc_input.reshape(1, -1)
            )[0]
            
            top_k_indices = np.argsort(top_k_probs)[-k:][
                ::-1
            ]  # Sort probabilities and get top-k indices
        
            top_k_labels = top_k_indices[:k]

            # Move to the best child node if possible
            best_label = top_k_labels[0]  # Select the label with the highest probability

            if best_label in current_htree.children:
                current_htree = current_htree.children[best_label]
                
            else: # Get the topk kb indexes
                
                cluster_labels = current_htree.clustering_model.labels()
                indices = cluster_labels == best_label
                
                conc_emb = current_htree.concatenated_embeddings[indices]
                
                # Conc input, embeddings from the input
                # Conc emb, embeddings of the cluster
                
                # Gives trhow similarity an partion of the embeddings
                (indices, scores) = cls.__reranker(conc_input, conc_emb, k=k, candidates=100)
                
                kb_indices = []
                counter = 0
                for idx in indices:
                    kb_indices.append((current_htree.kb_indices[idx], scores[counter]))
                    counter +=1
                
                return kb_indices # Stop if there are no more children

    @classmethod
    def inference(
        cls, htree, input_text, transformer_config, k=3, dtype=np.float32
    ):
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
        transfomer_n_features = htree.transformer_embeddings.shape[1]

        """Predict Embeddings using stored vectorizer"""
        text_emb = cls.__predict_vectorizer(vectorizer, input_text)
        text_emb = text_emb.toarray()
        
        # print(text_emb)

        """Reduce Dimensions"""
        # text_emb = cls.__reduce_dimensionality(text_emb, n_features)

        # print(text_emb)

        """Normalizing"""
        text_emb = normalize(text_emb, norm="l2", axis=1)

        """Predict embeddings using Transformer"""
        transformer_model = cls.__predict_transformer(
            input_text, transformer_config, dtype
        ).model
        transformer_emb = transformer_model.embeddings
        del transformer_model  # Delete the model when no longer needed

        """Reduce the dimension of the transformer to the dimension of the vectorizer"""
        if transformer_emb.shape[1] != transfomer_n_features:
            transformer_emb = cls.__reduce_dimensionality(transformer_emb, transfomer_n_features)

        """Normalize the transformer embeddings"""
        transformer_emb = normalize(transformer_emb, norm="l2", axis=1)

        """Concatenates the transformer embeddings with the text embeddings"""
        concatenated_array = np.hstack((transformer_emb, text_emb))
        
        # Predict labels for each concatenated input
        all_kb_indices = []
        LOGGER.info(f"Starting to predict an array with shape -> {concatenated_array.shape}")
        
        with parallel_backend("threading"):
            all_kb_indices = Parallel(n_jobs=-1)(
                delayed(cls.__inference_predict_input)(htree, conc_input.reshape(1, -1), k=k)
                for conc_input in concatenated_array
            )
        # print(all_kb_indices)
        # Turn kb_indices into labels
        return cls.__convert_predictions_into_csr(all_kb_indices)