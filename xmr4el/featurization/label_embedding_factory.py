import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MultiLabelBinarizer

from xmr4el.featurization.vectorizers import Vectorizer

class LabelEmbeddingFactory():
    
    def __init__(self, trn_corpus, labels, n_features, config):
        self.trn_corpus = trn_corpus
        self.labels = labels
        self.n_features = n_features
        self.config = config
    
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
        # Perform PCA with effective dimension
        svd = TruncatedSVD(n_components=n_features, random_state=random_state)
        dense_emb = svd.fit_transform(emb) # turns it into dense auto
        
        return dense_emb, svd
    
    def _gen_label_matrix(self):
        """Need to generate label matrix, with same labels, so that each correspond to an entity"""
        mlb = MultiLabelBinarizer(sparse_output=True)
        Y = mlb.fit_transform(self.labels)
        return Y, mlb.classes_
    
    def _gen(self):
        corpus = self.trn_corpus
        vec_model = Vectorizer.train(corpus, self.config)
        sparse_emb = vec_model.predict(corpus)
        dense_emb, dim_model = self._reduce_dimensionality(sparse_emb, self.n_features)
        emb = normalize(dense_emb, norm="l2", axis=1)
        
        return {
            "mention_embeddings": emb, 
            "model": vec_model, 
            "dim_model": dim_model
            }
    
    def gen_pifa(self):
        # Step 1: Compute label embeddings via PIFA        
        fact_dict = self._gen()
        X = fact_dict["mention_embeddings"]
        Y, Y_classes = self._gen_label_matrix()
        Z = []  # will hold z_ell for each label x
        
        for label_idx in range(Y.shape[1]):
            row_indices = Y[:, label_idx].nonzero()[0]
            if len(row_indices) == 0:
                Z.append(np.zeros(X.shape[1]))  # fallback
            else:
                positive_x = X[row_indices]
                v_ell = np.sum(positive_x, axis=0)
                z_ell = v_ell / (np.linalg.norm(v_ell) + 1e-10)
                Z.append(z_ell)

        Z = np.vstack(Z)  # shape: [num_labels, feature_dim]
        
        fact_dict["label_to_mention_matrix"] = Y
        fact_dict["label_matrix"] = Z
        fact_dict["label_classes"] = Y_classes
        return fact_dict
