import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, normalize

class LabelEmbeddingFactory():
        
    @staticmethod
    def generate_label_matrix(label_to_indices):
        label_to_matrix = []
        for key in list(label_to_indices.keys()):
            labels_ids = label_to_indices[key]
            for _ in labels_ids:
                label_to_matrix.append([key])
        
        return label_to_matrix    
        
    @staticmethod
    def label_binarizer(labels):
        """Need to generate label matrix, with same labels, so that each correspond to an entity"""
        mlb = MultiLabelBinarizer(sparse_output=True)
        Y = mlb.fit_transform(labels) # Sparse
        return Y, mlb.classes_
    
    @staticmethod
    def generate_PIFA(X, Y):
        # Partitioned Inverse Frequency Aggregation   
        # Used for creation of label/entity embeddings
        Z = []  # will hold z_ell for each label x
        
        for label_idx in range(Y.shape[1]):
            row_indices = Y[:, label_idx].nonzero()[0]
            
            if len(row_indices) == 0:
                Z.append(np.zeros(X.shape[1]))  # fallback
                
            else:
                positive_x = X[row_indices]
                v_ell = np.sum(positive_x, axis=0) # aggregate as described in PECOS
                v_ell = np.asarray(v_ell).ravel()
                z_ell = v_ell / (np.linalg.norm(v_ell) + 1e-10)
                Z.append(z_ell)

        Z = np.vstack(Z)  # shape: [num_labels, feature_dim] # label/entity embeddings
        Z = normalize(Z, norm="l2", axis=1)
        return Z
    
