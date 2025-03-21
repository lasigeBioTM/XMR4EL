import os
import pickle
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class XMRTree():
    
    def __init__(self,
                 text_embeddings=None,
                 transformer_embeddings=None,
                 concatenated_embeddings=None,
                 clustering_model=None,
                 classifier_model=None,
                 test_split=None,
                 children=None,  # Dictionary of XMRTree nodes
                 depth=0
                 ):
        
        self.text_embeddings = text_embeddings
        self.transformer_embeddings = transformer_embeddings
        self.concatenated_embeddings = concatenated_embeddings
        
        self.clustering_model = clustering_model
        
        self.classifier_model = classifier_model
        self.test_split = test_split
        
        self.children = children if children is not None else {}
        self.depth = depth
        
    def save(self, save_dir):
        """Save trained XMRTree model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "xmrtree.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved XMRTree model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            XMRTree: The loaded object.
        """
        
        LOGGER.info(f"Loading XMRTree Model from {load_dir}")
        tree_path = os.path.join(load_dir, "xmrtree.pkl")
        assert os.path.exists(tree_path), f"XMRTree path {tree_path} does not exist"
        
        with open(tree_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    def __str__(self):
        """String representation of XMRTree."""
        return (f"XMRTree(depth={self.depth}, "
                f"num_children={len(self.children)}, "
                f"clustering_model={type(self.clustering_model).__name__ if self.clustering_model else None}, "
                f"classifier_model={type(self.classifier_model).__name__ if self.classifier_model else None}, "
                f"text_embedding_shape={self.text_embeddings.shape if self.text_embeddings is not None else None}, "
                f"transformer_embedding_shape={self.transformer_embeddings.shape if self.transformer_embeddings is not None else None}, "
                f"concatenated_embedding_shape={self.concatenated_embeddings.shape if self.concatenated_embeddings is not None else None})")
        
    