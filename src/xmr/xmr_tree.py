import os
import pickle
import logging
import datetime
import glob

import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class XMRTree():
    
    def __init__(self,
                 text_embeddings=None,
                 transformer_embeddings=None,
                 concatenated_embeddings=None,
                 vectorizer=None,
                 clustering_model=None,
                 classifier_model=None,
                 test_split=None,
                 children=None,  # Dictionary of XMRTree nodes
                 depth=0
                 ):
        
        self.text_embeddings = text_embeddings
        self.transformer_embeddings = transformer_embeddings
        self.concatenated_embeddings = concatenated_embeddings
        
        self.vectorizer = vectorizer
        
        self.clustering_model = clustering_model
        
        self.classifier_model = classifier_model
        self.test_split = test_split
        
        self.children = children if children is not None else {}
        self.depth = depth
        
    def save(self, base_dir="saved_trees"):
        """Save trained XMRTree model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """ 
        # Generate directory name based on class name and current date-time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(base_dir, f"{self.__class__.__name__}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Prepare state dictionary without embeddings
        state = self.__dict__.copy()
        
        # Save embeddings separately
        if self.text_embeddings is not None:
            np.save(os.path.join(save_dir, "text_embeddings.npy"), self.text_embeddings)
            state.pop("text_embeddings", None)
        else:
            LOGGER.warning("Text embeddings is None")

        if self.transformer_embeddings is not None:
            np.save(os.path.join(save_dir, "transformer_embeddings.npy"), self.transformer_embeddings)
            state.pop("transformer_embeddings", None)
        else:
            LOGGER.warning("Transformer embeddings is None")

        if self.concatenated_embeddings is not None:
            np.save(os.path.join(save_dir, "concatenated_embeddings.npy"), self.concatenated_embeddings)
            state.pop("concatenated_embeddings", None)
        else:
            LOGGER.warning("Concatenated embeddings is None")

        # Save each child tree separately
        children_dir = os.path.join(save_dir, "children")
        os.makedirs(children_dir, exist_ok=True)
        for idx, child in self.children.items():
            child.save(os.path.join(children_dir, f"child_{idx}"))

        # Save the metadata using pickle
        with open(os.path.join(save_dir, "xmrtree.pkl"), "wb") as fout:
            pickle.dump(state, fout)
            
        LOGGER.info("Correctly Saved")
    
    @classmethod
    def load(cls, base_dir="saved_trees", load_dir=None):
        """Load a saved XMRTree model from disk.

        Args:
            base_dir (str): The root directory where trees are saved.
            load_dir (str): Specific folder to load from. If None, loads the latest saved model.

        Returns:
            XMRTree: The loaded object.
        """
        if load_dir is None:
            # Find the most recent directory
            all_saves = sorted(glob.glob(os.path.join(base_dir, f"{cls.__name__}_*")), reverse=True)
            if not all_saves:
                raise FileNotFoundError(f"No saved {cls.__name__} models found in {base_dir}.")
            load_dir = all_saves[0]  # Load the most recent one
            LOGGER.info(f"Loading latest model from: {load_dir}")

        tree_path = os.path.join(load_dir, "xmrtree.pkl")
        assert os.path.exists(tree_path), f"XMRTree path {tree_path} does not exist"

        # Load metadata from pickle
        with open(tree_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)

        # Load large embeddings separately
        for attr in ["text_embeddings", "transformer_embeddings", "concatenated_embeddings"]:
            emb_path = os.path.join(load_dir, f"{attr}.npy")
            if os.path.exists(emb_path):
                setattr(model, attr, np.load(emb_path, allow_pickle=True))

        # Load children trees
        children_dir = os.path.join(load_dir, "children")
        if os.path.exists(children_dir):
            for child_name in os.listdir(children_dir):
                child_path = os.path.join(children_dir, child_name)
                model.children[child_name] = cls.load(child_path)

        LOGGER.info(f"Model loaded successfully from {load_dir}")
        return model
    
    def set_text_embeddings(self, text_embeddings):
        self.text_embeddings = text_embeddings
        
    def set_transformer_embeddings(self, transformer_embeddings):
        self.transformer_embeddings = transformer_embeddings
        
    def set_concatenated_embeddings(self, concatenated_embeddings):
        self.concatenated_embeddings = concatenated_embeddings
        
    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer
        
    def set_clustering_model(self, clustering_model):
        self.clustering_model = clustering_model
        
    def set_classifier_model(self, classifier_model):
        self.classifier_model = classifier_model
    
    def set_test_split(self, test_split):
        self.test_split = test_split
        
    def set_children(self, idx, child_tree):
        self.children[idx] = child_tree
    
    def set_depth(self, depth):
        self.depth = depth
    
    def is_empty(self):
        return self.clustering_model is None
    
    def __str__(self):
        """String representation of XMRTree."""
        return (f"XMRTree(depth={self.depth}, "
                f"num_children={len(self.children)}, "
                f"clustering_model={type(self.clustering_model).__name__ if self.clustering_model else None}, "
                f"classifier_model={type(self.classifier_model).__name__ if self.classifier_model else None}, "
                f"text_embedding_shape={self.text_embeddings.shape if self.text_embeddings is not None else None}, "
                f"transformer_embedding_shape={self.transformer_embeddings.shape if self.transformer_embeddings is not None else None}, "
                f"concatenated_embedding_shape={self.concatenated_embeddings.shape if self.concatenated_embeddings is not None else None})")
        
    