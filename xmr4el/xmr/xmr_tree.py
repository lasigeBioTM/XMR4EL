import os
import pickle
import logging
import glob

import numpy as np

from datetime import datetime
from collections import Counter

from xmr4el.featurization.vectorizers import Vectorizer
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class XMRTree:

    def __init__(
        self,
        label_matrix=None,
        label_enconder=None,
        pifa_embeddings=None,
        text_embeddings=None,
        transformer_embeddings=None,
        concatenated_embeddings=None,
        kb_indices=None,
        vectorizer=None,
        clustering_model=None,
        cluster_id=None,
        classifier_model=None,
        test_split=None,
        children=None,  # Dictionary of XMRTree nodes
        depth=0,
    ):

        self.label_matrix = label_matrix
        self.label_enconder = label_enconder
        self.pifa_embeddings = pifa_embeddings

        self.text_embeddings = text_embeddings
        self.transformer_embeddings = transformer_embeddings
        self.concatenated_embeddings = concatenated_embeddings
        self.kb_indices = kb_indices
        
        self.vectorizer = vectorizer

        self.clustering_model = clustering_model
        self.cluster_id = cluster_id
        self.cluster_id_look_up = {}

        self.classifier_model = classifier_model
        self.test_split = test_split

        self.children = children if children is not None else {}
        self.depth = depth
        
        # Enumerate the clusters
        self._cluster_counter = 0

    def save(self, save_dir="data/saved_trees", child_tree=False):
        """Save trained XMRTree model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
            child_tree (bool): If True, saves the tree without the timestamp (for child trees).
        """
        if not child_tree:
            # Generate a timestamped directory for the main tree
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(save_dir, f"{self.__class__.__name__}_{timestamp}")

        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Prepare state dictionary without embeddings
        state = self.__dict__.copy()

        # Save models separately
        models = ["vectorizer", "clustering_model", "classifier_model"]
        models_data = [getattr(self, model_name, None) for model_name in models]

        for idx, model in enumerate(models_data):
            if model is not None:  # Ensure model exists before saving
                print(type(model))
                model.save(os.path.join(save_dir, models[idx]))

        # Remove saved models from state dictionary
        for model in models:
            state.pop(model, None)

        # Save embeddings separately (if they exist)
        for attr in [
            "text_embeddings",
            "transformer_embeddings",
            "concatenated_embeddings",
        ]:
            emb_data = getattr(self, attr, None)
            if emb_data is not None:
                np.save(os.path.join(save_dir, f"{attr}.npy"), emb_data)
                state.pop(attr, None)  # Remove from state to avoid duplication
            else:
                LOGGER.warning(f"{attr} is None and will not be saved.")

        state.pop("children", {})

        # Save child trees only if there are children
        if self.children:
            children_dir = os.path.join(save_dir, "children")
            os.makedirs(children_dir, exist_ok=True)
            for idx, child in self.children.items():
                child_save_dir = os.path.join(
                    children_dir, f"child_{idx}"
                )  # Unique directory for each child
                child.save(child_save_dir, child_tree=True)

        # Save the metadata using pickle
        with open(os.path.join(save_dir, "xmrtree.pkl"), "wb") as fout:
            pickle.dump(state, fout)

        LOGGER.info(f"Model saved successfully at {save_dir}")

    @classmethod
    def load(cls, load_dir=None, child_tree=False):
        """Load a saved XMRTree model from disk.

        Args:
            load_dir (str): Specific folder to load from. If None, loads the latest saved model.

        Returns:
            XMRTree: The loaded object.
        """
        if load_dir is None:
            # Find the most recent directory
            all_saves = sorted(
                glob.glob(os.path.join("data/saved_trees", f"{cls.__name__}_*")),
                reverse=True,
            )
            if not all_saves:
                raise FileNotFoundError(
                    f"No saved {cls.__name__} models found in saved_trees."
                )
            load_dir = all_saves[0]  # Load the most recent one
            LOGGER.info(f"Loading latest model from: {load_dir}")

        # Path to the main tree's metadata
        tree_path = os.path.join(load_dir, "xmrtree.pkl")
        assert os.path.exists(tree_path), f"XMRTree path {tree_path} does not exist"

        # Load metadata from pickle
        with open(tree_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)

        # Load models separately
        if not child_tree:
            setattr(
                model,
                "vectorizer",
                Vectorizer.load(os.path.join(load_dir, "vectorizer")),
            )

        setattr(
            model,
            "clustering_model",
            ClusteringModel.load(os.path.join(load_dir, "clustering_model")),
        )
        setattr(
            model,
            "classifier_model",
            ClassifierModel.load(os.path.join(load_dir, "classifier_model")),
        )

        # Load embeddings separately if they exist
        for attr in [
            "text_embeddings",
            "transformer_embeddings",
            "concatenated_embeddings",
        ]:
            emb_path = os.path.join(load_dir, f"{attr}.npy")
            if os.path.exists(emb_path):
                setattr(model, attr, np.load(emb_path, allow_pickle=True))
                LOGGER.debug(f"Loaded {attr} embeddings from {emb_path}")
            else:
                pass
                LOGGER.debug(f"{attr} not found at {emb_path}")

        # Load children trees if present
        children_dir = os.path.join(load_dir, "children")
        if os.path.exists(children_dir):
            for child_name in os.listdir(children_dir):
                child_tree_path = os.path.join(children_dir, child_name)
                if os.path.isdir(child_tree_path):  # Make sure it's a directory
                    try:
                        child_index = int(
                            child_name.replace("child_", "")
                        )  # Convert 'child_0' -> 0
                    except ValueError:
                        LOGGER.debug(f"Skipping unexpected child folder: {child_name}")
                        continue  # Ignore folders that don't match 'child_X' format

                    LOGGER.debug(
                        f"Loading child tree from {child_tree_path} as index {child_index}"
                    )
                    child_model = cls.load(
                        child_tree_path, child_tree=True
                    )  # Recursively load child
                    model.children[child_index] = child_model

        LOGGER.info(f"Model loaded successfully from {load_dir}")
        return model

    def set_label_matrix(self, label_matrix):
        self.label_matrix = label_matrix
        
    def set_label_enconder(self, label_enconder):
        self.label_enconder = label_enconder
        
    def set_pifa_embeddings(self, pifa_embeddings):
        self.pifa_embeddings = pifa_embeddings

    def set_text_embeddings(self, text_embeddings):
        self.text_embeddings = text_embeddings

    def set_transformer_embeddings(self, transformer_embeddings):
        self.transformer_embeddings = transformer_embeddings

    def set_concatenated_embeddings(self, concatenated_embeddings):
        self.concatenated_embeddings = concatenated_embeddings
        
    def set_kb_indices(self, kb_indices):
        self.kb_indices = kb_indices

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

    def is_leaf(self):
        return self.children is None
    
    def enumerate_clusters(self):
        enum = XMRTreeEnumerator()
        self.cluster_id_look_up = enum.enumerate(self)
        

    def __str__(self, level=0):
        """Recursively generates a string representation of the tree with all attribute states."""
        indent = "  " * level  # Indentation for hierarchy visualization
        """
        attributes = {
            "text_embeddings": self.text_embeddings is not None,
            "transformer_embeddings": self.transformer_embeddings is not None,
            "concatenated_embeddings": self.concatenated_embeddings is not None,
            "vectorizer": self.vectorizer is not None,
            "clustering_model": self.clustering_model is not None,
            "classifier_model": self.classifier_model is not None,
            "test_split": self.test_split is not None,
        }
        """
        
        attributes = {
            "labels": Counter(self.clustering_model.labels()),
            "kb_indices": "Not yet initialized" if self.kb_indices is None else len(self.kb_indices), # "True" if self.kb_indices is not None else "False",
            "label_matrix": "True" if self.label_matrix is not None else "False",
            "label_enconder": "True" if self.label_enconder is not None else "False",
            "pifa_embeddings": "True" if self.pifa_embeddings is not None else "False",
            "cluster_id": "Not decided yet" if self.cluster_id is None else self.cluster_id,
            "cluster_id_look_up": "Only in root" if self.cluster_id_look_up is None else self.cluster_id_look_up
        }

        tree_str = f"{indent * 2}- XMRTree (depth={self.depth}, children={len(self.children)}) [{attributes}]\n"

        # Recursively add children
        for key, child in self.children.items():
            tree_str += f"{indent}  |- Child: {key}\n{child.__str__(level + 1)}"

        return tree_str

    def __repr__(self):
        """Short representation for debugging."""
        return f"XMRTree(depth={self.depth}, children={len(self.children)})"
        

class XMRTreeEnumerator:
    
    def __init__(self):
        self.counter = 0
        self.id_to_node = {}
        
    def enumerate(self, root):
        self._enumerate_recursive(root)
        print(self.id_to_node)
        return self.id_to_node
            
    def _enumerate_recursive(self, node):
        node.cluster_id = self.counter
        self.id_to_node[self.counter] = node
        self.counter += 1
        for child in node.children:
            self._enumerate_recursive(child)