import os
import logging
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

from src.models.dtree import DTree
from src.models.linear_wrapper.linear_model import LinearModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HierarchicalLinear(LinearModel):
    
    
    class Config():
        
        def __init__(self, top_k=3, top_k_threshold=0.15, model=None):
            
            self.top_k = top_k
            self.top_k_threshold = top_k_threshold
        
            if model is None:
                self.model = {
                'type': 'sklearnlogisticregression', 
                'kwargs': {}
            }
            else:
                self.model = model
            
        def to_dict(self):
            return {
                "top_k": self.top_k,
                "top_k_threshold": self.top_k_threshold,
                "model": self.model
            }
            
        def __str__(self):
            return f"Config(top_k={self.top_k}, top_k_threshold={self.top_k_threshold}, model={self.model})"
    
    def __init__(self, config=None, dtree=None):
        self.config = config
        self.dtree: DTree = dtree
        
    def save(self, save_dir):
        """Save trained Hierarchical Linear model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "hierarchical_linear_model.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved Hierarchical Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """
        
        LOGGER.info(f"Loading Hierarchical Linear model from {load_dir}")
        linear_path = os.path.join(load_dir, "hierarchical_linear_model.pkl")
        assert os.path.exists(linear_path), f"linear model path {linear_path} does not exist"
        
        with open(linear_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        
        model.config = cls.Config(**model.config)
        return model
    
    # trn_corpus is a placeholder
    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        defaults = cls.Config().to_dict()
        
        try:
            config = {**defaults, **config}
            if trn_corpus is None or not isinstance(trn_corpus, DTree):
                raise Exception(f"Dtree not found")
            
            model = HierarchicalLinear(config=cls.Config(**config))
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for HierarchicalLinear"
            )
        model = model.fit(trn_corpus, dtype)
        return cls(config, model)
    
    def fit(self, dtree, dtype):
        """
        Fit the hierarchical tree, recursively training models for each node in the tree.

        Args:
            dtree: The root of the hierarchical tree (DTree).
            dtype: The data type for training the model (e.g., classifier type).

        Returns:
            DTree: A new tree with updated models and data classifications.
        """
        self.dtree = dtree
        config = self.config
        
        def train_subtree(dtree, config, dtype):
            """Recursively train the model on each node of the tree."""
            
            # Skip node if no cluster_node exists
            if dtree.node.cluster_node is None:
                LOGGER.debug(f"Skipping node at depth {dtree.depth} because cluster_node is None.")
                return

            X = dtree.node.cluster_node.cluster_points
            Y = dtree.node.cluster_node.labels
            
            if X is None or Y is None:
                LOGGER.debug(f"Skipping node at depth {dtree.depth} because data is missing.")
                return 

            # Train-test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=0.2, random_state=42, stratify=Y)
            except ValueError as e:
                LOGGER.error(f"Error in train-test split at depth {dtree.depth}: {e}")
                return 
            
            # Train the linear model
            try:
                linear_model = LinearModel.train(X_train, y_train, config.model, dtype=dtype).model
            except Exception as e:
                LOGGER.error(f"Model training failed at depth {dtree.depth}: {e}")
                return
            
            # Store Results
            test_split = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            
            # Create a new DTree node with the trained model and results
            dtree.node.set_linear_node(linear_model, test_split)
            
            for _, child in dtree.children.items():
                train_subtree(child, config, dtype)
                
        
        train_subtree(self.dtree, config, dtype)
        return self.dtree
    
    def predict(self, cluster_predictions, test_input):
        """
        Recursively traverses the hierarchical tree to classify an input embedding.
        """
        
        def predict_input(linear_model, test_input, top_k):
            probs = linear_model.predict_proba(test_input)[0]  # Get class probabilities

            # Get top-k indices sorted by confidence
            top_k_indices = np.argsort(probs)[::-1][:top_k]
            top_k_labels = linear_model.classes_[top_k_indices].tolist()  # Retrieve corresponding labels
            top_k_confidences = probs[top_k_indices].tolist()  # Retrieve corresponding confidence scores

            return top_k_labels, top_k_confidences
        
        root_dtree = self.dtree
        top_k = self.config.top_k
        predictions = []
        
        for cluster_pred in cluster_predictions:
            current_dtree = root_dtree
            pred_len = len(cluster_pred)
            
            for idx in range(pred_len):
                if current_dtree.children is None:
                    print(current_dtree)
                if cluster_pred[idx] in current_dtree.children:
                    current_dtree = current_dtree.children[cluster_pred[idx]]
                else:
                    top_k_label, top_k_confidences = predict_input(current_dtree.node.linear_node.model.model, test_input, top_k)
                    predictions.append({
                        'predicted_labels': top_k_label,
                        'top_k_confidences': top_k_confidences,
                        'true_label': cluster_pred[-1]
                    })
                
        return predictions
        
    
        
        
        
        