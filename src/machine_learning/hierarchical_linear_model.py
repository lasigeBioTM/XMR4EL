import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score

class HierarchicalLinearModel:
    """
    A hierarchical learning model that combines clustering and linear models.
    Iteratively refines clusters and predicts probabilities using a linear model.
    """

    def __init__(self, linear_model_type=None, tree_node=None, top_k=3, gpu_usage=False):
        """Initialization"""
        self.linear_model_type = linear_model_type
        self.tree_node = tree_node
        self.top_k = top_k
        self.gpu_usage = gpu_usage

    def save(self, hierarchical_folder):
        """Save the model and its parameters to a file """
        os.makedirs(os.path.dirname(hierarchical_folder), exist_ok=True)
        with open(hierarchical_folder, 'wb') as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, hierarchical_folder):
        """Load a saved model from a file."""
        if not os.path.exists(hierarchical_folder):
            raise FileNotFoundError(f"Hierarchical folder {hierarchical_folder} does not exist.")
        
        with open(hierarchical_folder, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def fit(cls, tree_node, linear_model_factory, config={}):
        """
        Train the model using clustering and logistic regression with iterative label refinement.
        """
        
        DEFAULTS = {
            'max_iter': 100,
            'min_leaf_size': 10,
            'max_leaf_size': 20,
            'random_state': 0,
            'top_k': 3,
            'top_k_threshold': 0.15,
            'gpu_usage': False,
        }
        
                
        config = {**DEFAULTS, **config}
        
        min_leaf_size = config['min_leaf_size']
        random_state = config['random_state']
        top_k = config['top_k']
        gpu_usage = config['gpu_usage']
        
        cls.__train_all_tree_nodes(
            tree_node=tree_node, 
            linear_model_factory=linear_model_factory,
            top_k=top_k,
            )
            
        return cls(
            linear_model_type=linear_model_factory.model_type,
            tree_node=tree_node,
            top_k=top_k,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, predicted_labels, top_k):        
        preds = self.linear_model.predict_proba(embeddings)
        return self.__top_k_score_sklearn(preds, predicted_labels, top_k)

    # Embeddings, Labels
    @classmethod
    def __train_linear_model(cls, X, Y, linear_model_factory, top_k):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        linear_model = linear_model_factory.fit(X_train, y_train)
            
        # Predict probabilities
        y_proba = linear_model.predict_proba(X_test)
        
        print(f"y_test labels: {np.unique(y_test)}, Y labels: {np.unique(Y)}")
        print(y_proba)
        
        all_labels = np.unique(Y)
        
        # Y_test doesnt have all the labels, have to fix
        # Compute top-k accuracy
        top_k_score = top_k_accuracy_score(y_test, y_proba, k=top_k, normalize=True, labels=all_labels)       
        
        return {'linear_model': linear_model, 
                'top_k_score': top_k_score, 
                'X_test': X_test, 
                'y_test': y_test}      
    
    @classmethod
    def __train_all_tree_nodes(cls, tree_node, linear_model_factory, top_k=3):
        """
        Apply logistic regression training to this node and all child nodes.
        """
        # Check if cluster_node is None
        if tree_node.cluster_node is None:
            # print(f"Skipping node at depth {tree_node.depth} because cluster_node is None.")
            return

        # Extract data from the current node
        X = tree_node.cluster_node.cluster_points
        Y = tree_node.cluster_node.labels

        # Check if data exists
        if X is None or Y is None:
            # print(f"Skipping node at depth {tree_node.depth} because data is missing.")
            return

        # Train model for the current node
        linear_model_data = cls.__train_linear_model(X, Y, linear_model_factory, top_k)
        
        tree_node.set_linear_node(linear_model_data['linear_model'],
                                     linear_model_data['top_k_score'],
                                     linear_model_data['X_test'],
                                     linear_model_data['y_test'])

        # Apply to child nodes recursively
        for child in tree_node.children.values():
            cls.__train_all_tree_nodes(child, linear_model_factory, top_k)
    
    @staticmethod
    def __top_k_score_sklearn(pred_probs, true_labels, k=3):
        """Calculate top-k score using sklearn's top_k_accuracy_score as a helper."""
        return top_k_accuracy_score(true_labels, pred_probs, k=k, labels=np.arange(pred_probs.shape[1]))
