import os
import pickle
import numpy as np

from collections import Counter
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, pairwise_distances_argmin_min, silhouette_score

class HierarchicalLinearModel:
    """
    A hierarchical learning model that combines clustering and linear models.
    Iteratively refines clusters and predicts probabilities using a linear model.
    """

    def __init__(self, linear_model=None, linear_model_type=None, labels=None,
                 top_k_score=None, top_k=3, x_test=None, y_test=None, gpu_usage=False):
        """Initialization"""
        self.linear_model = linear_model
        self.linear_model_type = linear_model_type
        self.labels = labels
        self.top_k_score = top_k_score
        self.top_k = top_k
        self.x_test = x_test
        self.y_test = y_test
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
    def fit(cls, X, Y, linear_model_factory, clustering_model_factory, config={}):
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
        top_k_threshold = config['top_k_threshold']
        gpu_usage = config['gpu_usage']
        
        
        linear_model_dict = cls.__train_linear_model(
            X, 
            Y, 
            linear_model_factory, 
            top_k, 
            top_k_threshold
        )
        
        best_linear_model = linear_model_dict['linear_model']
        best_labels = linear_model_dict['labels']
        best_top_k_indices = linear_model_dict['top_k_indices']
        best_top_k_score = linear_model_dict['top_k_score']
        best_X_test = linear_model_dict['X_test']
        best_y_test = linear_model_dict['y_test'] 
        
        # ok
        while True:
            
            new_labels_encoded = cls.__refine_top_k_clusters(
                X, 
                best_labels, 
                best_linear_model, 
                clustering_model_factory, 
                min_leaf_size, 
                random_state=random_state,
                k=top_k, 
                threshold=0.8
            )
            
            print(f"\nOld Silhouette Score: {silhouette_score(X, best_labels)}")
            print(f"New Silhouette Score: {silhouette_score(X, new_labels_encoded)}")
            
            # print(f"__fit, new_labels_encoded, Clusters: {np.unique(new_labels_encoded)}, Number of Clusters: {np.unique(new_labels_encoded).shape}")
            
            new_linear_model_dict = cls.__train_linear_model(
            X, 
            new_labels_encoded, 
            linear_model_factory, 
            top_k, 
            top_k_threshold
            )
        
            
        return cls(
            linear_model=best_linear_model, 
            linear_model_type=linear_model_factory.model_type, 
            labels=best_labels, 
            top_k_score=best_top_k_score, 
            top_k=top_k,
            x_test = best_X_test,
            y_test = best_y_test,
            gpu_usage=gpu_usage
        )
    
    def predict(self, embeddings, predicted_labels, top_k):        
        preds = self.linear_model.predict_proba(embeddings)
        return self.__top_k_score_sklearn(preds, predicted_labels, top_k)

    # Embeddings, Labels
    @classmethod
    def __train_linear_model(cls, X, Y, linear_model_factory, top_k, top_k_threshold):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        linear_model = linear_model_factory.fit(X_train, y_train)
            
        # Predict probabilities
        y_proba = linear_model.predict_proba(X_test)
        
        # Get top-k predictions
        top_k_indices = cls.__get_top_k_indices(y_proba, top_k, top_k_threshold)
        
        # Compute top-k accuracy
        top_k_score = top_k_accuracy_score(y_test, y_proba, k=top_k, normalize=True)       
        
        # print(f"__train_linear_model Labels: {np.unique(top_k_indices)}")
        
        return {'linear_model': linear_model, 
                'labels': Y, 
                'top_k_indices': top_k_indices, 
                'top_k_score': top_k_score, 
                'X_test': X_test, 
                'y_test': y_test}      

    
    @staticmethod
    def __get_top_k_indices(y_proba, k, top_k_threshold):
        """Returns the top-k indices for each sample based on predicted probabilities."""
        filtered_proba = np.where(y_proba >= top_k_threshold, y_proba, -np.inf)
        top_k_indices = np.argsort(filtered_proba, axis=1)[:, -k:][:, ::-1]
        return top_k_indices.tolist()
    
    @staticmethod
    def __top_k_score_sklearn(pred_probs, true_labels, k=3):
        """Calculate top-k score using sklearn's top_k_accuracy_score as a helper."""
        return top_k_accuracy_score(true_labels, pred_probs, k=k, labels=np.arange(pred_probs.shape[1]))
