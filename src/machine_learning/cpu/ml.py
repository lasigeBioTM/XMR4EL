from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from src.machine_learning.classifier import Classifier
from src.machine_learning.cpu.joblib_runtime import (
    force_multi_core_processing_clustering_models, 
    force_multi_core_processing_linear_models
)
from src.machine_learning.clustering import Clustering


class AgglomerativeClusteringCPU(Clustering):

    DEFAULTS = {
        'n_clusters': 16
        }
    
    @classmethod
    def create_model(cls, kwargs={}):
        params = {**cls.DEFAULTS, **kwargs}
        return cls(
            model = AgglomerativeClustering(**params), 
            model_type = 'AgglomerativeClusteringCPU'
        )

    def fit(self, X_train):
        self.model = force_multi_core_processing_linear_models(self.model, X_train)
        return self.model    

class LogisticRegressionCPU(Classifier):
    DEFAULTS = {
        'random_state': 0,
        'solver': 'lbfgs',
        'max_iter': 100,
        'verbose': 0,
        'n_jobs': -1
    }

    @classmethod
    def create_model(cls, kwargs={}):
        # Merge default parameters with any user-specified ones
        params = {**cls.DEFAULTS, **kwargs}
        
        # Return a new instance with the model initialized
        return cls(
            model=None,  # Model is initially None
            model_type='LogisticRegressionCPU',
            params=params
        )

    def fit(self, X_train, Y_train):
        # Recreate the model each time fit is called with the parameters
        self.model = LogisticRegression(**self.params)
        # Fit the model to the training data
        return self.model.fit(X_train, Y_train)
        
  
    
class KMeansCPU(Clustering):
    DEFAULTS = {
        'n_clusters': 16,
        'max_iter': 20,
        'random_state': 0,
    }

    @classmethod
    def create_model(cls, kwargs={}):
        params = {**cls.DEFAULTS, **kwargs}
        return cls(
            model=None,
            model_type='KMeansCPU',
            params=params
        )

    def fit(self, X_train):
        self.model = KMeans(**self.params)
        # Returns the model
        return force_multi_core_processing_clustering_models(self.model, X_train)
    
class MiniBatchKMeansCPU(Clustering):
    DEFAULTS = {
        'n_clusters': 16,
        'max_iter': 20,
        'random_state': 0,
        'n_init': 10
    }
    
    @classmethod
    def create_model(cls, kwargs={}):
        params = {**cls.DEFAULTS, **kwargs}
        return cls(
            model = MiniBatchKMeans(**params),
            model_type = 'MiniBatchKMeansCPU'
        )

    def fit(self, X_train):
        self.model = force_multi_core_processing_clustering_models(self.model, X_train)
        return self.model
    