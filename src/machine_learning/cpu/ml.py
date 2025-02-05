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
        'verbose': 0
    }

    @classmethod
    def create_model(cls, kwargs={}):
        params = {**cls.DEFAULTS, **kwargs}
        return cls(
            model = LogisticRegression(**params), 
            model_type = 'LogisticRegressionCPU'
        )

    def fit(self, X_train, Y_train):
        self.model = force_multi_core_processing_linear_models(self.model, X_train, Y_train)
        return self.model    
    
class KMeansCPU(Clustering):
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
            model = KMeans(**params),
            model_type = 'KMeansCPU'
        )

    def fit(self, X_train):
        self.model = force_multi_core_processing_clustering_models(self.model, X_train)
        return self.model

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
    