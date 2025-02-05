from src.machine_learning.classifier import Classifier
from src.machine_learning.clustering import Clustering

from cuml.cluster import KMeans
from cuml.linear_model import LogisticRegression

class KMeansGPU(Clustering):
    DEFAULTS = {
        'n_clusters': 16,
        'max_iter': 20,
        'random_state': 0,
        'n_init': 10,
        'output_type': 'numpy'
    }

    @classmethod
    def create_model(cls, kwargs):
        params = {**cls.DEFAULTS, **kwargs}
        return cls(
            model = KMeans(**params),
            model_type = 'KMeansGPU'
        )

    def fit(self, X_train):
        self.model = self.model.fit(X_train.toarray())
        return self.model
    
class LogisticRegressionGPU(Classifier):

    DEFAULTS = {
        'random_state': 0,
        'solver': 'qn',
        'max_iter': 1000,
        'verbose': 0,
        'penalty': 'l2',
        'output_type': 'numpy'
    }

    @classmethod
    def create_model(cls, kwargs):
        params = {**cls.DEFAULTS, **kwargs}
        params['max_inter'] *= 10
        return cls(
            model = LogisticRegression(**params), 
            model_type = 'LogisticRegressionGPU'
        )

    def fit(self, X_train, Y_train):
        self.model = self.model.fit(X_train.toarray(), Y_train)
        return self.model   
    
