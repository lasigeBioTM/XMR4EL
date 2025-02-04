from sklearn.cluster import AgglomerativeClustering, KMeans,MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from src.machine_learning.cpu.joblib_runtime import force_multi_core_processing_clustering_models, force_multi_core_processing_linear_models
from src.machine_learning.clustering import Clustering
from src.machine_learning.classifier import Classifier


class AgglomerativeClusteringCPU(Clustering):
    
    @classmethod
    def fit(cls, X_train, defaults={}):
        if defaults == {}:
            defaults = {
                'n_clusters': 16,           
            }
            
        model = force_multi_core_processing_clustering_models(AgglomerativeClustering(**defaults), X_train)

        return cls(model=model, model_type='AgglomerativeClusteringCPU')

    def get_labels(self):
        return self.model.labels_
        
class LogisticRegressionCPU(Classifier):

    @classmethod
    def train(cls, X_train, Y_train, defaults={}):
        if defaults == {}:
             defaults = {
                'random_state': 0,
                'solver': 'lbfgs',
                'max_iter': 100,
                'verbose': 0
            }
        
        model = force_multi_core_processing_linear_models(LogisticRegression(**defaults), X_train, Y_train)
        return cls(model=model, model_type='LogisticRegressionCPU')
    
class KMeansCPU(Clustering):

    @classmethod
    def fit(cls, X_train, defaults={}):
        if defaults == {}:
            defaults = {
                'n_clusters': 16,
                'max_iter': 20,
                'random_state': 0,
                'n_init': 10,
            }
        model = force_multi_core_processing_clustering_models(KMeans(**defaults), X_train)
        return cls(model=model, model_type='KMeansCPU')
    
class MiniBatchKMeansCPU(Clustering):
    
    @classmethod
    def fit(cls, X_train, defaults={}):
        if defaults == {}:
            defaults = {
                'n_clusters': 16,
                'max_iter': 20,
                'random_state': 0,
                'n_init': 10,
            }
        model = force_multi_core_processing_clustering_models(MiniBatchKMeans(**defaults), X_train)
        return cls(model=model, model_type='MiniBatchKMeansCPU')
