from featurization.transformers import BertVectorizer
from src.featurization.vectorizers import Vectorizer
from src.models.classifier_wrapper.classifier_model import ClassifierModel
from src.models.cluster_wrapper.clustering_model import ClusteringModel


class pipeline():
    
    def __init__(self,
                 text_vectorizer: Vectorizer,
                 transformer: BertVectorizer,
                 cluster_model: ClusteringModel,
                 ranker_model: ClassifierModel
                 ):
        
        self.text_vectorizer = text_vectorizer
        self.transformer = transformer
        self.cluster_model = cluster_model
        self.ranker_model = ranker_model
        
    
