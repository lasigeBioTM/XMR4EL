import numpy as np

from src.models.cluster_wrapper.clustering_model import ClusteringModel
from models.classifier_wrapper.classifier_model import LinearModel

test_input_path = "data/processed/vectorizer/test_input_embeddings.npy"

logistic_regression_config = {'type': 'sklearnlogisticregression', 'kwargs':{'random_state': 0}}    

hierarchical_linear_config = {
    'type': 'hierarchicallinear', 
    'kwargs':{
        'model': logistic_regression_config
        }
    }    

dmodel = ClusteringModel.load("test_cl")

dtree = dmodel.model.dtree

# print(divisive_cluster_model.dtree)

linear_model = LinearModel.train(dtree, None, hierarchical_linear_config)

linear_model.save("test_linear")

lmodel = LinearModel.load("test_linear")

test_input = np.load(test_input_path, allow_pickle=True)

cluster_predictions = dmodel.model.predict(test_input)

# print(cluster_predictions)

predictions = lmodel.model.predict(cluster_predictions, test_input)

print(predictions)



