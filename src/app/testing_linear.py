import numpy as np

from src.models.cluster_wrapper.clustering_model import ClusteringModel
from src.models.linear_wrapper.linear_model import LinearModel

test_input_path = "data/processed/vectorizer/test_input_embeddings.npy"

logistic_regression_config = {'type': 'sklearnlogisticregression', 'kwargs':{'random_state': 0}}    

hierarchical_linear_config = {
    'type': 'hierarchicallinear', 
    'kwargs':{
        'model': logistic_regression_config
        }
    }    

divisive_cluster_model = ClusteringModel.load("test_cl").model

dtree = divisive_cluster_model.dtree

linear_model = LinearModel.train(dtree, None, hierarchical_linear_config)

linear_model.save("test_linear")

linear_model = linear_model.load("test_linear").model

# test_input = np.load(test_input_path, allow_pickle=True)

# cluster_predictions = divisive_cluster_model.predict(test_input)

# predictions = linear_model.predict(cluster_predictions, test_input)

# print(predictions)



