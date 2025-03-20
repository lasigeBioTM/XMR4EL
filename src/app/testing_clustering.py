import numpy as np

from src.models.cluster_wrapper.clustering_model import ClusteringModel

onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease500_gpu.npy"

trn_corpus = np.load(onnx_gpu_embeddigns_filepath, allow_pickle=True)

kmeans_config = {'type': 'sklearnminibatchkmeans', 'kwargs':{'random_state': 0, 
                                                             'max_iter': 500}}

divisive_clustering_config = {
    'type': 'divisiveclustering', 
    'kwargs':{
        'depth': 1,
        'model': kmeans_config
        }
    }    

divisive_model = ClusteringModel.train(trn_corpus, divisive_clustering_config)

print(divisive_model.model.dtree)

divisive_model.save("test_cl")

dmodel = ClusteringModel.load("test_cl")

# test_input_path = "data/processed/vectorizer/test_input_embeddings.npy"

# test_input = np.load(test_input_path, allow_pickle=True)

# predictions = dmodel.model.predict(test_input)


