import numpy as np

from src.models.cluster_wrapper.clustering_model import ClusteringModel

onnx_gpu_embeddigns_filepath = "data/processed/vectorizer/biobert_onnx_dense_disease100_gpu.npy"

trn_corpus = np.load(onnx_gpu_embeddigns_filepath, allow_pickle=True)

kmeans_config = {'type': 'sklearnkmeans', 'kwargs':{'random_state': 0}}    

divisive_clustering_config = {
    'type': 'divisiveclustering', 
    'kwargs':{
        'depth': 1,
        'model': kmeans_config
        }
    }    

divisive_model = ClusteringModel.train(trn_corpus, divisive_clustering_config)

divisive_model.save("test_cl")

dmodel = divisive_model.load("test_cl")

predictions = dmodel.model.predict([trn_corpus[0]])


