import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.app.commandhelper import MainCommand
from src.app.utils import create_hierarchical_clustering, create_hierarchical_linear_model, create_vectorizer, load_train_and_labels_file, load_vectorizer
from src.featurization.preprocessor import Preprocessor
from src.machine_learning.clustering import Clustering
from src.machine_learning.cpu.ml import KMeansCPU
from src.machine_learning.hierarchical_clustering import DivisiveHierarchicalClustering
from src.machine_learning.hierarchical_linear_model import HierarchicalLinearModel



"""

    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,

"""
def main():
    args = MainCommand().run()
    kb_type = "medic"
    kb_location = "data/raw/mesh_data/medic/CTD_diseases.tsv"

    label_filepath = "data/raw/mesh_data/medic/labels.txt"
    training_filepath = "data/train/disease/train_Disease_100.txt"

    parsed_train_data = load_train_and_labels_file(training_filepath, label_filepath)

    # Dense Matrix
    Y_train = [str(parsed) for parsed in parsed_train_data["labels"]]
    X_train = [str(parsed) for parsed in parsed_train_data["corpus"]]
    
    # X_train = [lst[:5] for lst in X_train]

    # Turn on erase mode when training
    # vectorizer_model = create_vectorizer(X_train)
    vectorizer_model = load_vectorizer("data/processed/vectorizer/vectorizer.pkl")

    X_train_feat = vectorizer_model.predict(X_train)

    print(X_train_feat.shape)

    # hierarchical_clustering_model = create_hierarchical_clustering(X_train_feat.astype(np.float32))
    
    # Training of the Agglmomerative Clustering or Birch (Impossible to Read Birch)
    # silhouette avg: 0.0017
    
    # First Top-K Score: 0.9992447129909365
    # hierarchical_clustering_model = Clustering.load("data/processed/clustering/clustering_agglo_train_100.pkl").model
    # Y_train_feat = hierarchical_clustering_model.labels_
    
    # hierarchical_clustering_model = KMeansCPU.train(X_train_feat).model
    
    # First Top-K Score: 0.9920694864048338
    # centroid_array, Y_train_feat, sil_score = DivisiveHierarchicalClustering.fit(X_train_feat)
    
    Y_train_feat = DivisiveHierarchicalClustering.fit(X_train_feat).labels
    
    # print(Y_train_feat)
    
    
    # print(divisive_hierarchical_labels)
    
    # scores = KmeansRanker().ranker(X_train_feat)
    
    # print(scores)
    
    # hcm_labels = hierarchical_clustering_model.labels_
    
    # silhouette_avg = silhouette_score(X_train_feat, hcm_labels)
    
    # print(silhouette_avg)
    
    # print(silhouette_avg)
    
    # labels_df = pd.DataFrame(hierarchical_clustering_model.labels_, columns=['cluster_label'])
    
    """ Agglomerative Clustering
    cluster_label  count
        0           161
        1           969
        2           407
        3           10334
        4           231
        5           568
        6           216
        7            22
        8            55
        9            24
        10           53
        11           43
        12           27
        13           45
        14           63
        15           22
    """
    

    # Embeddings -> X_train_feat ClusterLabels -> Y_train_feat (is this data, more labels than embeddings)
    # hierarchical_linear_model = create_hierarchical_linear_model(X_train_feat, Y_train_feat[:13240], 2)
    
    print("Starting HML")
    hierarchical_linear_model = HierarchicalLinearModel.fit(X_train_feat, Y_train_feat)
    
    print("Top-K Score", hierarchical_linear_model.top_k_score)
    


if __name__ == "__main__":
    main()

