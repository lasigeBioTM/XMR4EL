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

    # Turn on erase mode when training
    # vectorizer_model = create_vectorizer(X_train)
    vectorizer_model = load_vectorizer("data/processed/vectorizer/vectorizer.pkl")

    X_train_feat = vectorizer_model.predict(X_train)

    print(X_train_feat.shape)
    
    # Training of the Agglmomerative Clustering or Birch (Impossible to Read Birch)
    # silhouette avg: 0.0017
    
    # First Top-K Score: 0.9992447129909365
    hierarchical_clustering_model = Clustering.load("data/processed/clustering/clustering_agglo_train_100.pkl").model
    Y_train_feat = hierarchical_clustering_model.labels_

    # print("Starting Divisive Hierarchical Clustering")
    # divisive_hierarchical_clustering = DivisiveHierarchicalClustering.fit(X_train_feat)
    # Y_train_feat = divisive_hierarchical_clustering.labels
    
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
    
    
    # Meu metodo deve ser melhor em termos de clustering
    # Divisive Hierarchical Clustering -> Top-1 Score 0.8557401812688822 / Top-3 Score 0.9569486404833837 / Top-5 Score 0.9724320241691843
    # Agglomerative Clustering -> Top-1 Score 0.9686555891238671 / Top-3 Score 1.0 / Top-5 Score 1.0 
    print("Starting HML")
    hierarchical_linear_model = HierarchicalLinearModel.fit(X_train_feat, Y_train_feat, top_k=3)
    
    print(f"Top-{hierarchical_linear_model.top_k} Score {hierarchical_linear_model.top_k_score}")

if __name__ == "__main__":
    main()

