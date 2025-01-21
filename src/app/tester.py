import pandas as pd

from src.featurization.preprocessor import Preprocessor
from src.machine_learning.clustering import Clustering

train_filepath = "data/raw/mesh_data/medic/train_Disease_500.txt"
labels_filepath = "data/raw/mesh_data/medic/labels.txt"

cluster_model_path = "data/processed/clustering/clustering_old.pkl"



cluster_model = Clustering.load(cluster_model_path).model

print(pd.DataFrame(cluster_model.labels_))
