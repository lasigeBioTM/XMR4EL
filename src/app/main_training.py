import numpy as np
import pandas as pd

from src.app.commandhelper import MainCommand
from src.app.utils import create_hierarchical_clustering, create_hierarchical_linear_model, create_vectorizer, load_train_and_labels_file
from src.featurization.preprocessor import Preprocessor
from src.machine_learning.clustering import Clustering


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
    vectorizer_model = create_vectorizer(X_train)

    X_train_feat = vectorizer_model.predict(X_train)

    # print(X_train_feat.shape)

    hierarchical_clustering_model = create_hierarchical_clustering(X_train_feat.astype(np.float32))
    
    # hierarchical_clustering_model = Clustering.load("data/processed/clustering/clustering_old.pkl").model

    Y_train_feat = hierarchical_clustering_model.labels_
    
    print(pd.DataFrame(Y_train_feat))
    
    # print(Y_train_feat)
    
    # hierarchical_linear_model = create_hierarchical_linear_model(X_train_feat, Y_train_feat[:13240], 2)


if __name__ == "__main__":
    main()

