from src.featurization.preprocessor import Preprocessor
from src.machine_learning.cpu.ml import AgglomerativeClusteringCPU, BirchCPU
from src.featurization.vectorizer import BioBertVectorizer, TfidfVectorizer
from src.machine_learning.clustering import Clustering
from src.machine_learning.hierarchical_linear_model import HieararchicalLinearModel


def load_train_and_labels_file(train_filepath, labels_filepath):
    print("Getting Processed Labels from Preprocessor")
    return Preprocessor.load_data_from_file(train_filepath, labels_filepath)

def create_vectorizer(corpus):
    vectorizer_path = "data/processed/vectorizer"
    print("Processing Vectorizer Algorithm")
    model = TfidfVectorizer.train(corpus)
    model.save(vectorizer_path)
    print("Saved Vectorizer\n")

    return model

def load_vectorizer(vectorizer_path):
    print("Trying to Load the Embeddings from Preprocessor")
    try:
        model = Preprocessor.load(vectorizer_path)
        print(f"Loaded Vectorizer, Type: {model.model_type}\n")
    except Exception as e:
        print(f"Could not load Vectorizer ({e}). Run the Training Script")
        exit()

    return model

def create_hierarchical_clustering(embeddings):
    clustering_path = "data/processed/clustering"
    print("Processing Hierarchical Clustering Algorithm")
    # Changing to Agglomerative Clustering
    model = AgglomerativeClusteringCPU.train(embeddings.toarray())
    model.save(clustering_path)
    print("Saved Cluster Labels")

    return model

def load_hierarchical_clustering(clustering_path):
    print("Trying to Load the Clustering Labels From Clustering")
    try:
        model = Clustering.load(clustering_path)
        print(f"Loaded Clustering Model, Type: {model.model_type}\n")
    except Exception as e:
        print(f"Could not load Clustering Model ({e}). Run the Training Script")
        exit()

    return model

def create_hierarchical_linear_model(X_train_feat, Y_train, k):
    hlm_path = "data/processed/hlm"

    print("Processing Hierarchical Linear Algorithm")
    model = HieararchicalLinearModel.execute_pipeline(X_train_feat, Y_train, k)
    model.save(hlm_path)
    print("Hierarchical Linear Model Saved")

    return model

def load_hierarchical_linear_model(hlm_path):
    try:
        model = HieararchicalLinearModel.load(hlm_path)
        print(f"Loaded HierarchicalLinearModel, Type: Express what clustering and linear model was used\n")
    except Exception as e:
        print(f"Could not load Hierarchical Linear Model ({e}). Run the Training Script")
        exit()
    
    return model

def run_machine_learning_matching(embeddings, clustering_labels):
    print("Trainning")
    # TrainCPU.train(embeddings, clustering_labels)
    HieararchicalLinearModel.execute_pipeline(embeddings, clustering_labels)

def run_ranking_model():
    pass