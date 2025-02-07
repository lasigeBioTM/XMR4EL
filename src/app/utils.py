import subprocess
import shutil
import os

from src.machine_learning.hierarchical_clustering import DivisiveHierarchicalClustering
from src.featurization.preprocessor import Preprocessor
from src.featurization.vectorizer import BioBertVectorizer, TfidfVectorizer
from src.machine_learning.hierarchical_linear_model import HierarchicalLinearModel


def is_cuda_available():
    # Default to using CPU models and set gpu_available to False
    gpu_available = False

    # Check if nvidia-smi is available on the system
    if shutil.which('nvidia-smi'):
        try:
            # Run the nvidia-smi command to check for an NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            gpu_available = True
        except subprocess.CalledProcessError as e:
            # Print the error message and continue execution
            print(f"GPU acceleration is unavailable: {e}. Defaulting to CPU models.")
    else:
        print("nvidia-smi command not found. Assuming no NVIDIA GPU.")

    return gpu_available

GPU_AVAILABLE = is_cuda_available()

def load_train_and_labels_file(train_filepath, labels_filepath):
    print("Getting Processed Labels from Preprocessor")
    return Preprocessor.load_data_from_file(train_filepath, labels_filepath)

def create_bio_bert_vectorizer(corpus, directory_embeddings, directory_cpu_onnx_model, output_embeddings_file):
    print("Running BioBert")
    assert os.path.exists(directory_embeddings),f"{directory_embeddings} does not exist"
    
    if GPU_AVAILABLE:
        embeddings = BioBertVectorizer.predict_gpu(corpus)
        Preprocessor.save_biobert_labels(embeddings, directory_embeddings)
    else:
        output_prefix = output_embeddings_file.split('.')[0]   
        embeddings = BioBertVectorizer.predict_cpu(corpus=corpus, 
                                                   directory_cpu_onnx_model=directory_cpu_onnx_model, 
                                                   output_prefix=output_prefix)
        Preprocessor.save_biobert_labels(embeddings, directory_embeddings)
    print("Saved BioBert Embeddings")
    return embeddings

def load_bio_bert_vectorizer(directory):
    print("Loading BioBert Labels")
    return Preprocessor.load_biobert_labels(directory)
    
def create_tfidf_vectorizer(corpus, directory):
    print("Running train on TF-IDF Vectorizer")
    assert os.path.exists(directory),f"{directory} does not exist"
    model = TfidfVectorizer.train(corpus)
    model.save(directory)
    print("Saved TF-IDF Model")
    return model

def load_tdidf_vectorizer(directory):
    print("Loading TF-IDF Vectorizer")
    assert os.path.exists(directory),f"{directory} does not exist"
    model = Preprocessor.load(directory)
    return model

def create_hierarchical_clustering(X_train_feat):
    if GPU_AVAILABLE:
        print("Processing Hierarchical Clustering Algorithm with GPU SUPPORT")
        
        from src.machine_learning.gpu.ml import KMeansGPU
        
        divisive_hierarchical_clustering = DivisiveHierarchicalClustering.fit(X_train_feat, CLUSTERING_MODEL=KMeansGPU.create_model())
    else:
        print("Processing Hierarchical Clustering Algorithm with CPU SUPPORT")
        
        from src.machine_learning.cpu.ml import KMeansCPU
        
        divisive_hierarchical_clustering = DivisiveHierarchicalClustering.fit(X_train_feat, CLUSTERING_MODEL=KMeansCPU.create_model())
    return divisive_hierarchical_clustering.labels

def create_hierarchical_linear_model(X_train_feat, Y_train_feat, k):
    
    if GPU_AVAILABLE:
        
        from src.machine_learning.gpu.ml import KMeansGPU, LogisticRegressionGPU
        
        print("Processing Hierarchical Linear Model with GPU SUPPORT")
        hierarchical_linear_model = HierarchicalLinearModel.fit(
            X_train_feat, 
            Y_train_feat, 
            LINEAR_MODEL=LogisticRegressionGPU.create_model(), 
            CLUSTERING_MODEL=KMeansGPU.create_model(), 
            top_k=k
        )
    else:
        
        from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU
        
        print("Processing Hierarchical Linear Model with CPU SUPPORT")
        hierarchical_linear_model = HierarchicalLinearModel.fit(
            X_train_feat, 
            Y_train_feat, 
            LINEAR_MODEL=LogisticRegressionCPU.create_model(), 
            CLUSTERING_MODEL=KMeansCPU.create_model(), 
            top_k=k
        )

    return hierarchical_linear_model.top_k, hierarchical_linear_model.top_k_score

