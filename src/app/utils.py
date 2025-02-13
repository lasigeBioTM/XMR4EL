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

def create_bio_bert_vectorizer(corpus, output_embeddings_file, directory_onnx_model=None):
    print("Running BioBert")
    
    output_prefix = output_embeddings_file.split('.')[0]  
    
    if GPU_AVAILABLE:
        print("Graphics Processing")
        embeddings = BioBertVectorizer.predict_gpu(corpus)
        output_file = f"{output_prefix}_gpu.npy"
        Preprocessor.save_biobert_labels(embeddings, output_file)
    else:  
        
        if directory_onnx_model == None:
            print("Not choosen an directory to save onnx model")
            exit()
        
        embeddings = BioBertVectorizer.predict_cpu(corpus=corpus, 
                                                   directory=directory_onnx_model, 
                                                   output_prefix=output_prefix)
        output_file = f"{output_prefix}_cpu.npy"
        Preprocessor.save_biobert_labels(embeddings, output_file)
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

def create_hierarchical_clustering(X_train_feat, save_directory):
    if GPU_AVAILABLE:
        print("Processing Hierarchical Clustering Algorithm with GPU SUPPORT")
        
        from src.machine_learning.gpu.ml import KMeansGPU
        
        divisive_hierarchical_clustering = DivisiveHierarchicalClustering.fit(
            X_train_feat, 
            clustering_model_factory=KMeansGPU.create_model(),
            gpu_usage=True
        )
        
        divisive_hierarchical_clustering.save(save_directory)
        
    else:
        print("Processing Hierarchical Clustering Algorithm with CPU SUPPORT")
        
        from src.machine_learning.cpu.ml import KMeansCPU
        
        divisive_hierarchical_clustering = DivisiveHierarchicalClustering.fit(
            X_train_feat, 
            clustering_model_factory=KMeansCPU.create_model(),
            gpu_usage=False
        )
        
        divisive_hierarchical_clustering.save(save_directory)
        
    return divisive_hierarchical_clustering.labels

def create_hierarchical_linear_model(X_train_feat, Y_train_feat, k, save_directory):
    
    if GPU_AVAILABLE:
        
        from src.machine_learning.gpu.ml import KMeansGPU, LogisticRegressionGPU
        
        print("Processing Hierarchical Linear Model with GPU SUPPORT")
        hierarchical_linear_model = HierarchicalLinearModel.fit(
            X_train_feat, 
            Y_train_feat, 
            top_k_threshold=0.15,
            linear_model_factory=LogisticRegressionGPU.create_model(), 
            clustering_model_factory=KMeansGPU.create_model(), 
            top_k=k,
            gpu_usage=True
        )
        
        hierarchical_linear_model.save(save_directory)
        
    else:
        
        from src.machine_learning.cpu.ml import KMeansCPU, LogisticRegressionCPU
        
        print("Processing Hierarchical Linear Model with CPU SUPPORT")
        hierarchical_linear_model = HierarchicalLinearModel.fit(
            X_train_feat, 
            Y_train_feat, 
            LINEAR_MODEL=LogisticRegressionCPU.create_model(), 
            CLUSTERING_MODEL=KMeansCPU.create_model(), 
            top_k=k,
            gpu_usage=False
        )
        
        hierarchical_linear_model.save(save_directory)

    return hierarchical_linear_model.top_k, hierarchical_linear_model.top_k_score

def load_hierarchical_linear_model(directory):
    return HierarchicalLinearModel.load(directory)

def load_hierarchical_clustering_model(directory):
    return DivisiveHierarchicalClustering.load(directory)

def predict_labels_hierarchical_clustering_model(hierarchical_clustering_model, test_input):
    
    if GPU_AVAILABLE:
        print("Predicting Labels with GPU SUPPORT")
        
        from src.machine_learning.gpu.ml import KMeansGPU
        
        predict_labels = hierarchical_clustering_model.predict(KMeansGPU.create_model(), test_input)
    
    else:
        print("Predict Labels with CPU SUPPORT")
        
        from src.machine_learning.cpu.ml import KMeansCPU
        
        predict_labels = hierarchical_clustering_model.predict(KMeansCPU.create_model(), test_input)
        
    return predict_labels

def predict_labels_hierarchical_linear_model(hierarchical_linear_model, embeddings, predicted_labels, k):
    
    return hierarchical_linear_model.predict(embeddings, predicted_labels, k)

