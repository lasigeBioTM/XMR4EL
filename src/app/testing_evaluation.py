import os
import time

from sklearn.decomposition import PCA

from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_hierarchical_clustering_model, load_hierarchical_linear_model, predict_labels_hierarchical_clustering_model, predict_labels_hierarchical_linear_model

def main():
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"    
    test_input_embeddings_filepath = "data/processed/vectorizer/test_input_embeddings.npy"
    
    hierarchical_clustering_model_filepath = "data/processed/clustering/test_hierarchical_clustering_model.pkl"
    hierarchical_linear_model_filepath = "data/processed/regression/test_hierarchical_linear_model.pkl"
    
    start = time.time()
    
    with open(test_input_filepath, 'r') as test_input_file:
        test_input = [line.strip() for line in test_input_file]
    
    
    if os.path.exists(test_input_embeddings_filepath):
        test_input = load_bio_bert_vectorizer(test_input_embeddings_filepath)
    else:
        test_input = create_bio_bert_vectorizer(corpus=test_input, 
                                                        output_embeddings_file=test_input_embeddings_filepath,
                                                        directory_onnx_model=onnx_directory)
    
    
    # Assuming test_input has shape (n_samples, 768) and the model was trained on 140 features
    
    pca = PCA(n_components=2)
    test_input = pca.fit_transform(test_input)  # Reduce to 140 features
    
    hierarchical_clustering_model = load_hierarchical_clustering_model(hierarchical_clustering_model_filepath)
    
    predicted_labels = predict_labels_hierarchical_clustering_model(hierarchical_clustering_model, test_input)
    
    hierarchical_linear_model = load_hierarchical_linear_model(hierarchical_linear_model_filepath)
    
    top_k = 1
    
    class_labels = hierarchical_clustering_model.labels
    
    top_k_acc = predict_labels_hierarchical_linear_model(hierarchical_linear_model, test_input, predicted_labels, top_k)
    
    # Overall Mean Top-1 Score: 0.6162762641906738
    # Overall Mean Top-3 Score: 0.8822723031044006
    # Overall Mean Top-5 Score: 0.9555599093437195
    print(f"Top-{top_k} Accuracy (sklearn): {top_k_acc}") 
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    
    
if __name__ == "__main__":
    main()