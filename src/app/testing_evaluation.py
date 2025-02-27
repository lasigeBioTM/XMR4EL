import os
import time
import unicodedata
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.predicter.predict import PredictTopK
from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_hierarchical_clustering_model, load_hierarchical_linear_model

def main():
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"    
    test_input_embeddings_filepath = "data/processed/vectorizer/test_input_embeddings_gpu.npy"
    
    hierarchical_clustering_model_filepath = "data/processed/clustering/test_hierarchical_clustering_model.pkl"
    hierarchical_linear_model_filepath = "data/processed/regression/test_hierarchical_linear_model.pkl"
    
    start = time.time()
    
    def normalize_text(text):
        return unicodedata.normalize("NFKC", text).strip()
    
    with open(test_input_filepath, 'r') as test_input_file:
        test_input = [normalize_text(line) for line in test_input_file]
    
    
    if os.path.exists(test_input_embeddings_filepath):
        test_input = load_bio_bert_vectorizer(test_input_embeddings_filepath)
    else:
        test_input = create_bio_bert_vectorizer(corpus=test_input, 
                                                            output_embeddings_file=test_input_embeddings_filepath,
                                                            directory_onnx_model=onnx_directory)
    
    
    # Assuming test_input has shape (n_samples, 768) and the model was trained on 140 features
    
    pca = PCA(n_components=2)
    test_input = pca.fit_transform(test_input)  # Reduce to 12 features
    test_input = normalize(test_input, norm='l2', axis=1)
    
    hierarchical_clustering_model = load_hierarchical_clustering_model(hierarchical_clustering_model_filepath)
    hierarchical_linear_model = load_hierarchical_linear_model(hierarchical_linear_model_filepath)
    
    # Top-1 -> 0.201057
    # Top-2 -> 0.411075
    # Top-3 -> 0.696691
    
    # top-1 -> 0.54, now
    top_k = 1
    
    # Infernce np.array([input_embedding])
    # predictions = PredictTopK.predict(tree_node, test_input, k=top_k)
    
    # print(tree_node.print_tree())
    
    # normalize the data to predict
    
    """
    Without normalizing:
        Top-1 Confidence: 0.9714009761810303
        Top-1 Accuracy: 0.383691
    Normalizing:
        Top-1 Confidence: 0.18224984407424927
        Top-1 Accuracy: 0.54
    """
    
    # Se normalizar     
    root_node = hierarchical_linear_model.tree_node
    
    # print(root_node)
    
    cluster_predictions = hierarchical_clustering_model.predict(test_input)
    
    # Batch
    predictions = PredictTopK.predict(cluster_predictions, root_node, test_input, k=top_k)
    # ece = PredictTopK.compute_ece(top_k_predictions)
    # print(top_k_results)
    
    # print(top_k_predictions)
    
    print(predictions)
    
    correct_count = 0
    total = len(predictions)
    top_k_confidence_list = []

    for idx in range(total):
        cluster_pred = cluster_predictions[idx]
        predicted_labels = predictions[idx]['predicted_labels']
        top_k_confidences = predictions[idx]['top_k_confidences']
        true_label = predictions[idx]['true_label']
        
        print(f"Cluster Predictions: {cluster_pred}")
        print(f"True Label: {true_label}")
        print(f"Predicted Labels: {predicted_labels}")
        print(f"Confidence Scores: {top_k_confidences}")
        print("-" * 50)
        
        top_k_confidence_list.append(top_k_confidences)
        
        if true_label in predicted_labels:
            correct_count += 1
                
    # print(f"Tree Structure:\n{tree_node.print_tree}")
    
    # 0.1352 with None
    # 0.26 with temperature scaling
    # 0.2 with label smoothing
    # 0.3 with label smoothing and temperature 
    # print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    top_k_confidence_list = np.concatenate(top_k_confidence_list).tolist()
    top_k_confidence_list = [float(x) for x in top_k_confidence_list]
    
    print(f"\nTop-1 Confidence: {np.mean(top_k_confidence_list)}")
    print(f"\nTop-{top_k} Accuracy: {round(correct_count / total if total > 0 else 0, 6)}")
    
    # print(tree_node.print_tree())
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    
if __name__ == "__main__":
    main()