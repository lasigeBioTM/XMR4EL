import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.predicter.predict import PredictTopK
from src.app.utils import load_bio_bert_vectorizer, load_hierarchical_clustering_model, load_hierarchical_linear_model

"""
    Depending on the train file, different number of labels, 

    * train_Disease_100.txt -> 13240 labels, 
    * train_Disease_500.txt -> 13190 labels, 
    * train_Disease_1000.txt -> 13203 labels,  

    Labels file has,

    * labels.txt -> 13292 labels,
"""
def main():

    hierarchical_clustering_model_filepath = "data/processed/clustering/test_hierarchical_clustering_model.pkl"
    hierarchical_linear_model_filepath = "data/processed/regression/test_hierarchical_linear_model.pkl"
    
    test_input_embeddings_filepath = "data/processed/vectorizer/test_input_embeddings_gpu.npy"
    
    start = time.time()
    
    # load inputs, 
    test_input = load_bio_bert_vectorizer(test_input_embeddings_filepath)
    
    # pca = PCA(n_components=2)
    # test_input = pca.fit_transform(test_input)  # Reduce to 12 features
    test_input = normalize(test_input, norm='l2', axis=1)
    
    hierarchical_clustering_model = load_hierarchical_clustering_model(hierarchical_clustering_model_filepath)
    hierarchical_linear_model = load_hierarchical_linear_model(hierarchical_linear_model_filepath)
    
    top_k = 3
    
    root_node = hierarchical_linear_model.tree_node
    
    cluster_predictions = hierarchical_clustering_model.predict(test_input)
    
    predictions = PredictTopK.predict(cluster_predictions, root_node, test_input, k=top_k)
    
    correct_count = 0
    total = len(predictions)
    top_k_confidence_list = []

    for idx in range(total):
        cluster_pred = cluster_predictions[idx]
        predicted_labels = predictions[idx]['predicted_labels']
        top_k_confidences = predictions[idx]['top_k_confidences']
        true_label = predictions[idx]['true_label']
        
        # print(f"Cluster Predictions: {cluster_pred}")
        # print(f"True Label: {true_label}")
        # print(f"Predicted Labels: {predicted_labels}")
        # print(f"Confidence Scores: {top_k_confidences}")
        # print("-" * 50)
        
        top_k_confidence_list.append(top_k_confidences)
        
        if true_label in predicted_labels:
            correct_count += 1
    
    top_k_confidence_list = np.concatenate(top_k_confidence_list).tolist()
    top_k_confidence_list = [float(x) for x in top_k_confidence_list]
    
    print(f"\nTop-1 Confidence: {np.mean(top_k_confidence_list)}")
    print(f"Top-{top_k} Accuracy: {round(correct_count / total if total > 0 else 0, 6)}")
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    

if __name__ == "__main__":
    main()