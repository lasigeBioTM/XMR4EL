import numpy as np

from src.models.cluster_wrapper.clustering_model import ClusteringModel
from src.models.linear_wrapper.linear_model import LinearModel

test_input_path = "data/processed/vectorizer/test_input_embeddings.npy"

test_input = np.load(test_input_path, allow_pickle=True)
cluster_model = ClusteringModel.load("test_cl").model
linear_model = LinearModel.load("test_linear").model

top_k = linear_model.config.top_k
print(f"Top-k is {top_k}")

cluster_predictions = cluster_model.predict(test_input)
final_predictions = linear_model.predict(cluster_predictions, test_input)

correct_count = 0
total = len(final_predictions)
top_k_confidence_list = []

for idx in range(total):
    cluster_pred = cluster_predictions[idx]
    predicted_labels = final_predictions[idx]['predicted_labels']
    top_k_confidences = final_predictions[idx]['top_k_confidences']
    true_label = final_predictions[idx]['true_label']
        
    print(f"Cluster Predictions: {cluster_pred}")
    print(f"True Label: {true_label}")
    print(f"Predicted Labels: {predicted_labels}")
    print(f"Confidence Scores: {top_k_confidences}")
    print("-" * 50)
        
    top_k_confidence_list.append(top_k_confidences)
        
    if true_label in predicted_labels:
        correct_count += 1
    
top_k_confidence_list = np.concatenate(top_k_confidence_list).tolist()
top_k_confidence_list = [float(x) for x in top_k_confidence_list]
    
print(f"\nTop-1 Confidence: {np.mean(top_k_confidence_list)}")
print(f"Top-{top_k} Accuracy: {round(correct_count / total if total > 0 else 0, 6)}")
    