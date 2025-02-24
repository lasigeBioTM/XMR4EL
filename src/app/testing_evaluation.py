import os
import time
import numpy as np

from sklearn.decomposition import PCA

from src.predicter.predict import PredictTopK
from src.app.utils import create_bio_bert_vectorizer, load_bio_bert_vectorizer, load_hierarchical_linear_model

def main():
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    
    onnx_directory = "data/processed/vectorizer/biobert_onnx_cpu.onnx"    
    test_input_embeddings_filepath = "data/processed/vectorizer/test_input_embeddings.npy"
    
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
    
    hierarchical_linear_model = load_hierarchical_linear_model(hierarchical_linear_model_filepath)
    
    tree_node = hierarchical_linear_model.tree_node
    
    # print(tree_node.print_tree())
    
    top_k = 5
    
    # Inferncenp.array([input_embedding])
    # predictions = PredictTopK.predict(tree_node, test_input, k=top_k)
    
    # print(tree_node.print_tree())
    
    # Batch
    top_k_predictions = PredictTopK.predict_batch(tree_node, test_input, k=top_k, max_depth=2)
    ece = PredictTopK.compute_ece(top_k_predictions)
    # print(top_k_results)
    
    # print(top_k_predictions)
    
    # exit()
    
    correct_count = 0
    total = len(top_k_predictions)
    top_k_confidence_list = []

    # print(top_k_predictions)

    for pred in top_k_predictions:
        true_label = pred['true_label']
        top_k_labels = pred['top_k_labels']
        top_k_confidence = pred['top_k_confidence']
        
        print(f"True Label: {true_label}")
        print(f"Top-K Labels: {top_k_labels}")
        print(f"Confidence Scores: {top_k_confidence}")
        print("-" * 50)
        
        # top_k_confidence_list.append(top_k_confidence)
        
        if true_label in top_k_labels:
            correct_count += 1
        else:
            print("")
            #print(f"True Label: {true_label} -> Predicted Labels: {top_k_labels}")
            
    # print(f"Tree Structure:\n{tree_node.print_tree}")
    
    # 0.1352 with None
    # 0.26 with temperature scaling
    # 0.2 with label smoothing
    # 0.3 with label smoothing and temperature 
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    # print(f"\nTop-1 Confidence: {np.mean(top_k_confidence_list)}")
    print(f"\nTop-{top_k} Accuracy: {round(correct_count / total if total > 0 else 0, 6)}")
    
    # print(tree_node.print_tree())
    
    end = time.time()
    
    print(f"{end - start} secs of running")
    
if __name__ == "__main__":
    main()