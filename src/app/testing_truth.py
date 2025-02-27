from src.app.utils import load_bio_bert_vectorizer


def main():
    
    test_input_filepath = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
    test_input_embeddings_filepath = "data/processed/vectorizer/test_input_embeddings_gpu.npy"
    
    test_input = load_bio_bert_vectorizer(test_input_embeddings_filepath)
    
    print(test_input[-3][0])
    print(test_input[-4][0])
    

if __name__ == "__main__":
    main()