from src.app.utils import load_hierarchical_clustering, load_vectorizer


def main():

    test_filepath = "data/mesh_data/bc5cdr/test_Disease.txt"

    clustering_path = "data/processed/clustering.pkl"
    vectorizer_path = "data/processed/vectorizer.pkl"

    vectorizer_model = load_vectorizer(vectorizer_path)
    clustering_model = load_hierarchical_clustering(clustering_path)

    # Impossibel to do without Pedro Modules



if __name__ == "__main__":
    main()