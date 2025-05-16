## eXtreme Multi-Label Ranking for Entity Linking - XMR4EL

This project extends the PECOS pipeline, aiming to support flexible integration of custom models for Extreme Multi-Label Ranking (XMR).

By leveraging a Hierarchical Tree structure, users can experiment with various vectorizers, transformers, clustering algorithms, and linear models to evaluate and improve model performance.

Although this project is primarily designed for Entity Linking, it is general enough to support any dataset that benefits from an XMR-based approach.

Note: Paper is in progress.
Status: Functional but not yet finalized.

### Requirements and Installation

* Python (3.12)

All the packages used are displayed in the requirements.txt.

## Install As An Package

```bash
pip install git+https://github.com/lasigeBioTM/XMR4EL.git
```

### CUDA Version

* CUDA (11.4 - 11.8)

Current GPU models are based on RAPIDS.ai and are tested within the provided Docker environment (see Dockerfile).

### Quick Tour

## Models that are working and vetted

* Vectorizers

    tfidf: Scikit-learn's TF-IDF Vectorizer

* Transformers

    biobert: BioBERT Transformer

* Clustering Models

    sklearnminibatchkmeans: Scikit-learn's MiniBatchKMeans

* Linear Models

    sklearnlogisticregression: Scikit-learn's Logistic Regression

## Model Configuration

All models must be defined before running the pipeline.

A model is specified using its type, and any optional kwargs (hyperparameters) the model supports.

If kwargs is omitted or partially specified, default parameters will be used.

```bash
    # Example Configuration
    vectorizer_config = {
        {"type": "tfidf", 
        "kwargs": {"ngram_range": [1, 2], 
                    "max_features": 500, 
                    "min_df": 0.0, 
                    "max_df": 0.98, 
                    "binary": false, 
                    "use_idf": true, 
                    "smooth_idf": true, 
                    "sublinear_tf": false, 
                    "norm": "l2", 
                    "analyzer": "word", 
                    "stop_words": null}}
    }
```

1. *Training Pipeline* 

```bash
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.xmr_pipeline import XMRPipeline

train_data = Preprocessor().load_data_labels_from_file(
    train_filepath=training_file,
    labels_filepath=labels_file,
    truncate_data=16
    )

htree = XMRPipeline.execute_pipeline(
    X_train,
    Y_train,
    label_enconder, # New
    vectorizer_config,
    transformer_config,
    clustering_config,
    classifier_config,
    n_features=n_features,
    max_n_clusters=16,
    min_n_clusters=2,
    min_leaf_size=min_leaf_size,
    depth=depth,
    dtype=np.float32,
    )

save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")
htree.save(save_dir)
```

The 'execute_pipeline' method compromises two stages:

1. Hierarchical Tree Construction - builds a tree of label clusters

2. Linear Classification - applies linear models to map instances embeddings to cluster labels

2. *Evaluate Pipeline*

```bash
from xmr4el.predict.predict import XMRPredict
from xmr4el.xmr.xmr_tree import XMRTree

trained_xtree = XMRTree.load(
    "data/saved_trees/XMRTree_2025-05-08_15-16-22"
)

predicted_labels = XMRPredict.inference(
    trained_xtree, name_list, transformer_config, k=k
)
```

Once trained, the model can be used to perform inference and predict labels for new instances.

pip install --extra-index-url=https://pypi.nvidia.com cuml-cu11==25.2.1