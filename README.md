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

    - tfidf: Scikit-learn's TF-IDF Vectorizer

* Transformers

    - biobert: BioBERT Transformer

* Clustering Models

    - sklearnminibatchkmeans: Scikit-learn's MiniBatchKMeans

* Linear Models

    - sklearnlogisticregression: Scikit-learn's Logistic Regression
    
    - sklearnrandomforestclassifier: Scikit-learn's Random Forest Classifier

## Basic Configuration

To train the model

```bash
>>> import os
>>> import numpy as np
>>> from xmr4el.featurization.preprocessor import Preprocessor
>>> from xmr4el.xmr.skeleton_builder import SkeletonBuilder
>>> onnx_directory = "test/test_data/onnx_dir/model.onnx"
>>> min_leaf_size = 10
>>> depth = 3
>>> n_features = 100
>>> max_n_clusters = 16
>>> min_n_clusters = 6
>>> vectorizer_config = {"type": "tfidf","kwargs": {"max_features":n_features}}
>>> transformer_config = {"type": "biobert","kwargs": {"batch_size": 400, "onnx_directory": onnx_directory}}
>>> clustering_config = {"type": "sklearnminibatchkmeans","kwargs": {"random_state": 0,"max_iter": 300}}
>>> classifier_config = {"type": "sklearnlogisticregression","kwargs": {"n_jobs": -1,"random_state": 0,"penalty":"l2","C": 1.0,"solver":"lbfgs","max_iter":1000}}
>>> training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
>>> labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
# No need to truncate data if we want to test all the data
>>> train_data = Preprocessor().load_data_labels_from_file(train_filepath=training_file,labels_filepath=labels_file,truncate_data=150)
>>> Y_train = train_data["labels_matrix"]
>>> X_train = train_data["corpus"]
>>> pipe = SkeletonBuilder(vectorizer_config,transformer_config,clustering_config,classifier_config,n_features,max_n_clusters,min_n_clusters,min_leaf_size,depth,dtype=np.float32)
>>> htree = pipe.execute(X_train,Y_train)
# Omiting logging
>>> save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
>>> htree.save(save_dir)
```