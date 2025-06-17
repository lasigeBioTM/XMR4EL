#### eXtreme Multi-Label Ranking for Entity Linking - XMR4EL

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
>>> min_leaf_size = 20
>>> depth = 3
>>> n_features = 768
>>> max_n_clusters = 16
>>> min_n_clusters = 6
>>> vectorizer_config = {"type": "tfidf","kwargs": {"max_features":20000}}
>>> transformer_config = {"type": "biobert","kwargs": {"batch_size": 400}
>>> clustering_config = {"type": "sklearnminibatchkmeans", "kwargs": { "n_clusters": 8, "init": "k-means++", "max_iter": 500, "batch_size": 0, "verbose": 0, "compute_labels": True, "random_state": 42, "tol": 1e-4, "max_no_improvement": 20, "init_size": 24, "n_init": 5, "reassignment_ratio": 0.01}}
>>> classifier_config = {"type": "sklearnlogisticregression", "kwargs": {"n_jobs": -1, "random_state": 0,"penalty":"l2",    "class_weight": "balanced", "C": 1.0, "solver":"lbfgs", "max_iter":1000}}
>>> training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
>>> labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
# No need to truncate data if we want to test all the data
>>> train_data = Preprocessor().load_data_labels_from_file(train_filepath=training_file,labels_filepath=labels_file,truncate_data=150)
>>> Y_train = train_data["labels_matrix"]
>>> raw_labels = parsed_train_data["raw_labels"]
>>> X_train = train_data["corpus"]
>>> x_cross_train = parsed_train_data["cross_corpus"]
>>> pipe = SkeletonBuilder(vectorizer_config, transformer_config, clustering_config, classifier_config, n_features, max_n_clusters, min_n_clusters, min_leaf_size, depth, dtype=np.float32)
>>> htree = pipe.execute(raw_labels, x_cross_train, X_train,Y_train)
# Omiting logging
>>> save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
>>> htree.save(save_dir)
```

To predict using the model

```bash
>>> from xmr4el.predict.predict import Predict
>>> from xmr4el.xmr.skeleton import Skeleton
>>> k = 5
>>> transformer_config = {"type": "biobert","kwargs": {"batch_size": 400}}
>>> file_test_input = "data/raw/mesh_data/bc5cdr/test_input_bc5cdr.txt"
>>> with open(file_test_input, "r") as file:
...     unique_names = set(file.read().splitlines())
...
>>> name_list = sorted(unique_names)
>>> trained_xtree = Skeleton.load("test/test_data/saved_trees/TreeDisease100_LG")
>>> predicted_labels = Predict.inference(trained_xtree, name_list, transformer_config, k=k)
>>> print(predicted_labels)
```

## Input Files / Data

To load data using files, you must provide two files:

1. A label file containing one label ID per line.
2. A test data file where each line starts with an index corresponding to the label ID and a text sample.

* Example Label File

Each line must contain one ID:

```text
C538288
C535484
C579849
C579850
C567076
C537805
C537806
```

* Example Test Data File

Each line must begin with an index, followed by a tab (\t) and the corresponding text. The index refers to the label ID in the label file:

```text
0	10p deletion syndrome (partial)
0	chromosome 10, 10p- partial
0	chromosome 10, monosomy 10p
0	chromosome 10, partial deletion (short arm)
0	monosomy 10p
1	13q deletion syndrome
1	chromosome 13q deletion
1	chromosome 13q deletion syndrome
1	chromosome 13q monosomy
1	chromosome 13q syndrome
1	deletion 13q
```

In this case, the label C538288 corresponds to all the lines starting with 0.

# Loading Files with the Preprocessor

To feed the data into the algorithm using files:

```bash
>>> training_file = os.path.join(os.getcwd(), "test/test_data/train/disease/train_Disease_100.txt")
>>> labels_file = os.path.join(os.getcwd(), "data/raw/mesh_data/medic/labels.txt")
# No need to truncate data if we want to test all the data
>>> train_data = Preprocessor().load_data_labels_from_file(
...     train_filepath=training_file,
...     labels_filepath=labels_file,
...     truncate_data=150)
```

# Loading Data Manually

You can also manually open and prepare the files before passing them to the algorithm.

* Test Data Format

The test data should be a list of concatenated strings, where all text samples with the same ID are merged into one string:

```bash
[
  "achm2 achromatopsia 2 colorblindness, total rmch2 rod monochromacy 2 ...",
  ...
]  # type: List[str]
```

* Labels Format

The labels should be a list of IDs, and can be encoded using:

```bash
>>> with open(labels_file, 'r') as f:
...     labels = [line.strip() for line in f]

>>> label_matrix, _ = Preprocessor().enconde_labels(labels)
>>> print(label_matrix, type(label_matrix))
```

# Important

Ensure that label_matrix and the test data list have the same length. If not, the program will exit.

# Truncating Data

When loading from files using Preprocessor().load_data_labels_from_file(...), the method will automatically truncate any labels that do not have corresponding test data entries.






