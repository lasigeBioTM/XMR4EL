# XMR4EL – eXtreme Multi-Label Ranking for Entity Linking

XMR4EL is a research-friendly framework that extends the [PECOS](https://github.com/amzn/pecos) pipeline for extreme multi-label ranking (XMR). It is designed to help you build, evaluate, and iterate on entity linking systems that must choose from very large label spaces. Although our primary focus is biomedical entity linking, every component is modular so you can plug in alternative algorithms or apply the framework to any XMR-ready dataset.

## Table of contents
1. [Key ideas](#key-ideas)
2. [Installation](#installation)
3. [Repository layout](#repository-layout)
4. [How the pipeline works](#how-the-pipeline-works)
5. [Training a model](#training-a-model)
6. [Running inference](#running-inference)
7. [Saving and loading models](#saving-and-loading-models)
8. [Customising components](#customising-components)
9. [Input data expectations](#input-data-expectations)
10. [Working with Docker](#working-with-docker)
11. [Status](#status)

## Key ideas

- **Hierarchical modelling** – Large label sets are handled by recursively clustering labels into a tree. At prediction time the tree is traversed top-down to focus computation on the most promising label subsets (`xmr4el/xmr/base.py`).

- **Flexible featurisation** – Text is converted to dense and sparse features through a configurable pipeline composed of vectorisers, transformers, and dimensionality reducers (`xmr4el/featurization`).

- **Label-aware ranking** – The framework creates label embeddings (PIFA) and trains ranking models that fuse matcher and ranker scores for better retrieval quality (`xmr4el/models`).

- **Swappable components** – Every stage (clustering, matcher, ranker) is configured through lightweight wrappers so you can experiment without editing the core training loop.

## Installation

> **Python**: 3.12

Install the package directly from GitHub:

## Install As An Package

```bash
pip install git+https://github.com/lasigeBioTM/XMR4EL.git
```

All runtime dependencies are listed in [`requirements.txt`](requirements.txt).

### GPU support

The repository contains CUDA-ready configurations (11.4 – 11.8). RAPIDS-backed GPU models have been validated with the provided Docker image.

## Repository layout

xmr4el/
├── featurization/        # Text encoders, preprocessing utilities, label embeddings
├── clustering/           # Clustering wrappers used to build the hierarchical tree
├── matcher/              # Candidate generation models
├── ranker/               # Ranking algorithms for leaf scoring
├── models/               # Component factories and configuration helpers
└── xmr/                  # Hierarchical model, training loop, persistence utilities

Test fixtures live under [`test/test_data`](test/test_data) and are useful when experimenting with the API.

## How the pipeline works

1. **Preprocessing** – Raw training files are grouped by label, producing lists of synonyms per concept. See `Preprocessor.load_data_labels_from_file` for the exact behaviour (`xmr4el/featurization/preprocessor.py`).

2. **Text encoding** – `TextEncoder` builds sparse (e.g., TF–IDF) and dense (e.g., BioBERT) representations according to the chosen configuration (`xmr4el/featurization/text_encoder.py`). Optional dimensionality reduction can be applied before training.

3. **Label embeddings** – `LabelEmbeddingFactory` converts grouped texts into a binary label matrix and produces PIFA label embeddings (`xmr4el/featurization/label_embedding_factory.py`).

4. **Hierarchical tree building** – `HierarchicaMLModel` recursively clusters labels, trains matchers to route queries, and fits ranking models (`xmr4el/xmr/base.py`).

5. **Prediction** – Queries are encoded with the trained `TextEncoder`, routed through the tree, and scored. Fusion strategies combine matcher and ranker outputs to produce the final top-*k* predictions.

The orchestration class [`XModel`](xmr4el/xmr/model.py) glues these stages together so you can train and evaluate an end-to-end system with a handful of method calls.

## Training a model

```python
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.model import XModel

# 1. Load synonym groups and labels from disk
paths = {
    "train": "data/raw/mesh_data/bc5cdr/train_bc5cdr.txt",
    "labels": "data/raw/mesh_data/medic/labels.txt",
}
dataset = Preprocessor.load_data_labels_from_file(paths["train"], paths["labels"])
X_text, Y_labels = dataset["corpus"], dataset["labels"]

# 2. Describe the components you want to use
model = XModel(
    vectorizer_config={"type": "tfidf"},
    transformer_config={"type": "sentencetbiobert", "kwargs": {"batch_size": 400}},
    clustering_config={"type": "sklearnminibatchkmeans", "kwargs": {"n_clusters": 256}},
    matcher_config={"type": "linear_l2"},
    ranker_config={"type": "sklearnlogisticregression"},
    min_leaf_size=20,
    ranker_every_layer=True,
    n_workers=8,
)

# 3. Train
model.train(X_text, Y_labels)
```

During training the model:
- Persists the raw texts and labels temporarily so they can be restored after fitting.
- Encodes the corpus, builds label embeddings, and prepares sparse training matrices.
- Constructs a hierarchical tree of classifiers and ranking models.

## Running inference

```python
queries = [
    "chromosome 10p deletion",
    "13q deletion syndrome",
]

# Request the top 5 labels per query
scores = model.predict(queries, topk=5)

# `scores` is a scipy CSR matrix with label scores per query
# Additional metadata (paths, fused scores) is returned when requesting global mode
```

You can adjust the traversal strategy with parameters such as `beam_size`, `fusion` (geometric vs. arithmetic), `topk_mode`, and `topk_inside_global` for exhaustive scoring.

## Saving and loading models

```python
model.save("artifacts/")
restored = XModel.load("artifacts/xmodel_2024-05-01_12-00-00")
```

The saved directory contains:
- Serialized hierarchical models (tree structure + trained components).
- Vectoriser/transformer artefacts used by the `TextEncoder`.
- Cached training metadata (label mappings, embeddings, configuration).

## Customising components

The following implementations are known to work well and are enabled out of the box (`README.md` in each submodule lists additional options):

- **Vectorisers** – `tfidf`
- **Transformers** – `biobert`, `sentencetbiobert`
- **Clustering** – `sklearnminibatchkmeans`, `balancedkmeans`
- **Rankers** – `sklearnlogisticregression`, `sklearnrandomforestclassifier`, `sgdclassifier`

To introduce a new algorithm, add a wrapper under `xmr4el/models` that exposes a `fit`/`predict` compatible interface, then reference it in the configuration dictionaries used when instantiating `XModel`.

## Input data expectations

1. **Label file** – One label identifier per line.
2. **Training file** – Tab-separated values where each row begins with the index of the corresponding label and is followed by a text mention.

Example label file:

```text
C538288
C535484
C579849
````

Example training file:

```text
0\t10p deletion syndrome (partial)
0\tchromosome 10, 10p- partial
1\t13q deletion syndrome
```

When loading with `Preprocessor.load_data_labels_from_file` the method:
- Groups mentions by the numeric label index.
- Aligns the resulting groups with label IDs.
- Optionally truncates the dataset for quick experiments.

Ensure the number of labels matches the number of grouped entries; otherwise an exception is raised.

## Working with Docker

Use the included [`dockerfile`](dockerfile) to spin up a CUDA-enabled development container.

```bash
docker build -t xmr4el .
docker run -v .:/app --name xmr4el -it xmr4el bash
```

Inside the container you may want to activate a virtual environment and set the `PYTHONPATH`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Add the export command to `.venv/bin/activate` if you prefer a persistent configuration.

## Status

- Paper: in progress.
- Implementation: functional but still evolving. Contributions and experimental feedback are welcome – see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.
