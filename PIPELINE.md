# Pipeline reference

This describes the implemented local training/evaluation pipeline. Configuration
and command recipes are in [SCRIPTS.md](SCRIPTS.md); recorded model quality is a
separate [benchmark](BENCHMARK.md). Source and regression paths below are
evidence for behavior, not a claim that corpus experiments were rerun.

Model JSON is passed to `XModel(**config)`; component configs use `type` and
`kwargs`. The main settings follow the pipeline boundaries:

| Keys | Owner / meaning |
| --- | --- |
| `context_window`, `abbrev_expansion` | PubTator preprocessing by the calling script. |
| `features`, `transformer_config` | Feature composition and mention transformer. |
| `vectorizer_config`, `dimension_config` | Whole-text TF-IDF/SVD for `tfidf`; mention TF-IDF/SVD for the three-block encoder. |
| `context_vectorizer_config`, `context_dimension_config` | Context TF-IDF/SVD for the three-block encoder. |
| `clustering_config` | `balancedkmeans`: `n_clusters`, `iter_limit`, `refine_iter`, `seed`. |
| `depth`, `min_leaf_size`, `cut_half_cluster` | Matcher layers, minimum valid cluster size and branching reduction. |
| `routing_features` | Append parent scores to child matcher features (default true). |
| `matcher_config` | `jointlogisticregression`: `C`, `class_weight` (`balanced` or null), `tol`, `max_iter`. |
| `predict_config` | Saved inference defaults, described below. |

## Preprocessing

[Preprocessor](xmr4el/data/readers.py) loads and groups training text by label.
`XModel.train(groups, labels)` expects one text list per label entry; it flattens
those lists into training rows. Repeated labels share a target column. A group
without texts contributes no column.

| Input | Contract |
| --- | --- |
| PubTator | Each usable annotation becomes `mention [SEP] context`, a concept ID, a document ID and a span. Grouping retains first-seen label order. |
| Grouped TSV | Integer group IDs are sorted numerically; texts within each group keep input order. Nonblank lines in the label file align with those groups. Extra labels are truncated with a warning; too few labels fail. |

The training CLI's `-ds_len` slices the first N groups after loading. This is not
random sampling. It does not change the ordering of output columns, which is
determined later by fitted label classes. `train.py` supports both inputs;
`evaluate.py` and `run_experiment.py` read PubTator.

PubTator offsets index the title, followed by one space and the abstract when
present. `context_window=None` uses that document. A configured window takes up
to that many whitespace-separated words on either side, excluding the mention.
`abbrev_expansion="append"` appends an in-document long form to the mention;
`"replace"` substitutes it, and `None` disables expansion.

Reader rules:

- `-1` IDs are skipped. A composite `D1|D2` needs matching per-part texts in
  column 7; otherwise it is skipped. Usable parts become separate single-gold rows.
- An annotation missing a concept ID is warned and skipped; relation lines are ignored.
- A mention/offset text mismatch is warned and retained. Document-ID mismatches
  and out-of-document spans fail the load.
- Warnings identify file/line and the problem without reproducing mention text.

Checks: [reader tests](tests/data/test_readers.py),
[conversion tests](tests/data/test_to_pubtator.py), and reader regressions in
[test_audit_regressions.py](tests/test_audit_regressions.py).

## Feature generation

[TextEncoder._encode](xmr4el/features/encoder.py) supports two compositions:

| `features` | Output |
| --- | --- |
| `tfidf` | TF-IDF of each whole input text, optional effective SVD, then L2 normalization. Suitable for plain TSV. |
| `sapbert_char_context` | Transformer of the mention, TF-IDF/SVD of the mention, TF-IDF/SVD of context. Requires `[SEP]` in every input. Each block and the concatenated row are L2-normalized. |

The three nonzero blocks therefore have equal norm before concatenation; there
is no configured block weighting. PIFA later consumes the complete feature row,
including context. All-empty or fully filtered TF-IDF vocabularies fail fitting;
for example, zero context windows can leave the context block unfit.

Backend names are explicit: `tfidf` in [vectorizers.py](xmr4el/features/vectorizers.py)
and `sklearntruncatedsvd` in [reduction.py](xmr4el/features/reduction.py). If the
requested SVD components exceed the vocabulary width, reduction is skipped with
a warning. SVD output is dense; an unreduced TF-IDF block stays sparse. Mixed
blocks are assembled as CSR. `block_widths` stores actual fitted widths for
`XModel.feature_blocks()`.

[Transformer](xmr4el/features/transformers.py) uses frozen Hugging Face models.
Short types are `sapbert`, `biobert`, and `sentencetbiobert`;
`kwargs.model_name` can select another Hub ID or a local directory. `pooling`
(`cls` or `mean`), `max_seq_length` and `revision` customize encoding. SapBERT's
built-in defaults are CLS pooling and at most 25 tokens.

Training resolves an explicit Hub model's revision to a commit, falling back to
the cached snapshot offline; the saved encoder retains that revision. Distinct
mention strings are encoded once and restored to input row order. Embeddings
are cached under `outputs/cache/transformers/` relative to the package parent,
keyed by checkpoint, revision, pooling, length, dtype and distinct texts.
`XMR4EL_TRANSFORMER_CACHE=0` disables reads and writes. Failed cache writes retain
computed embeddings. Local checkpoint edits require clearing/disabling the cache;
the key does not hash model-directory contents.

Checks: [encoder tests](tests/features/test_text_encoder.py),
[adapters](tests/features/test_adapters.py), [SVD](tests/features/test_reduction.py),
[transformer tests](tests/features/test_transformers.py), and revision/cache
regressions in [test_audit_regressions.py](tests/test_audit_regressions.py).

## Label embeddings and hierarchy

[XModel._fit](xmr4el/xmodel.py) builds:

- `X`: training rows × features, normally float32, dense or CSR.
- `Y`: sparse one-hot training rows × labels, fitted with `MultiLabelBinarizer`.
- `Z`: dense labels × features, `normalize(Y.T @ X)` (PIFA), preserving `X`'s dtype.

The invariant is **column j of Y = row j of Z = initial_labels[j]**. Label names
follow sorted fitted classes, not incoming group order; empty groups disappear.
[test_label_mapping.py](tests/integration/test_label_mapping.py) checks this
through train/save/load using stubbed encoder and hierarchy components.

[BalancedKMeans](xmr4el/hierarchy/clusterers.py) performs seeded spherical
recursive two-means followed by joint balanced refinement. The cluster matrix
`C` maps node-local labels to children. Undersized clusters are reassigned to
valid centroids; they do not discard labels. An internal node must retain at
least two valid clusters or training fails.

`depth` is the number of matcher layers, including leaves: `1` makes the root a
leaf. Every earlier layer is split; there is no adaptive early leaf. With
`cut_half_cluster=true`, requested cluster counts halve below the root (minimum
2). Defaults are resolved before halving and the caller's config is preserved.
Checks: [clustering](tests/hierarchy/test_clusterers.py) and
[tree tests](tests/hierarchy/test_tree.py).

## Training and inference

[MLModel.train](xmr4el/hierarchy/node.py) trains internal matchers on `Y @ C`.
Leaves use identity `C`, so each leaf target is a label. Child training includes
only rows with a positive label in that child; it has no out-of-cluster negatives.
[Matcher](xmr4el/learning/matcher.py) preserves local target order and excludes
rows without a local positive.

With `routing_features=true` (code default),
[prepare_layer](xmr4el/hierarchy/tree.py) appends the selected parent cluster
probability, sum of parent probabilities and maximum parent probability, then
normalizes. Prediction applies the same augmentation. Child label embeddings
remain selected parent `Z` rows; they are not recomputed from augmented features.

[JointOvRLogistic](xmr4el/learning/classifiers.py), type
`jointlogisticregression`, fits independent L2 logistic problems with a regularized
bias using Torch L-BFGS. Predictions are per-target sigmoids. Large target
matrices are fitted in label blocks. Low-density CSR features stay sparse, while
weights remain dense float32. CUDA is preferred; MPS is used only for sufficiently
large dense problems, otherwise CPU. CUDA/MPS matcher OOM falls back to CPU;
transformer OOM instead retries smaller batches on the selected device.

[HierarchicalMLModel.predict](xmr4el/hierarchy/tree.py) processes query batches,
keeps the best `beam_size` branches per query by cumulative log path probability,
and selects up to `leaf_candidates` labels from each visited leaf. Base score:

```text
leaf sigmoid probability × product of ancestor routing probabilities
```

With `knn_beta > 0`, [XModel.predict_encoded](xmr4el/xmodel.py) multiplies each
candidate by `exp(knn_beta × max_cosine)`, using the query's mention-transformer
block against the label's training mention rows (all features for `tfidf`).
[label_max_cos](xmr4el/learning/scoring.py) scores candidate pairs only. It cannot
recover a label discarded by routing or the leaf cap. The final `topk` cut is
applied after fusion. Results are CSR rows ordered by decreasing score.

| Search key | Code default | Constraint |
| --- | --- | --- |
| `beam_size` | 10 | Integer ≥ 1. |
| `leaf_candidates` | 100 | Integer ≥ 1, per visited leaf. |
| `topk` | 0 | Integer ≥ 0; 0 keeps all selected candidates. |
| `knn_beta` | 0.0 | Number in [0, 100]; 0 disables fusion. |

These defaults are stored in each tree's `predict_config`; non-`None` prediction
arguments override them. Presets differ: `xmr4el_base_config.json` uses beam 2,
beta 0; `xmr4el_bc5cdr_config.json` uses beam 10, beta 10. Both have depth 2 and
minimum cluster size 5; their root cluster requests are 6 and 128 respectively.
Both presets request 768 SVD components for each TF-IDF block, character mention
ngrams of lengths 2–4, word-unigram context, a 10-word context window and append
abbreviation expansion. Actual widths depend on fitting/reduction.
See [configs](configs/) before choosing a preset for a small dataset.

Matcher/path scores are ranking rules, not established calibration across
leaves. Pruning uses `argpartition` without a universal label-ID tie break;
stable sorting of retained candidates cannot recover excluded tied labels.
Dense `Z`, dense weights and temporary score matrices still determine memory
needs despite sparse input, label blocking and query batching.

Checks: [matcher](tests/learning/test_matcher.py),
[logistic solver](tests/learning/test_joint_logistic.py),
[kNN](tests/learning/test_scoring.py),
[child inputs](tests/hierarchy/test_prepare_layer.py), and
[persistence](tests/integration/test_pipeline_persistence.py).

## Evaluation

[evaluate_tree](xmr4el/eval.py) reads PubTator using the tree's preprocessing.
Each loaded row has one gold label. Only rows with an in-vocabulary gold label
enter ranking metrics; report `rows / rows_total` as coverage.

| Metric | Definition on eligible rows |
| --- | --- |
| `acc@1` | Fraction with gold at rank 1. |
| `MRR` | Mean reciprocal gold rank; an absent gold contributes 0. |
| `R@k` | Fraction with gold at rank ≤ k, for k = 5, 10, 20, 50, 100. |
| `R@cand` | Fraction with gold anywhere in the returned candidates, after any `topk` cut. |
| `acc@1_all` | `acc@1 × rows / rows_total`; out-of-vocabulary loaded rows count as misses. |

Tied evaluation scores retain CSR order. `rows_total` counts reader output,
so skipped malformed/unlinked/composite annotations are outside that denominator.
No eligible rows means no ranking rates: the CLI exits unsuccessfully and the
runner retains diagnostic metrics but marks the run failed.

An evaluation `train_path` enables a stripped, lowercased mention-string dictionary
over training rows in the model vocabulary. Groups distinguish unseen strings,
strings with one label and strings with multiple labels. The dictionary selects
the most frequent label; hybrid uses it for seen strings and the tree otherwise.
Empty groups have null rates.

Metrics record `eval_protocol` (currently `EVAL_PROTOCOL = 1`),
`population_sha256` (ordered loaded row positions and gold IDs), and, when a
reference training file is supplied, `strings_sha256` (effective dictionary
groups and answers). These hashes describe different evidence. TUI comparison
decisions belong to [comparison.assess](xmr4el_tui/services/comparison.py).
Changing metric definitions requires a protocol bump. Checks:
[test_eval.py](tests/test_eval.py).

## Persistence and optional reranking

`XModel.save(parent)` creates a timestamped child directory and returns its path:

| Artifact | Purpose |
| --- | --- |
| `hml/`, `text_encoder/` | Fitted hierarchy, matchers and feature components. |
| `xmodel.pkl` | Inference state, ordered labels, search defaults and artifact `tree_id`; written last via rename. Its presence is the save-completion marker. |
| `research.pkl` | Training groups, `X`, `Y`, `Z`. |
| `xmodel.json` | Readable summary, not a replacement for fitted components. |

`XModel.load(path)` leaves research arrays unloaded. `research=True` loads them;
enabling kNN on a tree saved without an index loads them lazily. Saving with
default `knn_beta > 0` includes the index so inference need not load `X`/`Y`.
Each save assigns a new tree ID. Load checks label counts against root labels
and, when loaded, `Z`; hierarchy load sorts numeric layer/node directories and
validates parent/child label membership. These checks do not make pickle safe:
**load only trusted artifacts**. Older layouts are not migrated.

[CrossEncoderReranker](xmr4el/rerank.py) is experimental and separate from
`XModel.predict`. [train_reranker.py](scripts/train_reranker.py) uses saved-tree
top-K negatives, or document-held-out fold trees when requested; label texts
come from frequent training mention strings. Reranker metadata is bound to a
saved tree ID and loading rejects another ID. `evaluate.py -reranker_path`
reorders the existing top K with `log(tree score) + w × logit` (`inf`: logit
only); candidates below K keep their ranks. It cannot restore routing misses.
See [commands and defaults](SCRIPTS.md#optional-reranker).
