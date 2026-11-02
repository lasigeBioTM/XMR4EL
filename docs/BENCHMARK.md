# Benchmark

Recorded 2026-11-02 validation of XMR4EL on NCBI-disease and BC5CDR-chemical,
against SapBERT 1-NN and PECOS XR-Linear. XMR4EL leads SapBERT 1-NN on the
disease set and trails it on the chemical set. These are local mention-to-ID
results over the training vocabulary, not full-ontology published scores.

## Task and populations

The loader produces one gold concept ID per retained mention row. Training is
label-centric: mentions and dictionary names provide alternative texts for each
label. The recorded training inputs are `train_plus_ctd.pubtator`, described as
train split + CTD MEDIC/OMIM for diseases and train split + CTD chemicals for
chemicals. Dictionary assembly and corpus contents were not independently
audited during this documentation check.

Counts below come from saved metrics and export logs. “Total” means rows loaded
by the PubTator reader, before filtering to the training vocabulary; it does not
include annotations already skipped by the reader. String groups include only
the in-vocabulary rows.

| Set | Training labels / rows | Dev rows, kept / total | Test rows, kept / total | Dev string groups, seen 1 / seen >1 / unseen | Test string groups |
| --- | --- | --- | --- | --- | --- |
| NCBI-disease | 13,385 / 95,831 | 729 / 757 | 923 / 945 | 537 / 17 / 175 | 633 / 33 / 257 |
| BC5CDR-chemical | 179,709 / 474,854 | 5,181 / 5,307 | 5,167 / 5,352 | 4,465 / 68 / 648 | 4,330 / 111 / 726 |

Definitions follow [xmr4el/eval.py](../xmr4el/eval.py), evaluation protocol **1**:

- **acc@1:** correct first label / in-vocabulary rows.
- **cov × acc:** correct first label / all loaded rows, counting
  out-of-vocabulary gold labels as misses (`acc@1_all` in JSON).
- **MRR / R@10:** mean reciprocal gold rank / fraction with gold in the first
  ten labels, over in-vocabulary rows. Missing gold contributes zero.
- **R@cand:** fraction with gold anywhere in the returned candidate scores.
  These runs have `topk=0`; the 100-candidate limit applies per visited leaf.
- **seen 1 / seen >1 / unseen:** the mention key occurs in training with one
  distinct label / several labels / no labels. The key is text before `[SEP]`,
  stripped and lowercased, after the configured abbreviation expansion.
- **hybrid:** use the training mention key's majority label when seen;
  otherwise use the system's first label. It uses the in-vocabulary denominator,
  unlike the separate `hybrid_all` metric. It is not the oracle “either” score.

XMR4EL uses stable score ordering for ties; the 1-NN baseline explicitly breaks
equal label scores by lower label index. PECOS uses the shared scorer on its
saved CSR predictions. These are the implemented rules, not a guarantee that
all systems order equal scores identically.

## Systems

The same [configuration](../configs/xmr4el_bc5cdr_config.json) was recorded for
both trees; its SHA-256 is
`8d1052315900b7f8d2841a5b0b022c4007b13901f5548b85ff429462aea26bcd`.

- **XMR4EL:** SapBERT mention embeddings
  (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, CLS pooling), character
  2–4-gram TF-IDF reduced to 768 dimensions, and word-unigram TF-IDF of a
  10-token context window reduced to 768 dimensions. Each block is normalized
  before concatenation; logs record 2,304 features. Abbreviations use `append`.
  PIFA label embeddings feed balanced k-means with 128 clusters, depth 2, and
  joint logistic-regression matchers with balanced class weights. Prediction
  multiplies leaf matcher and routing probabilities, then applies mention-kNN
  re-scoring with beta 10; beam 10, 100 candidates per leaf, no final top-k cap.
  See [PIPELINE.md](PIPELINE.md) for the implemented stages.
- **SapBERT 1-NN:** [nn_baselines.py](../scripts/baselines/nn_baselines.py) scores
  each label by maximum cosine to its training mention embeddings, using the
  saved tree's training rows, vocabulary and transformer configuration.
- **char TF-IDF 1-NN:** the same label scoring using unreduced character TF-IDF
  fitted on training mentions. It does not use the tree's SVD/context blocks.
  This is a string-similarity reference alongside the SapBERT baseline.
- **PECOS XR-Linear:** [pecos_compare.py](../scripts/baselines/pecos_compare.py)
  exports XMR4EL's fitted training and evaluation features and aligned labels.
  [pecos_run.py](../scripts/baselines/pecos_run.py) uses libpecos 1.2.8, its own
  PIFA/hierarchical-k-means hierarchy, `-nr_splits 16 -max_leaf_size 100
  -threshold 0`, beam 10, top 100. Logs confirm solver `L2R_L2LOSS_SVC_DUAL`,
  postprocessor `l3-hinge`, tolerance 0.1. The model is fitted to `X_trn`/`Y_trn`
  in the directory named `_dev`, not to dev labels, and reused for test.
  It has no mention-kNN re-scoring. Shared input features do not make its
  hierarchy, loss, scoring or candidate budget equivalent to XMR4EL's.

The project's recorded tuning account is that features/joint matchers were
developed on MedMentions dev subsets, and depth/clusters/kNN on BC5CDR-disease
dev; several candidates were also scored on BC5CDR-disease test. NCBI-disease
and BC5CDR-chemical are reported as not used for tuning. The bundle verifies
the common configuration, not this tuning history or independent
preregistration. Each new tree has one regular test evaluation in this bundle,
plus the fixed beta-zero ablation. These test files were also evaluated on
earlier trees; they are not previously untouched test sets.

## Recorded run and evidence

[run_validation.sh](../scripts/run_validation.sh) ran at commit
`aeca7a449ce6e5730ff729e975c4326fcfc3be43`. All six tracked run records report
clean code and completed status. The environment records a Tesla T4 on one
Linux x86_64 server, CUDA execution, Python 3.12.16, Torch 2.14.1+cu126, and
containers restricted to CPU cores 0–15, 16 threads, 40g RAM with no swap.
`env.txt` also records the two Docker image IDs.

| Set | Run directory under `outputs/runs/` | Saved tree basename |
| --- | --- | --- |
| NCBI-disease | `2026-11-02_14-07-20_val_ncbi_disease` | `xmodel_2026-11-02_14-09-29` |
| BC5CDR-chemical | `2026-11-02_14-14-41_val_bc5cdr_chemical` | `xmodel_2026-11-02_14-34-34` |

The base directory contains training + dev evaluation; `_test` and `_knn0`
suffixes identify eval-only runs on the same tree. Both base records say
`trained: true` and their logs show training, not a loaded pre-existing tree.

This documentation check read the following local evidence without extracting
or modifying the archives or running models:

| Evidence | Available contents and verification |
| --- | --- |
| `outputs/server/validation.tar.gz` | 95 files: six run records, eight metric JSONs, four 1-NN JSON reports, environment, RSS table, specs, and logs for training, exports, PECOS prediction/scoring. All table values below match these records after rounding. |
| Bundle `outputs/runs/<run>/` | `git.json`, `provenance.json`, `model_config.json`, `spec.json`, `tree.json`, status, resources and metrics. Both copied configs are byte-identical to the current preset. The seven recorded config/input hashes match the pins in the validation script. |
| Bundle `outputs/validation/` | `<set>_<split>_nn.json` supplies full-precision baseline values; `*_score.log` supplies PECOS overall metrics to four decimals and string-group accuracy to three. `*_export.log` confirms label, feature and row counts. |
| `outputs/server/validation_run.log` | Records successful input hash checks, environment, both training cache hits, run completion and the final bundle path. |
| `outputs/server/runs_2026-10-20/`, `outputs/server/nn.tar.gz` | Earlier metrics/config/input records and baseline reports for the retrain qualification below; only these two datasets were inspected. |

Evidence limits:

- These outputs are gitignored local artifacts; a source checkout alone does
  not contain the numerical evidence. Paths inside the archive retain their
  original `outputs/validation/` and `outputs/runs/` prefixes.
- The bundle does not contain corpora, saved trees, fitted transformer revision
  metadata, exported feature matrices, PECOS models or prediction matrices.
  The referenced validation trees and `outputs/pecos/validation/` are absent
  locally. Individual predictions and export alignment cannot be independently
  recomputed from this bundle. PECOS has score logs, not full-precision JSON.
- Input hashes were cross-checked as recorded metadata, not recomputed from
  local datasets during this session. Hash equality does not establish split
  independence or rule out dictionary leakage.
- Baseline reports and PECOS logs match vocabulary, row and string-group counts
  and the source-defined filtering/export contract. They lack XMR4EL's protocol,
  population and string-evidence hashes; counts alone cannot prove row identity.
  See [TUI comparison rules](TUI.md#comparison-contract) for the stricter gates
  used by tracked-run comparisons.

Numbers are rounded to three decimals below. Derived correct-count differences
use `round(raw acc@1 × n)` for each system before subtraction; they are net
correct counts, not counts of prediction disagreements. PECOS cov × acc is
derived from its four-decimal logged accuracy and kept/total counts.

## XMR4EL versus SapBERT 1-NN

The `n` triples are seen 1 / seen >1 / unseen in-vocabulary mentions.

| Set, split | System | acc@1 | cov × acc | MRR | R@10 | seen 1 | seen >1 | unseen | hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NCBI-disease dev (n 537 / 17 / 175) | XMR4EL | **0.926** | 0.892 | 0.947 | 0.974 | 0.968 | 0.941 | 0.794 | 0.930 |
| | SapBERT 1-NN | 0.894 | 0.861 | 0.932 | 0.984 | 0.972 | 0.176 | 0.726 | 0.914 |
| | char TF-IDF 1-NN | 0.868 | 0.836 | 0.911 | 0.973 | 0.972 | 0.176 | 0.617 | 0.888 |
| NCBI-disease test (n 633 / 33 / 257) | XMR4EL | **0.895** | 0.874 | 0.922 | 0.964 | 0.978 | 0.818 | 0.700 | 0.898 |
| | SapBERT 1-NN | 0.874 | 0.854 | 0.910 | 0.966 | 0.976 | 0.394 | 0.685 | 0.894 |
| | char TF-IDF 1-NN | 0.826 | 0.806 | 0.874 | 0.946 | 0.976 | 0.394 | 0.510 | 0.845 |
| BC5CDR-chemical dev (n 4465 / 68 / 648) | XMR4EL | 0.946 | 0.923 | 0.954 | 0.966 | 0.977 | 0.794 | 0.744 | 0.964 |
| | SapBERT 1-NN | **0.968** | 0.945 | 0.975 | 0.986 | 0.996 | 0.559 | 0.821 | 0.974 |
| | char TF-IDF 1-NN | 0.961 | 0.939 | 0.970 | 0.983 | 0.996 | 0.559 | 0.765 | 0.967 |
| BC5CDR-chemical test (n 4330 / 111 / 726) | XMR4EL | 0.933 | 0.901 | 0.943 | 0.960 | 0.963 | 0.811 | 0.774 | 0.957 |
| | SapBERT 1-NN | **0.952** | 0.919 | 0.963 | 0.983 | 0.989 | 0.649 | 0.774 | 0.957 |
| | char TF-IDF 1-NN | 0.961 | 0.928 | 0.969 | 0.983 | 0.989 | 0.649 | 0.837 | 0.966 |

Derived XMR4EL minus SapBERT correct counts, dev / test:

- **NCBI-disease: +23 / +19.** Seen 1: −2 / +1; seen >1: +13 / +14;
  unseen: +12 / +4.
- **BC5CDR-chemical: −117 / −98.** Seen 1: −83 / −116; seen >1: +16 / +18;
  unseen: −50 / 0.

Thus the disease gains are concentrated in ambiguous and unseen strings; the
chemical losses mostly involve strings seen with one training label. XMR4EL
R@cand is 0.9863 / 0.9870 on NCBI dev/test and 0.9718 / 0.9659 on chemical
dev/test: some gold labels cannot be recovered by reordering its candidates.
These observations do not isolate a cause such as context quality or label count.

## XMR4EL versus PECOS XR-Linear

This compares different models on the exported features, not published PECOS
pipelines. XMR4EL returns 1,000 candidates/query in these runs; PECOS returns
100. PECOS MRR and recall are scored within those returned predictions.

| Set, split | System | acc@1 | cov × acc | MRR | R@10 | seen 1 | seen >1 | unseen | hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NCBI-disease dev | XMR4EL | **0.926** | 0.892 | 0.947 | 0.974 | 0.968 | 0.941 | 0.794 | 0.930 |
| | PECOS XR-Linear | 0.894 | 0.861* | 0.916 | 0.947 | 0.946 | 0.882 | 0.737 | 0.916 |
| NCBI-disease test | XMR4EL | **0.895** | 0.874 | 0.922 | 0.964 | 0.978 | 0.818 | 0.700 | 0.898 |
| | PECOS XR-Linear | 0.828 | 0.808* | 0.875 | 0.939 | 0.913 | 0.879 | 0.611 | 0.873 |
| BC5CDR-chemical dev | XMR4EL | **0.946** | 0.923 | 0.954 | 0.966 | 0.977 | 0.794 | 0.744 | 0.964 |
| | PECOS XR-Linear | 0.826 | 0.806* | 0.860 | 0.910 | 0.853 | 0.574 | 0.665 | 0.954 |
| BC5CDR-chemical test | XMR4EL | **0.933** | 0.901 | 0.943 | 0.960 | 0.963 | 0.811 | 0.774 | 0.957 |
| | PECOS XR-Linear | 0.812 | 0.784* | 0.846 | 0.895 | 0.849 | 0.441 | 0.650 | 0.939 |

\* Derived cov × acc. XMR4EL minus PECOS acc@1 is approximately +3.2 / +6.7
percentage points on NCBI dev/test and +12.0 / +12.1 on chemical dev/test.

## Mention-kNN re-scoring ablation

The script fixes beta zero as an eval-only ablation on the same trees. Saved
population/string hashes, all other search settings and R@cand match the beta
10 evaluations. Candidate recall is unchanged; ranking changes.

| Set | Dev acc@1, beta 10 → 0 | Test acc@1, beta 10 → 0 | Net correct mentions lost, dev / test |
| --- | --- | --- | --- |
| NCBI-disease | 0.926 → 0.892 | 0.895 → 0.771 | 25 / 114 |
| BC5CDR-chemical | 0.946 → 0.603 | 0.933 → 0.631 | 1,776 / 1,559 |

On chemical dev, seen-1 accuracy drops from 0.977 to 0.636 without mention-kNN.
The beta-zero result still includes leaf matcher × routing path scores; it is
not a flat classifier or a routing-only measurement.

## Recorded cost

These are measurements from the same server run, not repeated timing trials.
XMR4EL training and test prediction logs show SapBERT embedding-cache hits;
the 1-NN logs do not record cache state. The XMR4EL numbers therefore do not
measure cold transformer encoding.

| Measurement | NCBI-disease | BC5CDR-chemical |
| --- | --- | --- |
| XMR4EL training, encode / hierarchy / runner total (seconds) | 72.5 / 39.4 / 116.0 | 441.4 / 725.8 / 1,261.0 |
| XMR4EL train + dev peak RSS, recorded scale | 5.1 | 21.8 |
| XMR4EL test batch prediction (ms / kept mention) | 2.77 | 3.97 |
| 1-NN command peak RSS, both baselines, recorded scale | 3.2 | 15.1 |
| PECOS training on exports (seconds) / train + dev peak RSS, recorded scale | 32.3 / 3.6 | 282.2 / 27.0 |
| Feature export peak RSS, recorded scale | 6.6 | 31.0 |

Training stage durations come from logs; XMR4EL runner total includes loading
and saving, while PECOS training excludes XMR4EL feature fitting/export and
model saving. Test `ms_per_query` times the batch prediction including encoding
and kNN, excluding model loading and metric computation; it is not single-query
request latency.

Memory values reproduce `peak_rss.tsv`: the script divides Linux
`ru_maxrss` (KiB) by 1,000,000 and rounds to one decimal. Earlier documentation
labelled these values “GB”; this recorded scale is neither exact decimal GB nor
GiB. The peak is the child command's process high-water RSS, not whole-container
RAM or GPU VRAM. The 1-NN peak covers the process running both SapBERT and TF-IDF;
the PECOS peak covers training plus dev prediction, not training alone.

## Retrain qualification and limits

Compared with the saved 2026-10-20 training, the new trees change net correct
counts by **+1 / −1** on NCBI dev/test and **−3 / +13** on chemical dev/test.
Recorded config bytes, relevant input hashes, search settings, label/row counts
and string-group sizes match. Both 1-NN reports are identical after excluding
the saved-tree path. Older XMR4EL metric records lack `eval_protocol`,
`population_sha256` and `strings_sha256`; this is a descriptive comparison of
recorded counts, not a comparison accepted by the current strict TUI gates.

This one earlier retrain shows variation of 1–13 net correct mentions per
split; it is not a repeated-seed estimate, confidence interval or a universal
noise threshold. Small differences need further evidence before attributing
them to an implementation change.

The result covers two datasets and one new tree per set. In-vocabulary scores
exclude unseen concepts; cov × acc makes that coverage loss visible. The
ambiguous-string group (seen >1) has only 17–111 mentions per split. Shared
features do not control PECOS's hierarchy, loss, score fusion or candidate count, and XMR4EL
remains below SapBERT 1-NN on the chemical set. No benchmark was rerun during
this documentation session.

## Reproduction recipe

The existing script builds `xmr4el` and `xmr4el-pecos` images, requires Docker
with CUDA GPU access and the pinned local PubTator files under
`datasets/ncbi_disease/` and `datasets/BC5CDR/chemical/` (each with
`train_plus_ctd.pubtator`, `dev.pubtator`, `test.pubtator`). Start in the repository
root with the `outputs/` parent present:

```bash
nohup bash scripts/run_validation.sh > outputs/validation_run.log 2>&1 &
```

The script checks clean tracked files and seven config/input hashes, and refuses
existing `outputs/validation/` or `outputs/pecos/validation/` directories. Its
top-level environment variables control CPU set/thread count, RAM, niceness,
CPU shares and GPUs; defaults match the recorded resource limits above.

It trains each tree, evaluates dev/test, runs the beta-zero ablation and both
1-NN baselines, exports fitted features, trains PECOS once and reuses it for
test prediction, then scores its predictions. It writes `outputs/validation/`,
`outputs/runs/<timestamp>_val_<set>[_test|_knn0]`, saved trees and PECOS exports,
and packages run records/logs as `outputs/validation.tar.gz`. The final archive
omits the saved trees and PECOS exports/models/predictions.

This recipe was inspected, not executed. For the recorded source use commit
`aeca7a4`; the Dockerfiles install some unpinned dependencies, and the recipe
does not pin embedding-cache state. A new build/run is not guaranteed to
reproduce the recorded image IDs, timing or exact predictions.
