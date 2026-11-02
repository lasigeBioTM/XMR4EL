# Script reference

Run commands from the checkout root after [setup](../README.md#setup). Examples
require your own local inputs and saved artifacts; replace paths before running.
Use `.venv/bin/python` locally and `python3` in the project container.

## Command map

Paths below are relative to `scripts/`. Python tools accept `--help` except the
positional `beam_sweep.py`. PECOS/scispaCy tools import their external dependencies
before parsing help; they require the corresponding environment even for help.

| Command | Purpose / output |
| --- | --- |
| `split_pubtator.py` | PMID-based train/dev/test splits, optional JSONL. |
| `to_pubtator.py filter` / `bioc` | PubTator type/ID filtering or BioC XML conversion. |
| `dict_to_pubtator.py` | CTD names/synonyms as one-mention PubTator documents. |
| `train.py` | Group, encode, embed labels, cluster, fit node matchers and save a tree. |
| `evaluate.py` | Saved-tree ranking metrics and optional reranker tables. |
| `run_experiment.py` | Validated train/eval or eval-only run with provenance and telemetry. |
| `train_reranker.py` | Experimental cross-encoder over a saved tree's candidates. |
| `diagnose_routing.py` | Root matcher, cosine routing, flat retrieval and dictionary diagnostics. |
| `diagnose_unseen.py` | String-group rank/error analysis; writes an error TSV. |
| `diagnose_context.py` | Context-shift routing diagnostics for three-block trees; writes JSON. |
| `experiments/screen_features.py` | Feature/block retrieval screens without fitting a hierarchy. |
| `experiments/screen_leaf_scorer.py` | Leaf scorers with oracle routing over exported features. |
| `experiments/beam_sweep.py` | Repeated evaluation in beam increments of five. |
| `baselines/nn_baselines.py` | SapBERT / mention TF-IDF nearest-neighbor label ranking; optional JSON. |
| `baselines/pecos_compare.py` | Aligned export, external-prediction scoring and agreement checks. |
| `baselines/pecos_run.py` | PECOS XR-Linear training/prediction in a separate environment. |
| `baselines/scispacy_run.py` | External alias-candidate baseline over exported rows. |
| `bench_latency.py` | Saved-tree / nearest-neighbor timing and memory measurements; JSON output. |

## Prepare PubTator inputs

```bash
.venv/bin/python scripts/split_pubtator.py \
  --input datasets/corpus_pubtator.txt --outdir datasets/splits \
  --train_ratio 0.8 --dev_ratio 0.1 --emit_jsonl
```

Ratios default to 0.8 / 0.1; the rest is test, assigned deterministically by PMID.
Outputs are `corpus_pubtator_{train,dev,test}.txt`, plus `{train,dev,test}.jsonl`
with `--emit_jsonl`. Existing target files are refused; caught write/publish
failures clean up files created by that invocation. For official splits, supply
all three of `--train_pmids`, `--dev_pmids`, `--test_pmids`. Overlapping/unknown
listed PMIDs fail; unlisted corpus documents are omitted and counted.

Filter a local corpus while retaining document headers:

```bash
.venv/bin/python scripts/to_pubtator.py filter \
  -input datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt \
  -out datasets/BC5CDR/chemical/train.pubtator -types Chemical
```

Create output parent directories first. Repeat for the release's dev/test files;
the converter does not invent a split. Type selectors used with local releases:

| Corpus subset | `-types` |
| --- | --- |
| NCBI-disease | `SpecificDisease DiseaseClass Modifier CompositeMention` |
| BioRED disease | `DiseaseOrPhenotypicFeature` |
| BioRED chemical | `ChemicalEntity` |
| BC5CDR disease / chemical | `Disease` / `Chemical` |

`filter` defaults to all types, removes `MESH:` prefixes, preserves other
prefixes such as `OMIM:`, maps absent/`-` IDs to `-1`, and normalizes multi-ID
separators to `|`. The reader still needs per-part mention texts for composites.
It is not a general ontology-ID mapping service.

For NLM-Chem-style BioC files and a split's PMC-ID list:

```bash
.venv/bin/python scripts/to_pubtator.py bioc \
  -input_dir datasets/nlm_chem/FINAL_v1/ALL \
  -pmcids datasets/nlm_chem/FINAL_v1/pmcids_train.txt \
  -out datasets/nlm_chem/train.pubtator -type Chemical
```

The BioC path expects `<pmcid>_v1.xml`; `-type` defaults to `Chemical`. It assembles
passages at their offsets, keeps matching single-location annotations and prints
skip counts. Both converter modes refuse an existing output.

CTD dictionary augmentation:

```bash
mkdir -p datasets/CTD
.venv/bin/python scripts/dict_to_pubtator.py \
  -ctd_path datasets/CTD/CTD_diseases.tsv.gz \
  -out datasets/CTD/ctd_disease_omim.pubtator -type Disease -omim
cat datasets/ncbi_disease/train.pubtator datasets/CTD/ctd_disease_omim.pubtator \
  > datasets/ncbi_disease/train_plus_ctd.pubtator
```

Use a new destination for the concatenation. The converter reads the CTD column
header and synonym columns; names are deduplicated case-insensitively per record.
Only `MESH:` concepts are kept by default, with that prefix stripped; `-omim`
also keeps `OMIM:` concepts. For chemical vocabulary, use the chemicals gzip and
`-type Chemical`. Each name is its pseudo-document title and mention; a context
window therefore contains no surrounding words. Dictionary-only data can leave
the context TF-IDF vocabulary empty. Conversion replaces its output only after
success and refuses using the input file as output.

These are local format recipes, not checks of downloaded releases, file contents,
split independence or benchmark input hashes.

## Train and evaluate

```bash
.venv/bin/python scripts/train.py \
  -train_path datasets/ncbi_disease/train.pubtator \
  -model_config configs/xmr4el_bc5cdr_config.json
```

`-model_config` defaults to `configs/xmr4el_base_config.json`; `-ds_len` defaults
to 10,000,000 label groups. PubTator groups are first-seen; TSV groups are sorted
by integer group ID. Counts are not random samples. Training logs the saved path
under `outputs/saved_trees/xmodel_<timestamp>/`. `-verbose` enables DEBUG;
`-quiet` keeps warnings/errors (mutually exclusive).

Choose the exact saved path and evaluation files:

```bash
MODEL=outputs/saved_trees/xmodel_REPLACE_WITH_SAVED_NAME
TRAIN=datasets/ncbi_disease/train.pubtator
DEV=datasets/ncbi_disease/dev.pubtator
.venv/bin/python scripts/evaluate.py \
  -xmodel_path "$MODEL" -test_path "$DEV" -train_path "$TRAIN"
```

The evaluator always reads PubTator. Its `-beam_size`, `-leaf_candidates`,
`-topk` and `-knn_beta` default to the saved tree's `predict_config`, not fixed
CLI values. `topk=0` retains all selected candidates; `knn_beta=0` disables fusion.
Code defaults and constraints are in [PIPELINE.md](PIPELINE.md#training-and-inference).
The optional evaluation `-train_path` provides the reference dictionary for
seen/unseen groups; pass the intended reference, including any augmentation.
`-verbose` enables library diagnostics; warnings remain visible by default.
No eligible evaluation rows gives exit status 1.

### Grouped TSV

Create a TF-IDF config from the base preset (choose a new destination):

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/xmr4el_base_config.json").read_text())
config["features"] = "tfidf"
with Path("configs/local_tfidf_config.json").open("x") as out:
    json.dump(config, out, indent=2)
PY
.venv/bin/python scripts/train.py \
  -train_path data/train.tsv -labels_path data/labels.txt \
  -model_config configs/local_tfidf_config.json
```

Labels align with sorted group IDs, one nonblank label per group. This retains
the preset's hierarchy settings; small datasets may need lower depth, minimum
cluster size or cluster count. Plain-text Python queries work with this tree;
the existing evaluation CLI still requires PubTator gold input.

## Tracked runs

```bash
.venv/bin/python scripts/run_experiment.py -spec configs/experiments/bc5cdr.json -check
.venv/bin/python scripts/run_experiment.py -spec configs/experiments/bc5cdr.json
.venv/bin/python scripts/run_experiment.py \
  -spec configs/experiments/bc5cdr.json -xmodel_path "$MODEL"
```

The first command validates without training, loading pickles or creating a run
directory; it checks referenced paths and returns one JSON object. It does not
fully parse corpora or prove vocabulary/SVD/hierarchy feasibility. All examples
need the referenced local files. `-spec -` reads JSON from stdin.

Training specs require `name`, `model_config`, `train_path`, and nonempty `evals`.
Each eval requires `split`, `test_path`; optional `train_path` supplies string
statistics and the four search keys override saved defaults. Eval-only mode
needs `name` and `evals`, plus CLI `-xmodel_path`; it does not retrain. Unknown keys,
duplicate splits, missing files and invalid search values fail validation.
There is no runner `ds_len` or grouped-TSV option.

Runs default to `outputs/runs/<timestamp>_<name>/`; `-run_dir` selects another
fresh directory. Existing run directories are refused. Outputs include:

| File | Contents |
| --- | --- |
| `spec.json`, `model_config.json` | Captured spec; captured model config for training. |
| `git.json`, optional `git_diff.patch` | Commit, dirty flag, untracked paths and tracked diff; not an executable snapshot of all source files. |
| `provenance.json` | Input SHA-256 hashes, package versions, interpreter/platform and selected Torch device. |
| `tree.json` | Saved tree path, ID and whether this run trained it. |
| `metrics_<split>.json` | Shared `evaluate_tree` metrics, search and evaluation evidence. |
| `run.log`, `events.jsonl`, `heartbeat`, `resources.json` | Logs, progress, liveness and resource summaries. |
| `status.json` | Running, completed, failed or stopped state, process identity and ending details. |

`-resource_interval` defaults to 5 seconds; 0 disables resource telemetry.
Input hashes are rechecked after reading training/evaluation inputs and before
publishing derived results. An empty eligible evaluation writes diagnostic
metrics and then fails the run. Identical train/eval files are rejected in
training mode; overlapping documents in separate files are not detected.
SIGTERM/SIGINT records a stopped run; SIGKILL cannot record a final state.

The launch-binding options `-model_config_sha256` and `-tree_id` reject changed
reviewed artifacts with exit 3. Detailed process/comparison ownership belongs to
the [TUI reference](TUI.md).
Implementation: [run_experiment.py](../scripts/run_experiment.py); checks:
[test_run_experiment.py](../tests/test_run_experiment.py).

## Diagnose and screen

Measure saved trees before choosing a retrain:

```bash
.venv/bin/python scripts/diagnose_routing.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" -max_rows 5000
.venv/bin/python scripts/diagnose_unseen.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -out outputs/logs/unseen_errors.tsv
```

Routing uses a seeded sample (default 5000 per split) and prints chance/fixed-cluster
references alongside matcher/cosine/flat retrieval. Report the chance line with
the routing numbers. It requires an internal root. Unseen diagnostics honor the
tree's beam/beta unless overridden; the default error path is the one above.
Both expose `-query_block` (default 256) for nearest-label computations.

```bash
.venv/bin/python scripts/experiments/screen_features.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -features 'sapbert+charsvd,sapbert+charsvd+ctxwin*0.5' \
  -scorers centroid,max -abbrev none,append -show_errors 10
.venv/bin/python scripts/experiments/beam_sweep.py "$MODEL" "$DEV" 5 25 20
```

The sweep has five positional arguments: model, test file, inclusive start/end
beam, topk. It uses the saved beta and fails if any evaluation fails.
`screen_leaf_scorer.py -xmodel_path "$MODEL" -data "$EXPORT"` consumes a PECOS-format
export (below); its cosine/logistic/SVM screens assume perfect routing and are
not end-to-end quality measurements.

## Baselines and external environments

```bash
.venv/bin/python scripts/baselines/nn_baselines.py \
  -xmodel_path "$MODEL" -test_path "$DEV" -train_path "$TRAIN" \
  -baselines sapbert tfidf -out outputs/nn_dev.json
```

Both baselines are selected by default. They use the saved training rows and
label vocabulary, the evaluator's eligible rows and shared ranking definitions.
The SapBERT option needs a saved mention-transformer block; choose
`-baselines tfidf` for a TF-IDF-only tree. JSON output is a separate baseline
report, not a tracked-run metrics/provenance record.

Aligned external export:

```bash
EXPORT=outputs/pecos/comparison
.venv/bin/python scripts/baselines/pecos_compare.py export \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" -out "$EXPORT"
```

Use the exact training file and a new/empty destination. Export verifies texts
and label alignment with research state, then writes `X_trn.npz`, `Y_trn.npz`,
`X_dev.npz`, `y_dev.npy`, `texts.json`, and `chain_<i>.npz`. It includes training
texts; handle the directory with the same access controls as the corpus.

In a PECOS environment with access to that export:

```bash
python3 scripts/baselines/pecos_run.py \
  -data "$EXPORT" -features ours,tfidf -beams 2,10 -topk 100 -threshold 0
```

Back in the project environment:

```bash
.venv/bin/python scripts/baselines/pecos_compare.py score -out "$EXPORT"
```

PECOS is absent from project dependencies; [pecos.dockerfile](../docker/pecos.dockerfile)
provides a separate Linux/amd64 environment. `ours` uses exported features;
`tfidf` fits PECOS preprocessing on the exported texts. Parser defaults are
beams `2,10`, topk 100, branching 8, maximum leaf size 100, weight-pruning
threshold 0.1. The example explicitly disables pruning; match feature rows,
vocabulary, hierarchy, loss, scoring and search when interpreting comparisons.
Shared parameter names do not establish algorithm equivalence.

`-model` reuses a saved PECOS model for prediction and requires `-features ours`.
`-chain -solver L2R_LR_DUAL -post sigmoid` selects the exported hierarchy and
logistic/path-product settings; `pecos_compare.py agree -out "$EXPORT" -beam_size 10`
compares predictions with the tree (kNN off).

scispaCy is also external ([scispacy.dockerfile](../docker/scispacy.dockerfile));
`scispacy_run.py -data "$EXPORT" -kb umls` writes `pred_scispacy_umls.npz` for the
same scorer. Its default UMLS candidate space is for CUI-labelled data such as
MedMentions, not an automatic mapping to MeSH IDs. Neither external runtime was
executed during this documentation checkpoint; CLI help for those two scripts
is unavailable locally because the packages are absent. Their options above
were checked in source.

## Optional reranker

```bash
.venv/bin/python scripts/train_reranker.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -k 10 -epochs 2
.venv/bin/python scripts/evaluate.py \
  -xmodel_path "$MODEL" -test_path "$DEV" -train_path "$TRAIN" \
  -reranker_path outputs/rerankers/REPLACE_WITH_SAVED_NAME -rerank_w 0 0.5 1
```

Training defaults: K 10, epochs 2, learning rate 2e-5, rows/batch 4, max length
96, initialization from SapBERT. By default negatives are in-sample saved-tree
candidates. `-folds N -tree_train_path <file>` (N ≥ 2) instead trains fold trees
excluding each query fold's documents; this can be expensive. Rows with fewer
than K−1 negatives are dropped. Output is under `outputs/rerankers/` and bound to
the saved tree ID. Reranker `train_path` should be the annotated rows to train on;
`tree_train_path` is the full tree corpus, including dictionary augmentation.

Evaluation K defaults to the trained K. Weight defaults are `0 0.5 1 2 inf`;
select a weight on dev and pass it alone for test. This is an optional evaluation
path, not default prediction or the tracked runner's configuration.

## Existing shell recipes

`run_g.sh`, `run_g1.sh`, `run_g2_pecos.sh`, `run_scispacy.sh`, `run_nn.sh`,
`run_i.sh`, `run_eq.sh`, `run_ctx.sh`, `run_lc.sh`, and `run_validation.sh` are
local research recipes. They assume particular datasets, artifacts, output
locations and, in several cases, Linux/Docker GPU images; some select the latest
matching run or replace exports. Inspect the chosen script before use.
`run_validation.sh` has pinned input/config checks and fresh-output requirements;
its benchmark procedure and recorded evidence are reserved for the
[benchmark reference](BENCHMARK.md). No shell recipe was run here.

## Development checks

```bash
uv sync --locked --group dev --extra tui
.venv/bin/python -m pytest tests/ -q
uvx ruff check xmr4el xmr4el_tui scripts tests
.venv/bin/python scripts/evaluate.py -selfcheck
.venv/bin/python scripts/diagnose_routing.py -selfcheck
.venv/bin/python scripts/diagnose_unseen.py -selfcheck
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
.venv/bin/python scripts/experiments/screen_leaf_scorer.py -selfcheck
.venv/bin/python scripts/baselines/pecos_compare.py selfcheck
```

The tests use synthetic files and mocked transformer paths. ML dependencies are
still required; some tests fit small models, and process/TUI checks depend on
the host environment and optional extra. These are not corpus or GPU validations.
Useful subsets:

| Command suffix after `.venv/bin/python -m pytest` | Focus |
| --- | --- |
| `tests/data/ -q` | Readers, IDs, splits and conversion. |
| `tests/features/ -q` | Encoding, adapters, SVD, cache/revision and OOM handling. |
| `tests/hierarchy/ tests/learning/ -q` | Clustering, child inputs, matchers and scoring. |
| `tests/integration/ -q` | Labels, configs and persistence. |
| `tests/test_eval.py tests/test_run_experiment.py -q` | Metrics, evidence and recorded-run behavior. |
| `tests/test_nn_baselines.py -q` | Nearest-neighbor ranks and baseline script. |
| `tests/tui/ -q` | Interface/service/workflow regressions. |
| `tests/ --collect-only -q` | Test discovery without execution. |

The library installs a `NullHandler`; Python applications supply their own
logging handler. `xmr4el.set_verbosity(0/1/2)` selects WARNING/INFO/DEBUG.
CI also runs [smoke_installed.py](../tests/smoke_installed.py) outside a read-only
wheel installation with a mocked transformer; it is not a pytest test or a
real checkpoint-download check.
