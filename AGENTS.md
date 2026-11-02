# Shared repository safeguards

XMR4EL trains and evaluates label-centric entity-linking models on local
PubTator or grouped TSV + label files. Preserve those data contracts and the
boundaries between preprocessing, features, label embeddings, clustering,
matchers and hierarchical inference.

Use [README.md](README.md) for setup, [PIPELINE.md](PIPELINE.md) for model and
data contracts, [SCRIPTS.md](SCRIPTS.md) for commands, [TUI.md](TUI.md) for
interface/execution contracts and [BENCHMARK.md](BENCHMARK.md) for recorded
evidence. Verify retained claims against current code; distinguish implemented
behavior, inference and unavailable evidence.

## Changes and execution

- Make the smallest complete change; preserve configuration semantics and
  component boundaries. Avoid unrelated refactors, speculative abstractions
  and unnecessary dependencies.
- The user runs corpus training/evaluation; agents provide commands and may run
  focused synthetic checks. Preserve `data/`, `datasets/` and `outputs/`,
  including saved trees, caches and exports.
- Load only trusted saved trees: component loading uses Python pickle and can
  execute code. Do not add compatibility code for old trees/configs; retrain
  incompatible artifacts instead.
- Add or update focused tests when behavior changes. Report checks actually
  run, failures and environment limits; synthetic checks do not establish
  corpus quality, accelerator behavior or benchmark reproduction.

## Pipeline invariants

- **Label index j = column j of Y = row j of Z = initial_labels[j].**
  `XModel._fit` uses the binarizer's sorted fitted `classes_`; empty label
  groups have no column. Map output indices through `initial_labels`, never
  incoming group order. Preserve the root-label count check in `XModel.load`
  and the Z-row count check in `load_research`
  ([regression](tests/integration/test_label_mapping.py)).
- Preserve numeric layer/node loading order and parent-cluster/child-label
  membership validation in `HierarchicalMLModel.load`
  ([tree.py](xmr4el/hierarchy/tree.py)).
- Preserve text/label alignment in [readers.py](xmr4el/data/readers.py): TSV
  label-file entries align with numerically sorted group IDs; PubTator groups
  retain first-seen label order. Transformer deduplication and batching must
  restore input row order ([transformers.py](xmr4el/features/transformers.py)).
- Leaves match individual labels through identity C in `MLModel.train`
  ([node.py](xmr4el/hierarchy/node.py)). Small-cluster reassignment must keep
  labels represented ([clusterers.py](xmr4el/hierarchy/clusterers.py)).
- Preserve leaf matcher score × ancestor routing-path probability in
  `HierarchicalMLModel.predict`; mention-kNN fusion precedes final `topk` in
  `XModel.predict_encoded` ([xmodel.py](xmr4el/xmodel.py)).

## Evaluation and comparison

- Changing a metric definition (formula, denominator, filtering or ties) used
  by `evaluate_tree` requires bumping `EVAL_PROTOCOL` in
  [eval.py](xmr4el/eval.py). Never show deltas between missing or different
  recorded protocols.
- Saved-evaluation comparability is decided only by `assess` in
  [comparison.py](xmr4el_tui/services/comparison.py); `allowed` gates individual
  deltas. Preserve its protocol, population and string-evidence gates. Matching
  paths, batch membership or equal coverage do not prove comparability. See
  [TUI.md's evidence contract](TUI.md#comparison-contract); obtain missing
  evidence by re-evaluating rather than relaxing the gates.
- Preserve the runner's post-read input-hash checks before training or writing
  metrics ([run_experiment.py](scripts/run_experiment.py), `check_inputs`).
- Report kept/total loaded rows and vocabulary coverage with ranking metrics;
  report the chance reference beside routing measurements. Small-vocabulary
  diagnostics do not establish full-label-space performance. Change one
  experimental factor at a time, or identify bundled changes without assigning
  their outcome to one component.
