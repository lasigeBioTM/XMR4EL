# Documentation inventory

## Current layout

`README.md`, `AGENTS.md` and `CLAUDE.md` remain at the repository root.
The instruction files stay tracked on the local development branch for `origin`;
omit them from the separate snapshot prepared for `upstream`.
The five reference documents live in `docs/`: `PIPELINE.md`, `SCRIPTS.md`,
`TUI.md`, `BENCHMARK.md` and this inventory. Dockerfiles live in `docker/`;
build commands still run from the repository root with `.` as the context.
The README retains its Dashboard and Results screenshots and test-batch caption.

The session records below describe the earlier consolidation and its original
root paths. They are historical evidence, not instructions for further changes.

## Scope and baseline

Consolidate documentation on `docs/current-state`, created from local `tui` at
`4b345f0bc0c35a03bed6f0bf7666b0ef47503e0f` on 2026-11-02. Make one local
checkpoint commit per session; do not merge or push. Complete only the next
unfinished session, update this inventory, commit, then stop.

The consolidation's final Markdown set was exactly eight root files: `README.md`, `PIPELINE.md`,
`TUI.md`, `SCRIPTS.md`, `BENCHMARK.md`, `AGENTS.md`, `CLAUDE.md`, `INVENTORY.md`.
Remove future-work documentation without recreating a backlog. Preserve all
non-Markdown assets, datasets, outputs and runtime behavior. Source edits may
change documentation references and explanatory text only.

Session 1 found **24 original Markdown files: 23 tracked and one untracked**.
Discovery used `git ls-files --cached --others --exclude-standard`, with
case-insensitive Markdown extensions, and a separate instruction-file search.
Only root `AGENTS.md` and `CLAUDE.md` were found; there are no nested instruction
files in scope. `INVENTORY.md` is new and excluded from the original count.

`CHANGES_TO_TUI.md` was untracked. Session 1 included it in the first checkpoint
so its later removal preserves the request in Git history, changing only trailing
whitespace and the final newline. No original file was deleted in Session 1.

## Original inventory and disposition

Paths are historical inventory entries, deliberately formatted as code rather
than links. **Read** means headings and purpose-bearing sections were inspected;
it does not verify implementation claims. S2–S5 completed the necessary
verification/extraction below. **Session 6 removed all 21 superseded originals**
after matching their dispositions to the 24 originals preserved at `e5d97ef`.
The three rewritten original root files and five added root references/inventory
formed the consolidation's eight-file Markdown set. Dated audit findings are not current
defects unless verified against current code.

| Original path | Actual purpose | Disposition | Destination | Verification status |
| --- | --- | --- | --- | --- |
| `README.md` | Setup, model overview, inputs, usage, TUI introduction, configuration, Docker, security and checks. | Rewritten entry point; pipeline/commands consolidated; interface details consolidated in S3. | `README.md`, `PIPELINE.md`, `SCRIPTS.md`, `TUI.md` | S2 core claims and S3 retained interface claims verified. Original wording remains in the S1 checkpoint; S6 root TUI/benchmark navigation repaired. |
| `AGENTS.md` | Repository conventions, pipeline model, invariants and diagnostic guidance. | Rewrite with concise durable safeguards; pipeline explanations extracted in S2. | `AGENTS.md`, `PIPELINE.md` | S5 shared safeguards rewritten and source-checked, including label/persistence, metric protocol and comparison authority; detailed behavior stays in root references. |
| `CLAUDE.md` | Shared-rule duplication, commands, label/comparison invariants, status handoff and model-switching instructions. | Move unique durable safeguards to shared instructions; replace with a short pointer. | `AGENTS.md`, `CLAUDE.md` | S5 complete: unique durable safeguards retained in AGENTS; replaced by a three-line pointer without handoff priorities or model switching. |
| `STATUS.md` | Release/session handoff, completed work, next steps, research interpretation and working rules. | Extract durable rules or necessary current limitations; remove handoff/history. | `AGENTS.md`, `README.md`, `TUI.md`, `BENCHMARK.md` | S2 core/S3 interface boundaries verified; S4 tuning account explicitly qualified as project-reported, not independently established. S5 durable execution/data and scientific interpretation rules extracted; superseded original removed in S6. |
| `CHANGES_TO_TUI.md` | User request: after a launched batch, Dashboard `n` should start a new experiment rather than reopen Review. | Preserve implemented behavior in the reference; completed note removed in S6. | `TUI.md` | S3 verified `batch_launched` / `action_new_experiment` and the passing `test_new_after_a_launch_starts_a_new_experiment` regression. |
| `docs/README.md` | Index separating current references, TUI contracts, audits and archive. | Navigation consolidated in root README; index removed in S6. | `README.md` | S2 complete; current links checked. S6 root TUI/benchmark destinations repaired; superseded index removed. |
| `docs/pipeline.md` | Implemented pipeline table, persistence/label invariants, synthetic checks, observations and baseline qualifications. | Current behavior/boundaries consolidated; proposed experiments and dated result claims omitted. | `PIPELINE.md`, `README.md`, `SCRIPTS.md` | S2 retained claims verified against implementation/configs; superseded original removed in S6. |
| `scripts/README.md` | Script catalog, local data recipes, training/evaluation/diagnostics, tracked runs and baseline commands. | Commands and local conversion recipes consolidated; historical experiment/result commentary omitted. | `SCRIPTS.md`, `README.md` | S2 source/parser/help checks complete, with external-help limits below; datasets/recipes not executed. |
| `tests/README.md` | Test prerequisites, coverage map, subset commands and regression/self-check guidance. | Development checks and useful test recipes consolidated. | `README.md`, `SCRIPTS.md` | S2 paths and prerequisites checked; targeted tests executed, full suite not rerun. |
| `docs/design/tui.md` | Built screen/workflow contract, keyboard use, presentation, progress, process ownership and comparison rules. | Merge verified implemented interface behavior; remove superseded document in S6. | `TUI.md` | S3 retained workflows, keys, progress, execution and comparison claims checked against source and synthetic tests; illustrative layouts/history omitted. |
| `docs/design/tui_architecture.md` | Module ownership, run data model, comparison evidence table, process control, persistence and tests. | Merge architecture and evidence contracts with interface reference. | `TUI.md`, `AGENTS.md` | S3 file/module ownership and evidence gates verified; S5 shared comparison rule points to the root TUI evidence contract. Superseded original removed in S6. |
| `docs/benchmark.md` | Current 2026-11-02 two-dataset results, systems, metrics, ablation, costs, retrain noise and reproduction. | Verify recorded evidence; consolidate using uppercase filename. | `BENCHMARK.md` | S4 all 160 result-table values checked against archived JSON/PECOS logs; costs/populations/derived counts verified, evidence and tuning/retrain limits qualified. Superseded original removed in S6. |
| `docs/results.md` | Dated ablation/results appendix from September–November, historical configurations and commands. | Inspect only reproduction details or qualifications required by retained benchmark claims; remove historical experiments. | `BENCHMARK.md`, `SCRIPTS.md` | S4 consulted only the 2026-11-02 section; checked its retrain qualification directly against October metrics/config/input records and baseline JSON. No historical experiment tables copied. |
| `docs/future_work.md` | October release snapshot, research priorities, proposed experiments and resume guidance. | Remove; do not recreate priorities, backlog or old result tables. | None | Purpose read; removed in S6. |
| `docs/archive/architecture.md` | 2026-10-23 cleanup assessment, component/data contracts, old TUI readiness and synthetic validation record. | Core boundaries/limits extracted after source checks; omit assessment/history. | `PIPELINE.md`, `README.md`, `TUI.md` | S2 core/S3 TUI boundaries verified directly; pre-TUI readiness claims not retained. |
| `docs/archive/experiment_management.md` | 2026-10-09 baseline inventory and proposed orchestration/API/specification design. | Remove proposal; retain only necessary current reproducibility/comparison qualifications verified elsewhere. | `TUI.md`, `BENCHMARK.md` if needed | S3 supervision/provenance/comparison and S4 benchmark evidence limits verified directly; proposed APIs not retained. |
| `docs/archive/full_cuda_comparison.md` | Historical full-vocabulary CUDA/PECOS procedure referencing deleted config and old runtime assumptions. | Obsolete recipe not retained; S2 external comparison qualifications verified from current scripts. | `SCRIPTS.md`, `BENCHMARK.md` if needed | S2 complete; S4 reproduction checked against current validation/baseline scripts and archived logs, without retaining the obsolete recipe. |
| `docs/archive/status_log.md` | Long historical status/session transcript, old research work and completed TUI development log. | Reranker scope documented from current implementation without rereading the transcript; remove history. | `PIPELINE.md`, `TUI.md` if needed | S2 reranker/S3 current TUI behavior verified from source; no transcript extraction needed. |
| `docs/audits/tool_readiness_audit.md` | 2026-10-09 readiness audit and proposed execution/validation fixes, predating current TUI. | Current core limits verified directly in source; audit/roadmap not carried forward. | `README.md`, `PIPELINE.md`, `TUI.md` if needed | S2 core/S3 current TUI boundaries complete from source; dated findings omitted. |
| `docs/audits/comprehensive_audit_2026-10-16.md` | Dated broad correctness, packaging, resources and security findings with synthetic check record. | Necessary current memory/trust boundaries retained after independent source checks; dated findings omitted. | `README.md`, `PIPELINE.md` | S2 retained claims verified; superseded original removed in S6. |
| `docs/audits/code_quality_audit_2026-10-16.md` | Separate dated quality assessment covering validation, memory, dependencies and artifact integrity. | Necessary current validation/cache/tie/trust limits retained after source checks; dated findings omitted. | `README.md`, `PIPELINE.md` | S2 retained claims verified; superseded original removed in S6. |
| `docs/audits/tui_application_ux_integration_audit_2026-10-28.md` | Dated notification/navigation/lifecycle findings, test evidence and remediation priorities. | Verify necessary current interface limits; remove dated findings and remediation list. | `TUI.md` if needed | S3 notification/navigation behavior checked against current implementation/tests; no dated findings carried forward as current defects. |
| `docs/audits/new_experiment_step6.md` | 2026-10-29 engineering acceptance record, repairs and bounded preflight/process/provenance/manual-check limits. | Consolidate verified current limits, omitting acceptance/history. | `TUI.md`, `README.md`, `SCRIPTS.md` | S2 core/S3 TUI limits complete. Corrected superseded missing-eval-train_path claim; retained untested environment and crash/preflight/provenance limits. |
| `docs/audits/code_quality_architecture_review_2026-10-30.md` | Dated subsystem architecture review, five findings, reproductions and implementation roadmap. | Core invariants/current limits verified from source; omit findings list and roadmap. | `TUI.md`, `PIPELINE.md` if needed | S2 core/S3 TUI contracts complete, including scan writes, event/sample memory and corrected eval train_path; roadmap/old defect claims omitted. |

## Evidence map

Session 1 checked candidate paths for existence only. S2–S5 statuses below
identify source/config/parser and saved-report verification; executed checks are
listed separately under each session record. S4 read raw benchmark evidence
without rerunning models. Original Markdown is a consolidation input, not
implementation evidence. Record unavailable evidence explicitly instead of
inferring support.

| Retained document section | Relevant source/config/test/report | Verification |
| --- | --- | --- |
| `README.md`: setup, packaging, Docker and development | `pyproject.toml`, `uv.lock`, `xmr4el.dockerfile`, `.github/workflows/ci.yml`, `tests/smoke_installed.py` | S2 declarations/recipe checked; no dependency sync, build, fresh wheel install or GPU execution. |
| `README.md`: inputs, quick start and limitations | `scripts/train.py`, `scripts/evaluate.py`, `xmr4el/data/readers.py`, both base/BC5CDR presets, `tests/data/test_readers.py` | S2 source/help/preset checks; reader tests passed. Example corpus paths are placeholders, not inspected data. |
| `PIPELINE.md`: public API, labels and persistence | `xmr4el/xmodel.py`, `xmr4el/hierarchy/tree.py`, `tests/integration/test_label_mapping.py`, `tests/integration/` | S2 implementation verified; mapping regression passed with stubbed encoder/hierarchy. Other persistence tests are references, not rerun evidence. |
| `PIPELINE.md`: features and label representations | `xmr4el/features/encoder.py`, `xmr4el/features/vectorizers.py`, `xmr4el/features/reduction.py`, `xmr4el/features/transformers.py`, `xmr4el/features/label_embeddings.py`, `tests/features/` | S2 implementation/presets verified, including dense/sparse behavior, revision/cache limits and block normalization. No real transformer execution/download. |
| `PIPELINE.md`: hierarchy, matchers, traversal and leaf scoring | `xmr4el/hierarchy/clusterers.py`, `xmr4el/hierarchy/node.py`, `xmr4el/hierarchy/tree.py`, `xmr4el/learning/classifiers.py`, `xmr4el/learning/matcher.py`, `xmr4el/learning/scoring.py`, `tests/hierarchy/`, `tests/learning/` | S2 source-verified targets, augmentation, branching, path product, kNN-before-topk, device handling and limits. No classifier/tree fit or hardware validation. |
| `PIPELINE.md`: evaluation and optional reranker | `xmr4el/eval.py`, `xmr4el/rerank.py`, `scripts/train_reranker.py`, `scripts/evaluate.py`, `tests/test_eval.py` | S2 source/help verified; metric/evidence tests passed. Reranker training/inference not run. |
| `SCRIPTS.md`: commands, inputs/outputs and checks | Parsers/entry points under `scripts/`, `configs/experiments/bc5cdr.json`, `tests/data/`, `tests/test_run_experiment.py`, `tests/test_nn_baselines.py`, external Dockerfiles | S2 parser/defaults/help and example syntax checks complete; PECOS/scispaCy help unavailable locally (missing packages), source verified. No recipes executed. |
| `TUI.md`: screens, navigation and session draft | `xmr4el_tui/app.py`, `xmr4el_tui/screens/`, `xmr4el_tui/draft.py`, `tests/tui/test_dashboard.py`, `tests/tui/test_model_builder.py`, `tests/tui/test_review_batch.py`, `tests/tui/test_workflow_e2e.py` | S3 source/Pilot checks include the original `n` request, session resume, Apply/discard, queue order and reconnection. Actual terminal rendering/clipboard/SSH not checked. |
| `TUI.md`: dataset, builder, review and generated specs | `xmr4el_tui/datasets.py`, `xmr4el_tui/components.py`, `xmr4el_tui/review.py`, `tests/tui/test_dataset_setup.py`, `tests/test_tui_components.py` | S3 bounded sniff, validation, template, copied plan, typed defaults and eval train_path verified. Backend constructor/config checks run; no trained scientific output or corpus inspection. |
| `TUI.md`: files, scans, progress and execution ownership | `xmr4el_tui/runs.py`, `xmr4el_tui/progress.py`, `xmr4el_tui/logs.py`, `xmr4el_tui/services/experiment_repository.py`, `xmr4el_tui/process.py`, `xmr4el_tui/procinfo.py`, `xmr4el_tui/batch.py`, `xmr4el_tui/coordinator.py`, `xmr4el/telemetry/`, `tests/tui/test_batch_audit.py` | S3 source/synthetic process and telemetry checks; preserves locks, ownership, graceful Run Cancel versus escalating Batch Stop, survivor/LOST policy, file contracts and bounded-log/unbounded-event limits. Linux/hardware/power-loss behavior not exercised. |
| `TUI.md`: comparability and allowed metric deltas | `xmr4el_tui/services/comparison.py` (`assess`, `allowed`), `xmr4el/eval.py` (`EVAL_PROTOCOL`, `evaluate_tree`), `scripts/run_experiment.py`, `tests/tui/test_results.py`, `tests/test_run_experiment.py` | S3 verified distinct eligibility/protocol/population/string gates, strict derivation, Hybrid versus Hybrid_all, null values, malformed evidence and provenance refresh. Stubbed metric/input-mutation tests run; real runner seen/unseen training regression inspected, not executed. |
| `BENCHMARK.md`: current results and provenance | Local `outputs/server/validation.tar.gz` (95 files), `outputs/server/validation_run.log`; original benchmark and only `docs/results.md` § 2026-11-02 as consolidation inputs | S4 six clean completed run records, eight metric JSONs, four baseline JSONs and four PECOS score logs checked; all 160 result-table values match. Config bytes and seven recorded input hashes match the preset/script pins. No corpus rehash or prediction recomputation. |
| `BENCHMARK.md`: costs and retrain qualification | Bundle `peak_rss.tsv`, `env.txt`, run resources/logs; local `outputs/server/runs_2026-10-20/`, `outputs/server/nn.tar.gz` | S4 timings, cache evidence, RSS values, correct-count deltas and baseline equality checked for these two sets. Corrected RSS units and command scope. Older metrics lack protocol/population/string hashes; no repeated-seed estimate. |
| `BENCHMARK.md`: systems and reproduction | `scripts/run_validation.sh`, three baseline scripts, `xmr4el/eval.py`, `xmr4el/features/encoder.py`, `xmr4el/features/transformers.py`, `xmr4el/xmodel.py`, both dataset specs, `configs/xmr4el_bc5cdr_config.json`, both Dockerfiles | S4 retained definitions, systems, filtering/export contract, candidate limits and recipe checked in source and logs; involved runtime sources/configs match recorded commit `aeca7a4`. Saved validation trees and PECOS exports/predictions absent locally; source/tuning/reproduction limits stated. No recipe executed. |
| `AGENTS.md` / `CLAUDE.md`: durable safeguards | Original root instructions; `STATUS.md` working/interpretation rules; runtime sources and label-mapping regression linked from `AGENTS.md`; `xmr4el/features/label_embeddings.py`; `tests/test_eval.py`, `tests/tui/test_results.py` | S5 current source checked for sorted labels/empty groups, load guards, row alignment, label-level leaves, reassignment, score order, post-read hashes and protocol/population/string gates. Root links checked; existing regressions inspected, not rerun. Handoff priorities and prescribed model switching removed from instructions. |
| All documents and explanatory references | Links/anchors in retained Markdown; source/CLI/build/configuration text | S6 complete: 125 local links/images/anchors and 51 non-Markdown documentation references resolve. Repaired 42 non-Markdown files, including shorthand TUI section citations missed by the initial 10-file path search. No stale references outside historical inventory entries. |

## Session checkpoints

- [x] **1. Inventory:** branch created; all 24 originals have purposes, destinations,
  dispositions and verification status; no files deleted. Checkpoint adds this
  inventory and preserves the original untracked request note.
- [x] **2. Core documentation:** write verified `README.md`, `PIPELINE.md`,
  `SCRIPTS.md`; retain original inputs until Session 6.
- [x] **3. TUI documentation:** write verified `TUI.md`, preserving execution and
  comparison contracts.
- [x] **4. Benchmark:** write `BENCHMARK.md` with recorded provenance, evidence
  availability and limits; exclude historical experiment tables.
- [x] **5. Agent instructions:** rewrite shared root rules and short Claude
  pointer; no redundant nested instructions were found.
- [x] **6. References and removal:** repaired links/anchors/source references,
  removed 21 superseded originals, and verified exactly the eight intended root files.

Session 1 checks: inventory coverage, evidence-path existence, unchanged original
contents except the request note's whitespace, and `git diff --check`. No runtime
claims or benchmark values have been independently verified in this checkpoint.

### Session 2 verification (2026-11-02)

Rewrote `README.md` as the entry point; added `PIPELINE.md` and `SCRIPTS.md`.
Retained claims were checked against the S2 implementation/config/parser paths
above. Corrected preset versus code defaults, PubTator-only evaluation,
dense label/weight memory, post-topk candidate recall and empty-evaluation
failure behavior. Preserved label/persistence, score fusion, local-checkpoint
cache and trusted-pickle boundaries. Historical quality claims, proposed
experiments and audit roadmaps were not copied.

Checks completed:

- 18 CLI help invocations passed: train, evaluate, runner, reranker, split, CTD,
  converter plus both subcommands, three diagnostics, two screens, 1-NN, PECOS
  export/score tool, latency tool, and TUI module. Beam sweep's no-argument
  positional usage returned its expected exit 1.
- `pecos_run.py --help` and `scispacy_run.py --help` could not import absent
  external packages. Their retained options/defaults were read from source;
  no dependency installation or container execution was attempted.
- Assertions matched documented search constants, protocol, both presets and
  CLI defaults against source/constructors, without fitting models.
- All 69 local links/anchors in the three documents resolved. All 23 Bash and
  two embedded Python blocks passed syntax checks without executing recipes.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/data/test_readers.py tests/integration/test_label_mapping.py tests/test_eval.py -q --tb=short`
  — **10 passed**; encoder/hierarchy fitting was stubbed in mapping/CLI tests.
- `git diff --check` passed before the checkpoint commit. Only these three
  documents and this inventory changed; no original path or non-Markdown asset
  was deleted, and runtime behavior was not changed.

Not checked: full pytest/self-check suite, datasets, trained artifacts, numerical
benchmark evidence, real model downloads, GPU execution, Docker builds or fresh
installation. TUI workflow/comparison verification belongs to S3. The existing
TUI/benchmark links remain valid until their root replacements are written;
repository-wide old references/removal remain for S6.

### Session 3 verification (2026-11-02)

Added `TUI.md`, consolidating the interface and architecture references after
checking their retained claims against the S3 source/test paths above. Preserved
execution ownership, locks, file contracts, graceful run cancellation versus
escalating batch stop, coordinator-loss/surviving-runner behavior and independent
protocol/population/string gates (including Hybrid versus Hybrid_all).

Verified the original `CHANGES_TO_TUI.md` request in `app.batch_launched` /
`action_new_experiment` and its regression: after launch, Dashboard `n` creates
a fresh draft at Dataset Setup. Session `c` resume restores editing only.
Corrected superseded audit claims: generated evaluations now include the shared
training `train_path`; scans can persist LOST reconciliation; bounded log reads
do not bound event-history or active-stage sample memory. No audit defect list,
acceptance verdict, historical measurements or future-work roadmap was copied.

Checks completed:

- TUI CLI `--help` passed. Assertions checked 11 timing/count/protocol constants,
  eight builder slots, generated evaluation `train_path`, and the absence of
  backend imports from the TUI package. No model fitting or corpus reads.
- All 31 local links/anchors in `TUI.md` resolved; all three Bash blocks passed
  syntax checks without executing installation, Docker or launch recipes.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/tui/ -q --tb=short`
  — **312 passed, 11 failed** in the restricted macOS sandbox. Ten failures
  were process/identity checks (inspection returned unavailable UID/start
  information); the other was active-batch quit teardown. Every failed test
  passed in the outside-sandbox reruns below; the entire suite was not rerun
  outside the sandbox. Tests use temporary roots and fake runners, not training.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_eval.py tests/test_telemetry.py tests/test_tui_components.py::test_closed_components_take_exactly_their_fields tests/test_tui_components.py::test_values_the_builder_rejects_the_backend_rejects tests/test_tui_components.py::test_every_builder_config_round_trips_through_load_config tests/test_run_experiment.py::test_input_changed_after_fingerprint_fails_eval -q --tb=short`
  — **31 passed**. Metrics and runner evaluation use stubs; GPU telemetry checks
  mock hardware. Constructor/config round trips do not train.
- Isolated rerun of
  `tests/tui/test_batch_ui.py::test_quit_with_an_active_batch_offers_three_choices`
  — **1 failed** during `run_test` teardown: `app.on_scanned` → `apply_scan` →
  `toast` accessed an empty screen stack (`textual.app.ScreenStackError`). This
  restricted macOS result is qualified by the outside-sandbox reruns below,
  rather than presented as a verified interactive defect.
- Outside the macOS sandbox,
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/tui/test_process.py tests/tui/test_procinfo.py -q --tb=short`
  — **13 passed**. The isolated active-batch quit regression also passed outside
  the sandbox (**1 passed**). No runtime/test code changed between runs.
- `git diff --check` and staged whitespace checks passed before the checkpoint;
  no original path was deleted.

Not checked: the full repository suite, biomedical data, trained artifacts,
scientific runner training/seen-unseen integration, benchmarks, model downloads,
GPU execution, Linux process control, SSH disconnect/reattach, actual terminal
rendering/clipboard, Docker builds, fresh installation or power-loss durability.
Only `TUI.md` and this inventory change; originals, non-Markdown assets and
runtime behavior remain untouched. Root navigation/source-reference migration
and removal remain S6.

### Session 4 verification (2026-11-02)

Added `BENCHMARK.md` for the 2026-11-02 two-dataset validation only, retaining
comparison populations, systems, result tables, the fixed beta-zero ablation,
costs, recorded provenance, retrain qualification and the existing reproduction
recipe. Original Markdown remains in place. No historical experiment tables or
future-work proposals were copied. Only the 2026-11-02 section of
`docs/results.md` and the necessary tuning account in `STATUS.md` were consulted.

Available evidence was read directly from the local validation archive and log,
the two datasets' October run records, and their earlier baseline reports. No
archive was extracted or modified, and no dataset or pickle/model was opened.
Six run records report commit `aeca7a449ce6e5730ff729e975c4326fcfc3be43`, clean
code and completed status; both base runs record fresh training. Source checks
confirmed that the involved runtime code, preset, specs and Dockerfiles still
match that commit.

Corrections and qualifications retained:

- String keys are stripped/lowercased text before `[SEP]`, after configured
  abbreviation expansion; hybrid uses the in-vocabulary denominator. Totals
  count loaded rows, not annotations already skipped by the loader.
- PECOS uses different hierarchy/loss/scoring, no mention-kNN, and 100 returned
  candidates versus XMR4EL's recorded 1,000. The `_dev` export directory supplies
  training matrices; it does not train PECOS on dev labels.
- RAM figures reproduce the script's Linux `ru_maxrss / 1e6` scale, not exact
  GB/GiB. The 1-NN peak covers both baselines; the PECOS peak covers train + dev.
  Training/test XMR4EL cache hits and batch-prediction timing scope are explicit;
  the 1-NN logs do not establish cache state.
- October versus November net correct-count changes are +1 / −1 on NCBI and
  −3 / +13 on chemicals; baseline JSON is identical apart from tree paths.
  Older metrics lack protocol/population/string hashes, so this descriptive
  comparison is not a current strict TUI delta or a repeated-seed noise bound.
- The bundle omits corpora, fitted transformer revision metadata, saved trees,
  exports, PECOS models and prediction matrices. The referenced validation
  trees and PECOS export directory are absent locally. PECOS values are from
  rounded score logs. Local input bytes were not rehashed; count alignment is
  weaker than population hashes. Dictionary construction, split independence,
  tuning history and independent preregistration are not established here.
- Reproduction is a source-checked recipe, with unpinned image dependencies and
  unspecified cache state; it was not executed and does not promise exact
  images, timings or predictions. Test splits were previously evaluated on older
  trees; “once” applies to each validation tree's regular test evaluation.

Checks completed:

- `.venv/bin/python /private/tmp/xmr4el-session4-check.py` — passed read-only
  standard-library assertions over six completed clean run records, eight metric
  JSONs, four baseline JSONs, four PECOS score logs, seven recorded config/input
  hashes, byte-identical preset copies, prior retrain metadata and baseline
  equality. This temporary checking script is outside the repository checkpoint.
- All **160 result-table values** in both the original and new benchmark match
  raw JSON / PECOS logs after documented rounding. Population/ablation/cost
  tables, candidate recall, correct-count/group/ablation deltas and PECOS point
  differences also checked. No predictions or metrics were regenerated.
- All **8 local links/anchors** in `BENCHMARK.md` resolved. Its one Bash recipe
  and `scripts/run_validation.sh` passed `bash -n` syntax checks without running
  training, benchmarks, Docker or data preparation.
- `git diff --check` and staged whitespace checks passed before the checkpoint.
  Only `BENCHMARK.md` and this inventory changed; originals and non-Markdown
  assets were preserved, and runtime behavior was not changed.

Not checked: corpus/dictionary contents or leakage, individual predictions,
real transformer revisions/cache contents, repeated seeds, cold latency,
hardware execution, dependency installation or image rebuilding. No training,
benchmark rerun or runtime test suite was executed for this documentation-only
session. S3 environment-specific test qualifications remain recorded above.

### Session 5 verification (2026-11-02)

Rewrote `AGENTS.md` as concise shared safeguards; replaced `CLAUDE.md` with a
three-line pointer. Retained local-file scope, user-run corpus work, data/output
preservation, trusted-pickle and no-migration boundaries, focused checks and
scientific reporting rules. Removed duplicated command/pipeline catalogs,
handoff priorities, dated context and prescribed model switching. No nested
instructions or research backlog added.

Checked the retained implementation claims against the S5 source paths above:
sorted binarizer classes and empty groups; root/Z count guards; numeric node
load order and parent/child membership guard; reader/transformer row alignment;
label-level leaves and small-cluster reassignment; routing-path multiplication
and mention-kNN before final topk; post-read input rehashing; metric-protocol
bumps and `assess`/`allowed` authority. The comparison evidence reference now
points to `TUI.md`, preserving independent protocol, population and string gates.

Checks completed:

- Standard-library assertions checked all 17 links/anchors and 12 referenced
  source symbols in the rewritten instructions, required safeguards, the short Claude
  pointer, absence of nested instructions and the three-file change scope.
- `git diff --check` and staged whitespace checks passed before the checkpoint.
  Only `AGENTS.md`, `CLAUDE.md` and this inventory changed; no other original or
  non-Markdown asset was deleted, and runtime behavior was not changed.

Not checked: runtime test suite, datasets, saved models, model downloads,
hardware execution or benchmark reproduction. Existing regressions were
inspected without execution; no training or benchmark rerun was performed.
Repository-wide reference repairs and superseded-file removal remain S6.

### Session 6 verification (2026-11-02)

Completed from checkpoint `17370e7` on `docs/current-state`. Read all eight
retained root documents and confirmed each superseded original's disposition
and completed S2–S5 extraction (or deliberate omission) before removal.

Repaired root navigation in `README.md`, `PIPELINE.md` and `SCRIPTS.md`.
Updated documentation citations in 42 non-Markdown files: 40 Python files,
`pecos.dockerfile` and `scripts/run_validation.sh`. These changes are docstrings,
comments and one CLI help reference only. Replaced obsolete numbered TUI
citations with current root anchors and removed the dated quality claim beside
the builder's fallback config; the config value and defaults are unchanged.
Existing packaging/configuration references already target `README.md`.

Removed all 21 superseded originals, including handoff, audit, archive, request
and future-work documents. Their historical paths remain as code in this
inventory; original contents remain in Git history. No backlog or nested
instructions were added. The final in-scope Markdown set is exactly the eight
root files listed under Scope and baseline. `docs/images/pipeline.svg` and all
other non-Markdown assets are preserved; no datasets, outputs or runtime behavior
were changed.

Checks completed:

- `python3 /private/tmp/xmr4el-session6-check.py` — read-only standard-library
  assertions passed: exactly eight case-insensitive Markdown paths among existing
  tracked/nonignored untracked files; only root agent instructions; all 21
  removals match the inventory and Session 1 checkpoint `e5d97ef`.
- All **125 local links/images/anchors** in the retained documents and **51
  documentation references** in non-Markdown text resolve. No removed-path
  references remain outside this inventory's historical entries. The initial
  10-file search missed shorthand `tui.md` citations, which are now repaired too.
- All **40 changed Python files** compile without importing or executing them.
  AST comparisons against `17370e7` match after excluding docstrings and the one
  explicitly migrated CLI help reference. Shell/Docker non-comment lines match;
  no non-Markdown file was added or deleted, and asset/config values are unchanged.
- `bash -n scripts/run_validation.sh` — passed without running the recipe.
- `git diff --check` and staged whitespace checks passed before the checkpoint.
  The temporary checking script is outside the repository checkpoint.

Not checked: external URLs, runtime test suite, fresh installation, containers,
corpora, saved models, hardware or benchmark reproduction. Earlier S2–S5 evidence
and environment limitations remain applicable. No training, benchmark rerun,
merge or push was performed.

## Consolidation completion

All six documentation sessions are complete; no documentation blocker remains.
The earlier evidence/environment limitations are recorded above. The original
Session 6 checkpoint instructions were to commit locally and stop; they do not
authorize committing or publishing the current layout changes.
