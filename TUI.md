# Terminal workbench

The optional Textual application browses `outputs/runs/` and `outputs/batches/`,
edits experiment configurations, and launches sequential training/evaluation
batches. It reads records and starts external processes; the scientific pipeline
stays in [the runner](scripts/run_experiment.py) and [XMR4EL](PIPELINE.md).

This reference describes implemented behavior. Verification and untested
environments are recorded in [INVENTORY.md](INVENTORY.md#session-3-verification-2026-11-02).

## Start and reconnect

From the checkout:

```bash
uv sync --locked --extra tui
.venv/bin/python -m xmr4el_tui --root .
```

`xmr4el-tui --root .` is the installed entry point. `--root` defaults to the
current directory; launches also need the checkout's `scripts/run_experiment.py`.
The process controller uses POSIX sessions, process groups and `flock`, with
identity inspection for Linux and macOS.

Run the TUI where the data and runner are available. For an existing, long-lived
Docker container with the checkout mounted at `/app`:

```bash
docker exec -it -e TERM -e COLORTERM xmr4el-tui \
  python3 -m xmr4el_tui --root /app
```

Keep the container alive independently of the terminal; `--init` reaps orphaned
children. Coordinator and runner sessions write output to files, so closing the
TUI does not terminate them. Reopening reconstructs monitoring from records,
while unpersisted editing state is lost. Actual SSH disconnect/reattach and
container terminal behavior were not exercised in this documentation session.

Copied run records can be browsed without model artifacts. Missing trees are
labelled as missing; a process from another or unknown runtime is UNVERIFIED and
cannot be stopped from this TUI.

## Screens and workflow

| Screen | Purpose and main actions |
| --- | --- |
| Dashboard | Explorer groups trees by dataset and evaluations under their tree; separate groups cover runs without a saved tree. Description shows configuration, files, metrics and View Logs. Active runs includes active batches. `r` opens Results, `b` lists batches, `n` opens New Experiment, `c` resumes the session draft when available. |
| Dataset Setup | Choose a recognized dataset or register local files, map one training file and enabled evaluation splits, then confirm with `ctrl+s`. `m` opens Model Builder with the confirmed mapping. |
| Model Builder | Edit applied configurations for one or more trees; Review Batch opens the final inspection. Component edits use a working copy: Apply (`ctrl+s`) validates the slot and commits it together; Cancel discards it. Leaving an editor with unapplied changes asks before discarding. |
| Review Batch | Inspect Overview and Differences, rename the batch, reorder trees, and use Fix Issue to open the owning dataset/component editor. `e` edits the selected tree; `l` asks to launch, and a second `l` confirms. |
| Batch Monitor | Show batch state, execution order and the selected tree's log. `enter` or `o` opens its run; `s` requests Stop Batch after confirmation. `esc` returns to Dashboard. |
| Run Monitor | Show one run's overview, event-derived pipeline, selected stage's resources and live log. `c` follows the current stage; `f` expands the focused panel. Cancel Run (`s`) is available only for a verified TUI-owned process group. |
| Results | Compare a chosen baseline evaluation with up to three challengers. `space` adds/removes a challenger; `enter` inspects. Groups `1`–`5` are Performance, Ranking, Generalization, Efficiency and Specs. |

Launch closes the New Experiment screens and opens Batch Monitor over Dashboard.
After returning to Dashboard, **`n` creates a fresh draft and opens Dataset
Setup**, rather than reopening the launched Review Batch. Before starting that
new draft, `c` can restore the previous workflow within the same TUI session;
this restores editing, not execution. A failed launch preserves the draft and
Review screen. Source: [app navigation](xmr4el_tui/app.py),
[draft](xmr4el_tui/draft.py), and
[launch/navigation regressions](tests/tui/test_batch_ui.py).

Dataset Setup accepts PubTator. Discovery uses experiment specs, recognized
dataset directories and the local registry. It checks file availability and
identity, rejects a training file reused as an evaluation file (including
symlinks), requires an enabled evaluation and unique split names, and sniffs only
the first **64 KiB**. Ready means these preflight checks passed; it does not
establish full parsing, vocabulary size, SVD rank or hierarchy feasibility.
Discovery, confirmation and the file picker still have synchronous filesystem
paths; slow/large directories can stall them. See [datasets.py](xmr4el_tui/datasets.py).

Model Builder supports the hybrid `sapbert_char_context` recipe: transformer,
mention TF-IDF and SVD, context TF-IDF and SVD, hierarchy, matcher and search.
PIFA is fixed in the diagram. The component registry mirrors backend fields
without importing the backend; unrelated keys are preserved. Integer TF-IDF
document-frequency counts stay distinct from floating-point fractions. New
trees use an available hybrid config from a matching training spec, falling back
to `configs/xmr4el_bc5cdr_config.json`; unsupported recipes are skipped, not
converted. Changing the dataset does not rewrite existing trees.
See [components.py](xmr4el_tui/components.py) and
[Model Builder](xmr4el_tui/screens/model_builder.py).

In Tree Selection, `n` adds, `d` duplicates the applied config, `x` removes after
confirmation (the last tree stays), `shift+left`/`shift+right` reorder, and `F2`
renames. Stable tree IDs identify queued work; names are display text. Review's
Differences reference is the oldest surviving tree, independent of queue order
and Results' baseline. Review compares effective component defaults, preserving
the TF-IDF count/fraction distinction. Invalid issues block launch; duplicate
configs produce a nonblocking Caution. Launch rechecks readiness and copies the
applied configurations into a plan. Each generated evaluation includes the
shared training file as `train_path`, enabling dictionary and seen/unseen
evidence. See [review.py](xmr4el_tui/review.py).

Results opens with the selected evaluation as baseline, identified by
`(run id, split)`. A tree with several evaluations requires a choice. Each new
Results screen starts without challengers; its baseline stays fixed. Show
incompatible reveals inspectable evaluations that cannot be added. Inspection
returns to the same comparison. Written metrics from a failed run remain usable,
subject to the same [evidence gates](#comparison-contract).

### Navigation, logs and notifications

| Context | Keys |
| --- | --- |
| Help / commands | `?` (`F1` in text fields); `ctrl+p` command palette. Dashboard offers `Palette: <name>`. |
| Back | `esc` closes the current overlay, search or editing level, then returns. Contextual help describes each screen's order. |
| Quit | `ctrl+c` asks; without an active batch a second `ctrl+c` confirms and `esc` cancels. With an active batch, choose Leave batch running, Stop batch and exit, or Stay (default). Stop-and-exit waits for stop requests to be written, not for runners to finish. |
| Refresh | Dashboard/Results `ctrl+r` rescans; Dataset Setup `ctrl+r` rediscovers datasets. |
| Panels | Dashboard, Results, Run Monitor and Dataset Setup use `ctrl+tab` / `ctrl+shift+tab` or `]` / `[`. Builder uses nested Tree Selection, grid, library and editor levels, with local Tab navigation. |
| Explorer | Arrows navigate/collapse/expand; `/` searches node and ancestor text; `esc` clears search. Dashboard `ctrl+b` hides/shows Explorer. |
| Review queue | `space` picks up/drops a tree; arrows carry it; `esc` restores order. `alt+left` / `alt+right` move it one place. |
| Log viewer | `/` searches, `n` / `N` find next/previous, `l` cycles all / INFO+ / WARNING+ / ERROR+, End follows the tail. |
| Notifications | `ctrl+n` opens/closes the center; arrows select, `enter` opens, `c` clears session history. |

Footer confirmations accept only their confirmation key, cancellation or quit;
underlying keys, paste and mouse actions are blocked. Ordinary printable
shortcuts yield to text entry. Mouse text selection copies through OSC 52 when
the terminal permits it. Palette choice persists in `outputs/tui/settings.json`;
notifications and draft/navigation state are session memory.

Notifications retain the newest 50 items. The first scan is silent; subsequent
run outcomes and batch transitions create notices. Run destinations include
start/process identity, so an old notice cannot open a different run that reused
the directory name. LOST is provisional and can be replaced by a later recorded
outcome; an unreadable status is waited out. A STOPPED notice claims user
cancellation only when this TUI requested it. Opening a notice preserves the
screen beneath it. Sources: [notifications](xmr4el_tui/services/notifications.py)
and [app](xmr4el_tui/app.py).

## Execution and cancellation

The TUI launches a detached coordinator; **the coordinator owns the runners**.
[batch.py](xmr4el_tui/batch.py) prepares records and starts
`python -m xmr4el_tui.coordinator <batch-directory>` in its own session with
stdin disconnected and stdout/stderr in `coordinator.log`.
[coordinator.py](xmr4el_tui/coordinator.py) launches one tree at a time through
[process.spawn](xmr4el_tui/process.py), using an argument list, the project root
as working directory, a new session/group and file-backed stdout/stderr. The
runner receives its spec on stdin (`-spec -`), a new `-run_dir`, and
`-model_config_sha256`. Existing run/console paths are not overwritten.

`coordinator.lock` admits one coordinator for a batch, and
`outputs/batches/.active.lock` admits one active batch per root. Locks are held
for the coordinator's lifetime and are not inherited by runners. This excludes
other coordinators; it does not reserve hardware against shell-launched jobs.

Before execution, the coordinator checks `batch.json` against the launch hash.
Before each tree it rehashes shared inputs and checks for a stop, including after
hashing. The runner binds its config to captured bytes; a config-hash mismatch
exits **3** before creating a run directory. Training input is rechecked after
loading, and evaluation inputs after evaluating, before writing metrics.
Successful tree classification also requires completed status, readable output
objects, the saved `xmodel.pkl` marker and provenance matching the batch inputs.
The coordinator never loads the pickle or recomputes metrics.

Ordinary tree failures continue to the next tree, without retries. Shared-input
changes, invalid launch plans, stop requests and unrecoverable status-write
errors end the batch and cancel unstarted work; finished outputs remain. A failed
start retains diagnostic status/logs when possible; preparation write failures
clean up the incomplete batch directory.

Batch states are `preparing`, `running`, `stopping`, `completed`,
`completed_with_failures`, `stopped`, `blocked`, `failed` and `lost`. Queue entries
use `pending`, `running`, `succeeded`, `failed`, `cancelled` and `interrupted`.
Unreadable batch status is reported as unreadable, rather than treated as active.

**Stop Batch** creates an idempotent `stop` file. The coordinator sends SIGTERM
to its own unreaped child's process group, escalates to SIGKILL after **30 s**,
cancels remaining trees, and records the outcome. **Cancel Run** in Run Monitor
sends graceful SIGTERM after confirmation; that screen has no force button and
does not cancel the remaining queue. To end the queue, use Stop Batch.

Run stopping requires all of:

- The same runtime, PID, UID and process-start token as the recorded run.
- A live process leading its own group and session (`pid == pgid == sid`).
- A matching `.console/<run-id>.launch.json` naming the canonical run directory
  and process identity. An isolated CLI process alone is insufficient.

`process.stop` rereads status and rechecks identity before signalling; completed,
changed, unverified or shared-group processes are not signalled. A reattached TUI
in the same runtime can use the persistent launch record. Linux runtime identity
combines host boot ID and PID namespace; macOS uses boot time. Hostname alone is
not ownership evidence. See [procinfo.py](xmr4el_tui/procinfo.py).

| Run state | Meaning |
| --- | --- |
| RUNNING | Recorded running and live identity verified; cancellation still needs the ownership/group checks above. |
| UNVERIFIED | Recorded running but runtime or identity cannot be checked; heartbeat age may be displayed. |
| LOST | Recorded running but the original process is gone or its PID has been reused. |
| COMPLETED / FAILED / STOPPED | Runner's final status, including its error/signal when present. |
| UNKNOWN | No readable recognized status. |

The runner touches `heartbeat` every **30 s**. On handled SIGTERM/SIGINT it closes
open telemetry stages and records STOPPED; completed split metrics remain.
Final status is written after events, resource summaries and final log records.
The monitor performs a final read after observing the end. A verified forced
stop can record `stopped`, `error: SIGKILL`, `forced: true` only after the process
is gone, preserving an existing final status. An external SIGKILL/OOM leaves no
such record and normally appears LOST.

If a recorded-active batch's coordinator lock is free, a scan reconciles it to
**lost** (a preparing batch gets a **60 s** grace period). The current tree becomes
interrupted; pending trees stay pending. Nothing is restarted or signalled. An
uncatchably killed coordinator can leave a runner alive; inspect that run before
launching more work. There is no lost-batch resume or automatic survivor cleanup.

## Files and read behavior

Paths are relative to `--root`. The application reads metadata, never model
pickles or corpora to decide Results compatibility. Dataset preflight and batch
input hashing are separate operations described above.

| Location | Contract |
| --- | --- |
| `outputs/runs/<run-id>/spec.json` | Captured spec, including invocation `xmodel_path` (null for training), training path and per-split evaluation/search inputs. |
| `status.json` | Recorded state, start/end/error and process identity; final records retain identity. |
| `model_config.json`, `tree.json` | Captured training configuration; saved tree location, tree ID and whether this run trained it. |
| `metrics_<split>.json` | One evaluation's metrics, search, preprocessing and comparison evidence. |
| `provenance.json`, `git.json` | Input hashes and environment; recorded commit/dirty state. |
| `events.jsonl`, `resources.json`, `heartbeat`, `run.log` | Stage progress, completed-stage resource summaries, heartbeat and readable log. |
| `outputs/runs/.console/<run-id>.log` and `.launch.json` | Captured runner stdout/stderr (log fallback) and persistent launch ownership evidence. |
| `outputs/saved_trees/<tree>/` | Presence of `xmodel.pkl` marks the tree available locally; Description reads `xmodel.json`, never unpickles. |
| `outputs/batches/<batch-id>/batch.json` | Plan written once: dataset, input hashes, git/Python provenance, ordered tree IDs/names, runner specs and config hashes. |
| Batch `configs/<tree-id>.json`, `status.json` | Copied applied configs; atomic mutable progress with the plan hash. Preparation/startup write initial/failure status; the coordinator owns execution updates, with lock-protected lost reconciliation by scans. |
| Batch `stop`, `coordinator.log`, `coordinator.lock` | Stop request, supervisor log and liveness/ownership lock. |
| `outputs/tui/datasets.json`, `settings.json` | Registered local mappings and palette. An unreadable dataset registry is not overwritten by registration. |

[runs.py](xmr4el_tui/runs.py) tolerates missing/corrupt metadata and reports
problems. [ExperimentRepository](xmr4el_tui/services/experiment_repository.py)
owns cached snapshots and selection details. Scans run every **5 s** in a
serialized, coalescing worker; unchanged run metadata is reused by file
signature. Provenance, git and metrics changes invalidate comparison facts.
Explicit rescan bypasses caches. A failed run-directory scan preserves the last
good snapshot and reports the problem. **Scanning can write batch LOST
reconciliation**; browsing is not globally free of writes.

Run Monitor telemetry, live logs and Batch Monitor status poll every **1 s**
while active; run state comes from the shared scan. Logs prefer `run.log`, falling
back to the console log. [LogTail](xmr4el_tui/logs.py) reads at most **8 MiB** per
call, restarting after replacement/truncation; excess earlier bytes are skipped.
The viewer keeps at most **100,000 lines**. Partial live lines wait for completion.
Missing logs are reported, not generated.

## Progress and resources

[EventLog](xmr4el/telemetry/events.py) writes schema-v1 JSON lines with `v`, an
offset-aware timestamp `t`, `stage` and `phase`, plus `split`/`part` where needed.
Phases include start, complete, failed, stopped, tick and cache_hit; end events
carry monotonic `duration_seconds`. Events are structured logging fields, not
parsed prose. A torn final line waits for its newline; malformed complete lines
are skipped. Without events, the pipeline stage list is unknown.

Training stages follow the model pipeline: load data, encode features, build
label embeddings, train hierarchy/matchers, save, then evaluate each enabled
split. Encoder parts distinguish mention/context TF-IDF, transformer and
assembly; hierarchy layer/node starts provide detail, not separate clustering
and matcher stages. Evaluation parts are data, predict, metrics and write.
Transformer ticks count distinct mentions, throttled to five seconds with a final
count/cache-hit event. They do not estimate whole-run progress or ETA.

[ResourceMonitor](xmr4el/telemetry/resources.py) samples at
`-resource_interval` seconds (default **5**, **0** disables it), also at stage
boundaries, and writes summaries by atomic replacement at stage end. Raw samples
are not persisted. Missing values are null and displayed as N/A.

| Resource | Scope |
| --- | --- |
| CPU average | This process's threads plus exited/waited-for children; 100% means one core. |
| RSS average/peak | This process; sampled, with lifetime peak information where applicable. |
| GPU utilization | Whole CUDA device, including other processes; sampled NVML data, unavailable without the optional library/driver. |
| GPU memory peak | This process's PyTorch tensor allocations: CUDA allocator peak, MPS sampled current allocations. Excludes reserved cache, context and non-PyTorch allocations. |

Parts show their parent stage's resources. Event readers retain parsed history;
resource aggregation retains samples for the active stage. Incremental reads and
bounded log display do not make all monitoring memory constant.

## Comparison contract

Only [services/comparison.py](xmr4el_tui/services/comparison.py) `assess` decides
comparability; `allowed` gates individual deltas. `read_facts` gathers evidence
for the selected `(run id, split)` and its tree's training run off the UI thread.
Batch membership, matching paths or equal coverage do not prove comparability.

| Evidence | What it establishes and gates |
| --- | --- |
| Metrics versus spec `test_path`, `train_path`, `split` | Recorded fields must agree with the selected evaluation; conflict is INCOMPATIBLE. |
| Test-file SHA-256 from run provenance | Equal evaluated bytes are required. Missing or different hashes are INCOMPATIBLE. |
| Metrics `rows_total` | Equal recorded loaded-row counts are required. Missing or different counts are INCOMPATIBLE. |
| `eval_protocol` | Equal recorded numeric metric definitions are required for **every delta**. Missing means CAUTION; different means a note and no deltas. Population/string hashes cannot substitute for protocol. |
| `population_sha256` | Equal ordered retained `[loaded row position, gold label]` populations permit in-vocabulary deltas. A mismatch withholds them. Missing hashes require the strict derivation below; otherwise no population-dependent deltas. |
| `strings_sha256` | Equal per-retained-row `[loaded row position, distinct training-label count for the mention string, dictionary answer]` evidence permits string-dependent deltas. Vocabulary, abbreviation expansion and reference input affect this evidence. Missing/different hashes withhold these deltas; when both breakdowns exist but hashes are missing, the assessment is CAUTION. |

SHA-256 evidence must be **64 lowercase hexadecimal characters**; commit IDs
must be 40 or 64. Malformed evidence counts as missing.

When population hashes are unavailable, derivation requires the already matched
test hash and row count, equal tree-training-file hashes, equal recorded
preprocessing (`context_window`, `abbrev_expansion`), and every involved run at
one clean recorded git commit. Training-file evidence comes from the run that
trained each tree, found by `tree_id`; an evaluation's dictionary `train_path`
does not replace it. Missing training records or uncertain evidence prevent
derivation. A known training-file mismatch also withholds population deltas.

After the basic eligibility and protocol gates pass:

| Delta | Additional evidence required |
| --- | --- |
| Acc@1_all, coverage | None; all loaded rows remain the comparison population. |
| Acc@1, MRR, R@k, R@cand | Equal or strictly derived in-vocabulary population. |
| Seen/unseen breakdown rates | Equal string evidence. |
| Hybrid | Both population and string evidence. |
| Hybrid_all | String evidence. |
| Timing, counts and specs | No deltas are shown. Equal hardware/load are not established for timing. |

VERIFIED means no required evidence remains unknown; known protocol/population/
string differences can still withhold specific deltas with notes. CAUTION marks
uncertainty and explains what can still be compared. INCOMPATIBLE evaluations
cannot join the comparison. A withheld numeric delta is `Δ —`; absent/null values
are `N/A`, never zero. Empty string groups have null rates. Fractional-rate
deltas use percentage points; MRR uses its native scale. Summary always uses
Acc@1_all, Acc@1 and coverage, without selecting a winner. Beam, vocabulary and
search differences are shown rather than used as eligibility gates.

[evaluate_tree](xmr4el/eval.py) currently records `EVAL_PROTOCOL = 1`. Changing a
metric formula, denominator, filtering or tie rule requires bumping that
constant. Re-evaluate saved trees to obtain missing evidence; do not relax gates
to subtract older metrics. See [evaluation definitions](PIPELINE.md#evaluation).

## Code ownership and verification

| Modules | Responsibility |
| --- | --- |
| `app.py`, `screens/`, `widgets/`, `messages.py`, `theme.py` | Navigation, draft lifecycle, workers, rendering, notifications and presentation. |
| `runs.py`, `services/experiment_repository.py`, `explorer.py` | Tolerant artifact reads, cached snapshots, grouping and selection details. |
| `draft.py`, `datasets.py`, `components.py`, `review.py` | Session editing, bounded input checks, component metadata, validation, differences and copied launch plans. |
| `batch.py`, `coordinator.py`, `process.py`, `procinfo.py` | Batch persistence/locks, sequential supervision, runner spawning, identity and cancellation. |
| `progress.py`, `logs.py`, `services/comparison.py`, `services/notifications.py` | Structured progress/log reads, comparison evidence and session lifecycle notices. |
| `scripts/run_experiment.py`, `xmr4el/telemetry/` | Scientific execution, provenance, status, events and resource writing. |

The TUI package does not import `xmr4el`; data/process/comparison helpers are
stdlib modules. The runner imports lightweight `procinfo` without importing
Textual. Screen and Results reads use workers; the bounded/synchronous dataset
exceptions and monitoring memory limits are described above.

Isolated TUI checks use temporary roots, synthetic run records and fake runner
processes, with headless Textual Pilot interactions:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -p no:cacheprovider tests/tui/ -q --tb=short
```

Relevant tests cover Dashboard/Builder/Review navigation, `n` after launch,
comparison evidence, process ownership, locks, stop/failure/crash handling and
reconnection. [Metric tests](tests/test_eval.py),
[telemetry tests](tests/test_telemetry.py),
[registry/backend contracts](tests/test_tui_components.py) and
[runner tests](tests/test_run_experiment.py) cover their separate boundaries;
they are not all real execution evidence. Session-specific commands/results
and exclusions are in [the inventory](INVENTORY.md).

The restricted macOS test run failed process-identity checks and active-batch quit
teardown (`ScreenStackError` in a scan notification callback). All 13 process/
identity checks and the isolated quit regression passed when rerun outside the
sandbox, without code changes. Actual interactive shutdown was not checked.

The TUI has no eval-only launch or grouped-TSV builder; use the
[CLI recipes](SCRIPTS.md#tracked-runs). Training resume, retries,
concurrent trees/batches and persistent drafts are not implemented. Git metadata
is not a frozen executable-code snapshot. Hash checks detect ordinary mutation,
not transient write-and-restore attacks; atomic replacement is not a power-loss
durability guarantee. Real biomedical training, model downloads, accelerators,
Linux process control, SSH disconnects, terminal colors and OSC 52 behavior were
not validated by the headless documentation checks.
