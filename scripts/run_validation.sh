#!/usr/bin/env bash
# LOCAL TEST ARTIFACT (benchmark validation, BENCHMARK.md#reproduction-recipe): not part of the release.
# Current-code check of two Oct 20 benchmark sets (NCBI-disease, BC5CDR-chemical), configs/xmr4el_bc5cdr_config.json
# unchanged: per set XMR4EL train + dev, test once on that tree, knn_beta 0 eval-only (dev + test, pre-registered ablation),
# SapBERT / TF-IDF 1-NN and PECOS XR-Linear on the tree's exported features (trained once on the dev export, test
# predicted with that model, as run_g2_pecos.sh). No setting is chosen from these results.
# Runs on the host (images xmr4el + xmr4el-pecos). Every container: CPUs $CPUSET only, $MEM RAM without swap, lower
# CPU weight (--cpu-shares) and niceness $NICE; BLAS/OpenMP/torch threads = the CPU count.
# Writes outputs/validation/ (env.txt, runs.txt, peak_rss.tsv, specs/, <set>_<split>_*.log, *_nn.json), run dirs
# outputs/runs/<timestamp>_val_<set>[_test|_knn0], exports + PECOS models outputs/pecos/validation/, bundle
# outputs/validation.tar.gz. Stops at the first failure (an OOM kill shows as exit 137).
# Usage: nohup bash scripts/run_validation.sh > outputs/validation_run.log 2>&1 &
set -euo pipefail
CPUSET=${CPUSET:-0-15}; NCPU=${NCPU:-16}; MEM=${MEM:-40g}; NICE=${NICE:-10}; SHARES=${SHARES:-512}; GPUS=${GPUS:-all}
CONFIG=configs/xmr4el_bc5cdr_config.json
CONFIG_SHA=8d1052315900b7f8d2841a5b0b022c4007b13901f5548b85ff429462aea26bcd
V=outputs/validation; E=outputs/pecos/validation

test -z "$(git status --porcelain --untracked-files=no)" || { echo "tracked files modified: commit or stash first"; exit 1; }
for d in "$V" "$E"; do [ ! -e "$d" ] || { echo "$d exists: move it away (runs are never overwritten)"; exit 1; }; done
# Inputs = the Oct 20 benchmark files (server + local hashes, 2026-11-02); the runner re-hashes what it reads
sha256sum -c - <<'EOF'
8d1052315900b7f8d2841a5b0b022c4007b13901f5548b85ff429462aea26bcd  configs/xmr4el_bc5cdr_config.json
1ac8abe966e603104a3f418f8832d8b368a5524619230cca25f68a5aafcb9fb7  datasets/ncbi_disease/train_plus_ctd.pubtator
dc7b2df0c24ba6e3d0d687b444fd389a2b227f28eaeea92811af59755419699e  datasets/ncbi_disease/dev.pubtator
f7149229ec17a5dfbfc1e006dcc0402c3760d7f365b46ec8e6e0abaae3a40392  datasets/ncbi_disease/test.pubtator
8e322df8db16ad47417f0c912c0a788704ea9bf1933d300e0c4fe2da3b10fbd3  datasets/BC5CDR/chemical/train_plus_ctd.pubtator
c59dbbb880244e6006bf7fb550cc1d95d7d73767184410b5b1562542a2370483  datasets/BC5CDR/chemical/dev.pubtator
c5666488b9b3aa4ff950ee39a0488fd07cf796e4e7307323d652f2371f05cb6a  datasets/BC5CDR/chemical/test.pubtator
EOF
mkdir -p "$V/specs" "$E" "$HOME/.cache/huggingface"
docker build -q -f xmr4el.dockerfile -t xmr4el . && docker build -q -f pecos.dockerfile -t xmr4el-pecos .

THREADS=(-e OMP_NUM_THREADS="$NCPU" -e MKL_NUM_THREADS="$NCPU" -e OPENBLAS_NUM_THREADS="$NCPU" -e NUMEXPR_NUM_THREADS="$NCPU")
LIMITS=(--cpuset-cpus "$CPUSET" --memory "$MEM" --memory-swap "$MEM" --cpu-shares "$SHARES" "${THREADS[@]}")
U=(--rm --init --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app "${LIMITS[@]}")
X=(docker run "${U[@]}" --gpus "$GPUS" --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface"
   -e HF_HOME=/cache/huggingface xmr4el nice -n "$NICE" python3)
P=(docker run "${U[@]}" xmr4el-pecos nice -n "$NICE" python3)
PEAK='import resource, subprocess, sys
r = subprocess.run(sys.argv[3:]).returncode
gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
open(sys.argv[1], "a").write(f"{sys.argv[2]}\t{gb:.1f}\n"); sys.exit(r)'
run_x() { local label=$1; shift; "${X[@]}" -c "$PEAK" "$V/peak_rss.tsv" "$label" python3 "$@"; }
run_p() { local label=$1; shift; "${P[@]}" -c "$PEAK" "$V/peak_rss.tsv" "$label" python3 "$@"; }

{ echo "commit $(git rev-parse HEAD)"; echo "date $(date -Is)"; echo "host $(hostname) nproc $(nproc)"
  echo "limits cpuset $CPUSET threads $NCPU memory $MEM (no swap) cpu-shares $SHARES nice $NICE gpus $GPUS"
  nvidia-smi -L 2>/dev/null || true
  docker image inspect -f '{{.Id}} {{index .RepoTags 0}}' xmr4el xmr4el-pecos
  docker run --rm xmr4el-pecos python3 -c 'import importlib.metadata as m; print("libpecos", m.version("libpecos"))'; } > "$V/env.txt"
cat "$V/env.txt"

RUNS=()
for S in ncbi_disease:datasets/ncbi_disease bc5cdr_chemical:datasets/BC5CDR/chemical; do
  N=${S%%:*}; D=${S#*:}; TR=$D/train_plus_ctd.pubtator; TS=$(date +%Y-%m-%d_%H-%M-%S)
  R=outputs/runs/${TS}_val_$N
  run_x "$N train+dev" scripts/run_experiment.py -spec configs/experiments/$N.json -run_dir "$R" \
    -model_config_sha256 "$CONFIG_SHA" 2>&1 | tee "$V/${N}_dev.log"
  T=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["xmodel_path"])' "$R/tree.json")
  run_x "$N test" scripts/run_experiment.py -spec configs/experiments/${N}_test.json -xmodel_path "$T" \
    -run_dir "${R}_test" 2>&1 | tee "$V/${N}_test.log"
  python3 -c 'import json, sys
n, d, tr = sys.argv[1:4]
json.dump({"name": f"{n}_knn0", "evals": [{"split": f"{s}_knn0", "test_path": f"{d}/{s}.pubtator", "train_path": tr,
           "knn_beta": 0.0} for s in ("dev", "test")]}, open(sys.argv[4], "w"), indent=2)' "$N" "$D" "$TR" "$V/specs/${N}_knn0.json"
  run_x "$N knn0" scripts/run_experiment.py -spec "$V/specs/${N}_knn0.json" -xmodel_path "$T" -run_dir "${R}_knn0" \
    2>&1 | tee "$V/${N}_knn0.log"
  RUNS+=("$R" "${R}_test" "${R}_knn0")
  echo "$N train=$R test=${R}_test knn0=${R}_knn0 tree=$T" | tee -a "$V/runs.txt"
  for sp in dev test; do
    run_x "$N nn $sp" scripts/baselines/nn_baselines.py -xmodel_path "$T" -test_path "$D/$sp.pubtator" -train_path "$TR" \
      -out "$V/${N}_${sp}_nn.json" 2>&1 | tee "$V/${N}_${sp}_nn.log"
    run_x "$N export $sp" scripts/baselines/pecos_compare.py export -xmodel_path "$T" -train_path "$TR" \
      -test_path "$D/$sp.pubtator" -out "$E/${N}_$sp" 2>&1 | tee "$V/${N}_${sp}_export.log"
  done
  rm -f "$E/${N}_test/X_trn.npz" "$E/${N}_test/Y_trn.npz"  # test is predicted with the dev model: no copy needed
  run_p "$N pecos train+dev" scripts/baselines/pecos_run.py -data "$E/${N}_dev" -features ours -nr_splits 16 \
    -max_leaf_size 100 -beams 10 -topk 100 -threshold 0 2>&1 | tee "$V/${N}_dev_pecos.log"
  run_p "$N pecos test" scripts/baselines/pecos_run.py -data "$E/${N}_test" -model "$E/${N}_dev/model_ours" \
    -features ours -beams 10 -topk 100 2>&1 | tee "$V/${N}_test_pecos.log"
  for sp in dev test; do
    run_x "$N score $sp" scripts/baselines/pecos_compare.py score -out "$E/${N}_$sp" 2>&1 | tee "$V/${N}_${sp}_score.log"
  done
done
tar czf outputs/validation.tar.gz "$V" "${RUNS[@]}"
echo "done: outputs/validation.tar.gz"
