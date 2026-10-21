# Future work and where to pick up

State at release (2026-10-20). Numbers are from the five benchmark trees in `docs/results.md`
(`configs/xmr4el_bc5cdr_config.json`, train + CTD dictionary, beam 10, leaf candidates 100, knn beta 10).
A "row" is one eval mention with a gold ID in the tree's vocabulary; "+N rows" = N more top-1 hits.
Same-code GPU retrains move 10-20 rows, so smaller gaps are noise.

## Where things stand

| set (labels) | XMR4EL acc@1 dev / test | vs SapBERT 1-NN, rows dev / test |
|---|---|---|
| NCBI-disease (13k) | 0.925 / 0.896 | +22 / +20 |
| BioRED disease (13k) | 0.883 / 0.882 | +3 / +5 |
| BC5CDR-chemical (180k) | 0.946 / 0.930 | -114 / -111 |
| BioRED chemical (180k) | 0.870 / 0.912 | -39 / -6 |
| NLM-Chem (180k) | 0.875 / 0.842 | +150 / -118 |

- Diseases: XMR4EL is ahead of SapBERT 1-NN (same mention encoder, no tree). The gain is mentions whose string appears in
  training under several labels (context decides them) and unseen strings.
- Chemicals (180k labels): XMR4EL is behind SapBERT 1-NN except NLM-Chem dev, and the 1-NN is faster and smaller (GPU brute force
  ~0.17 ms/query and a 1.5 GB index vs ~6 ms/query and a 10 GB tree).
- Context is what resolves shared strings: zeroing the query's context block drops BC5CDR-chemical dev seen >1 from
  0.632 to 0.015 and NCBI dev from 0.941 to 0.235.

## Open problems, in priority order

1. **Exact training names lost on the chemical sets.** Most of the chemical gap is rows whose string appears in
   training under one label, which the 1-NN almost always gets right. On BC5CDR-chemical dev those errors split into
   gold cluster dropped at the root (29), gold cut by the leaf's top 100 (54) and gold ranked below another
   candidate (24) (2026-10-13 trees). Measured: `leaf_candidates` 200 / 500 recovers +25 / +37 rows at +21% / +84% search time
   (still -86 at 500); beam 20 recovers +34. Untested, cheapest next step: add each query's top mention-kNN labels
   to the candidate set before the knn re-scoring in `XModel.predict_encoded` (`xmr4el/xmodel.py`). The knn index
   already covers every label (`xmr4el/learning/scoring.py`: `knn_index`, `label_max_cos`).
2. **Root routing at 180k labels.** The root has 128 clusters; its top-1 cluster accuracy is ~0.77 on the chemical
   sets vs 0.91 on NCBI, and beam 10 hides it only for seen strings. NLM-Chem dev lost 245 of 1566 unseen strings
   at the root (2026-10-13 tree). Options: more root clusters or a deeper tree (`configs/*.json` clustering settings), a root
   trained with routed negatives (see `docs/pipeline.md`, Matcher row).
3. **Empty context on dictionary rows.** CTD rows have no context (96% / 99% of NLM-Chem / BC5CDR-chemical training rows), so the
   context block is zero for them. On NLM-Chem dev, unseen strings score better with the query context removed
   (0.723 -> 0.747), which points at this mismatch. Hypothesis, not tested: route or score dictionary-heavy
   labels without the context block, or down-weight it per row.
4. **Reranker on the benchmark sets.** A cross-encoder over the tree's top 10 (`xmr4el/rerank.py`,
   `scripts/train_reranker.py`) gave +23 rows on BC5CDR-disease test when gated to non-trivial rows; it has not been
   run on the five benchmark sets.
5. **Smaller items.** Ties at the leaf `argpartition` cut are broken arbitrarily (could move tied rows; check with an
   eval-only run); internal nodes route on matcher cluster scores only (no fusion with other scores at predict
   time); multi-ID mentions are dropped by the converters; the literature protocol is not matched (training here is
   train + CTD only, multi-ID rows dropped), so numbers are not directly comparable to published SapBERT/BioSyn results.

## How to pick up

```bash
uv sync --locked
.venv/bin/python -m pytest -q && uvx ruff check xmr4el scripts tests && .venv/bin/python scripts/evaluate.py -selfcheck
```

- Data: corpora and CTD are not in the repo; `docs/results.md` lists the sources and the conversion commands into
  `datasets/`.
- One set, train + dev: `python scripts/run_experiment.py -spec configs/experiments/bc5cdr_chemical.json`.
  Test on the same tree, eval-only: `-spec configs/experiments/bc5cdr_chemical_test.json -xmodel_path <tree>`.
  Run dirs record the spec, config, git state, input hashes and metrics (`outputs/runs/`).
- Baselines on the same rows: `python scripts/baselines/nn_baselines.py -xmodel_path <tree> -test_path <dev> -train_path <train> -out nn.json`.
- Cost on one GPU server: chemical sets ~22 GB peak RAM, ~13 min hierarchy training plus ~8 min encoding
  (SapBERT embeddings are cached); disease sets ~5 GB and ~2 min.
- Search knobs (`predict_config` in the config, or `evaluate.py` flags): `beam_size`, `leaf_candidates`, `knn_beta`,
  `topk`. Change one factor at a time, choose on dev, run test once, and report rows moved.
- Code map: reading `xmr4el/data/readers.py`; features `xmr4el/features/encoder.py`; hierarchy and beam search
  `xmr4el/hierarchy/tree.py`; matchers `xmr4el/learning/`; public API and knn re-scoring `xmr4el/xmodel.py`;
  metrics `xmr4el/eval.py`. Label index `j` = column `j` of `Y` = row `j` of the label embeddings =
  `XModel.initial_labels[j]` (sorted); map indices to names only through `initial_labels`.
