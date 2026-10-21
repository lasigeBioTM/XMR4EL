# Results

Benchmark of XMR4EL against SapBERT 1-NN (same mention encoder, no tree) on seven biomedical entity-linking sets, with
PECOS XR-Linear and the scispaCy UMLS linker as external systems. Character TF-IDF 1-NN is a string-match reference row,
not a baseline. The PECOS and scispaCy runners are not part of this repository; their setup is described below.

## Setup

- Config `configs/xmr4el_bc5cdr_config.json` for every set, unchanged; no tuning between dev and test. Search = the config's
  `predict_config` (beam 10, `leaf_candidates` 100, knn beta 10, topk 0).
- One tree per set, trained on its train split + a CTD dictionary (diseases: CTD MEDIC + OMIM; chemicals: CTD chemicals,
  ~180k labels). Test is evaluated on the dev tree, once.
- Rows: mentions with exactly one concept id (multi-id rows are dropped by the converters). acc@1, MRR and R@10 count
  in-vocabulary rows (gold label has training text); `cov x acc` counts every row, out-of-vocabulary rows as misses.
- Mention-string groups, by the training file: `seen, 1` = the exact string is in training under one label, `seen, >1` =
  under several labels (context decides), `unseen` = string not in training. `hybrid` = exact training string -> its
  majority label, otherwise the system.
- 1-NN baselines (`scripts/baselines/nn_baselines.py`): nearest training row of the tree, same rows and vocabulary as
  XMR4EL; exact ties -> lower label index.
- PECOS XR-Linear (libpecos, Docker linux/amd64): trained on XMR4EL's exported features of the 2026-10-13 trees (same rows
  and labels), `-nr_splits 16 -max_leaf_size 100 -threshold 0` (hierarchy 16 -> 256 -> labels on diseases and MedMentions,
  8 -> 128 -> 2048 -> labels on chemicals, 8 -> 128 -> labels on BC5CDR-disease), beam 10, top 100; trained once on the dev
  export, test predicted with that model. PECOS's default weight pruning (0.1) cost 0.075 acc@1 on BC5CDR-disease dev.
- scispaCy (MedMentions only: its knowledge bases are keyed by UMLS CUIs): scispaCy 0.6.2 UMLS `CandidateGenerator` (KB
  2022AB; MedMentions is 2017AA, so retired CUIs get no candidate), k 100, score = max alias similarity, no 0.7 threshold,
  candidates restricted to the tree's labels (8.7 / 8.5 per row; 0.6 / 0.4% of rows get none). Unrestricted, its top-1 is
  outside the label set for 23.9 / 23.7% of rows (dev / test).
- Trees: the five sets' XMR4EL and 1-NN rows are the 2026-10-20 trees (current code). MedMentions st21pv (2026-10-13) and
  BC5CDR-disease (2026-10-09) trees predate the current saved-tree layout and float32 training, so they have no SapBERT
  1-NN row (a rerun needs a retrain).
- Each mention has one gold label, so this is multi-class: precision@k would be recall@k / k and is not reported.
- Same-code GPU retrains move 10-20 rows; smaller gaps are noise.

## Dev and test

dev:

| dataset (labels, rows) | system | acc@1 | cov x acc | MRR | R@10 | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|---|
| NCBI-disease (13,385, 729/757; n 537/17/175) | XMR4EL | 0.925 | 0.890 | 0.946 | 0.974 | 0.968 | 0.941 | 0.789 | 0.929 |
|  | SapBERT 1-NN | 0.894 | 0.861 | 0.932 | 0.984 | 0.972 | 0.176 | 0.726 | 0.914 |
|  | char TF-IDF 1-NN (reference) | 0.868 | 0.836 | 0.911 | 0.973 | 0.972 | 0.176 | 0.617 | 0.888 |
|  | PECOS XR-Linear, our features | 0.894 | 0.861 | 0.916 | 0.947 | 0.946 | 0.882 | 0.737 | 0.916 |
| BioRED disease (13,325, 956/961; n 605/23/328) | XMR4EL | 0.883 | 0.878 | 0.914 | 0.963 | 0.979 | 0.870 | 0.707 | 0.886 |
|  | SapBERT 1-NN | 0.880 | 0.875 | 0.912 | 0.970 | 0.979 | 0.826 | 0.701 | 0.883 |
|  | char TF-IDF 1-NN (reference) | 0.847 | 0.843 | 0.889 | 0.957 | 0.979 | 0.826 | 0.607 | 0.850 |
|  | PECOS XR-Linear, our features | 0.844 | 0.840 | 0.879 | 0.935 | 0.931 | 0.739 | 0.692 | 0.877 |
| BC5CDR-chemical (179,709, 5181/5307; n 4465/68/648) | XMR4EL | 0.946 | 0.924 | 0.954 | 0.965 | 0.978 | 0.632 | 0.764 | 0.967 |
|  | SapBERT 1-NN | 0.968 | 0.945 | 0.975 | 0.986 | 0.996 | 0.559 | 0.821 | 0.974 |
|  | char TF-IDF 1-NN (reference) | 0.961 | 0.939 | 0.970 | 0.983 | 0.996 | 0.559 | 0.765 | 0.967 |
|  | PECOS XR-Linear, our features | 0.827 | 0.807 | 0.859 | 0.911 | 0.853 | 0.559 | 0.677 | 0.956 |
| BioRED chemical (179,702, 779/809; n 629/22/128) | XMR4EL | 0.870 | 0.838 | 0.896 | 0.942 | 0.968 | 0.955 | 0.375 | 0.895 |
|  | SapBERT 1-NN | 0.920 | 0.886 | 0.937 | 0.969 | 0.998 | 0.864 | 0.547 | 0.920 |
|  | char TF-IDF 1-NN (reference) | 0.929 | 0.895 | 0.937 | 0.951 | 0.998 | 0.864 | 0.602 | 0.929 |
|  | PECOS XR-Linear, our features | 0.739 | 0.712 | 0.792 | 0.875 | 0.822 | 0.409 | 0.391 | 0.897 |
| NLM-Chem (179,696, 4907/4945; n 3168/173/1566) | XMR4EL | 0.875 | 0.869 | 0.892 | 0.916 | 0.963 | 0.636 | 0.723 | 0.883 |
|  | SapBERT 1-NN | 0.845 | 0.838 | 0.893 | 0.959 | 0.975 | 0.202 | 0.653 | 0.857 |
|  | char TF-IDF 1-NN (reference) | 0.872 | 0.865 | 0.898 | 0.941 | 0.975 | 0.202 | 0.738 | 0.884 |
|  | PECOS XR-Linear, our features | 0.689 | 0.684 | 0.751 | 0.841 | 0.807 | 0.526 | 0.471 | 0.795 |
| MedMentions st21pv (18,520, 31564/40884; n 11303/11049/9212) | XMR4EL | 0.741 | 0.572 | 0.806 | 0.908 | 0.932 | 0.763 | 0.480 | 0.743 |
|  | char TF-IDF 1-NN (reference) | 0.593 | 0.458 | 0.704 | 0.884 | 0.933 | 0.412 | 0.392 | 0.718 |
|  | PECOS XR-Linear, our features | 0.688 | 0.531 | 0.756 | 0.860 | 0.861 | 0.767 | 0.381 | 0.715 |
|  | scispaCy UMLS linker (restricted) | 0.683 | 0.528 | 0.749 | 0.852 | 0.875 | 0.633 | 0.509 | 0.752 |
| BC5CDR-disease (11,744, 4305/4306; n 3140/212/953) | XMR4EL | 0.900 | 0.900 | 0.931 | 0.973 | 0.978 | 0.778 | 0.669 | 0.904 |
|  | char TF-IDF 1-NN (reference) | 0.874 | 0.874 | 0.908 | 0.957 | 0.978 | 0.618 | 0.589 | 0.886 |
|  | PECOS XR-Linear, our features | 0.852 | 0.851 | 0.889 | 0.944 | 0.934 | 0.769 | 0.597 | 0.887 |

test:

| dataset (labels, rows) | system | acc@1 | cov x acc | MRR | R@10 | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|---|
| NCBI-disease (13,385, 923/945; n 633/33/257) | XMR4EL | 0.896 | 0.875 | 0.923 | 0.964 | 0.978 | 0.818 | 0.704 | 0.900 |
|  | SapBERT 1-NN | 0.874 | 0.854 | 0.910 | 0.966 | 0.976 | 0.394 | 0.685 | 0.894 |
|  | char TF-IDF 1-NN (reference) | 0.826 | 0.806 | 0.874 | 0.946 | 0.976 | 0.394 | 0.510 | 0.845 |
|  | PECOS XR-Linear, our features | 0.829 | 0.810 | 0.875 | 0.939 | 0.913 | 0.879 | 0.615 | 0.874 |
| BioRED disease (13,325, 904/904; n 549/18/337) | XMR4EL | 0.882 | 0.882 | 0.914 | 0.967 | 0.971 | 0.444 | 0.760 | 0.892 |
|  | SapBERT 1-NN | 0.876 | 0.876 | 0.910 | 0.969 | 0.973 | 0.500 | 0.739 | 0.885 |
|  | char TF-IDF 1-NN (reference) | 0.840 | 0.840 | 0.878 | 0.961 | 0.973 | 0.500 | 0.641 | 0.848 |
|  | PECOS XR-Linear, our features | 0.829 | 0.829 | 0.868 | 0.935 | 0.920 | 0.556 | 0.694 | 0.866 |
| BC5CDR-chemical (179,709, 5167/5352; n 4330/111/726) | XMR4EL | 0.930 | 0.898 | 0.943 | 0.960 | 0.966 | 0.604 | 0.770 | 0.957 |
|  | SapBERT 1-NN | 0.952 | 0.919 | 0.963 | 0.983 | 0.989 | 0.649 | 0.774 | 0.957 |
|  | char TF-IDF 1-NN (reference) | 0.961 | 0.928 | 0.969 | 0.983 | 0.989 | 0.649 | 0.837 | 0.966 |
|  | PECOS XR-Linear, our features | 0.811 | 0.783 | 0.846 | 0.899 | 0.848 | 0.514 | 0.640 | 0.938 |
| BioRED chemical (179,702, 728/741; n 559/27/142) | XMR4EL | 0.912 | 0.896 | 0.928 | 0.944 | 0.966 | 0.704 | 0.739 | 0.945 |
|  | SapBERT 1-NN | 0.920 | 0.904 | 0.950 | 0.989 | 0.996 | 0.185 | 0.761 | 0.926 |
|  | char TF-IDF 1-NN (reference) | 0.919 | 0.903 | 0.947 | 0.979 | 0.996 | 0.185 | 0.754 | 0.924 |
|  | PECOS XR-Linear, our features | 0.786 | 0.772 | 0.836 | 0.902 | 0.839 | 0.296 | 0.669 | 0.908 |
| NLM-Chem (179,696, 11029/11200; n 6751/941/3337) | XMR4EL | 0.842 | 0.829 | 0.871 | 0.909 | 0.940 | 0.758 | 0.668 | 0.876 |
|  | SapBERT 1-NN | 0.853 | 0.840 | 0.896 | 0.949 | 0.980 | 0.438 | 0.714 | 0.892 |
|  | char TF-IDF 1-NN (reference) | 0.843 | 0.830 | 0.880 | 0.931 | 0.980 | 0.438 | 0.678 | 0.881 |
|  | PECOS XR-Linear, our features | 0.723 | 0.712 | 0.772 | 0.843 | 0.805 | 0.762 | 0.547 | 0.835 |
| MedMentions st21pv (18,520, 31085/40157; n 11290/10768/9027) | XMR4EL | 0.744 | 0.576 | 0.808 | 0.912 | 0.933 | 0.769 | 0.479 | 0.747 |
|  | char TF-IDF 1-NN (reference) | 0.594 | 0.460 | 0.702 | 0.877 | 0.932 | 0.410 | 0.390 | 0.721 |
|  | PECOS XR-Linear, our features | 0.694 | 0.537 | 0.760 | 0.864 | 0.863 | 0.777 | 0.383 | 0.719 |
|  | scispaCy UMLS linker (restricted) | 0.681 | 0.527 | 0.747 | 0.850 | 0.866 | 0.642 | 0.495 | 0.751 |
| BC5CDR-disease (11,744, 4399/4410; n 3284/174/941) | XMR4EL | 0.922 | 0.920 | 0.943 | 0.976 | 0.985 | 0.770 | 0.732 | 0.925 |
|  | char TF-IDF 1-NN (reference) | 0.892 | 0.889 | 0.921 | 0.968 | 0.984 | 0.603 | 0.621 | 0.902 |
|  | PECOS XR-Linear, our features | 0.865 | 0.863 | 0.901 | 0.955 | 0.931 | 0.718 | 0.661 | 0.910 |

Rows, XMR4EL vs SapBERT 1-NN (dev / test): NCBI-disease +22 / +20, BioRED disease +3 / +5, BC5CDR-chemical -114 / -111,
BioRED chemical -39 / -6, NLM-Chem +150 / -118.

- Diseases (13k labels): XMR4EL is ahead of SapBERT 1-NN on dev and test. The gain is mentions whose string appears in
  training under several labels (context decides them) and unseen strings; seen-once strings are level.
- Chemicals (180k labels): XMR4EL is behind SapBERT 1-NN except NLM-Chem dev. It still wins `seen, >1` on five of six splits but
  loses seen-once strings, and R@cand is 0.92-0.97 (diseases 0.98-0.99): part of the loss is routing (gold not among the
  candidates). See `docs/future_work.md`.
- PECOS is below XMR4EL on every set and split (vs the trees in the table): diseases -3.1 / -3.9 pt dev (NCBI / BioRED),
  -6.7 / -5.3 test; chemicals -11.9 to -18.6 pt; MedMentions -5.3 / -5.0; BC5CDR-disease -4.8 / -5.7. On chemicals it
  misses exact training names (seen once 0.80-0.85 vs 0.94-0.98). Same rows and feature design, but PECOS has no knn
  re-scoring.
- MedMentions st21pv (18.5k labels, no dictionary, 77% of rows in vocabulary): 64% of the rows are `seen, >1` or unseen
  strings, so context decides more here than on any other set. XMR4EL 0.741 / 0.744 vs PECOS 0.688 / 0.694 and scispaCy
  0.683 / 0.681; scispaCy is better on unseen strings (0.509 / 0.495 vs 0.480 / 0.479) and its hybrid is above XMR4EL's.
- Context: zeroing the query's context block drops BC5CDR-chemical dev `seen, >1` from 0.632 to 0.015 and NCBI dev from
  0.941 to 0.235.
- Cost (one GPU server, peak RSS train + dev): chemicals 22 GB, ~13 min hierarchy + ~8 min encoding (SapBERT embeddings
  cached between runs); diseases 5 GB, ~2 min. 1-NN baselines 15 GB on chemicals.

## Datasets

Corpora and CTD are not in the repository. Sources:

- NCBI-disease: PubTator release (`NCBI{train,develop,test}set_corpus.zip`, ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE).
- BioRED: `BioRED.zip` (ftp.ncbi.nlm.nih.gov/pub/lu/BioRED), PubTator files `{Train,Dev,Test}.PubTator`.
- BC5CDR: BioCreative V CDR corpus `CDR_Data.zip` (`CDR.Corpus.v010516`).
- NLM-Chem: `ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip`.
- CTD: `ctdbase.org/reports/CTD_diseases.tsv.gz`, `ctdbase.org/reports/CTD_chemicals.tsv.gz`.

Conversion into `datasets/` (the paths the specs in `configs/experiments/` expect):

```bash
P=.venv/bin/python; T=scripts/to_pubtator.py
N=datasets/ncbi_disease; for s in train:trainset dev:developset test:testset; do $P $T filter -input $N/NCBI_pubtator/NCBI${s#*:}_corpus.txt -out $N/${s%%:*}.pubtator -types SpecificDisease DiseaseClass Modifier CompositeMention; done
B=datasets/BioRED; mkdir -p $B/disease $B/chemical; for s in train:Train dev:Dev test:Test; do $P $T filter -input $B/BioRED/${s#*:}.PubTator -out $B/disease/${s%%:*}.pubtator -types DiseaseOrPhenotypicFeature; $P $T filter -input $B/BioRED/${s#*:}.PubTator -out $B/chemical/${s%%:*}.pubtator -types ChemicalEntity; done
C=datasets/BC5CDR; mkdir -p $C/chemical; for s in train:TrainingSet dev:DevelopmentSet test:TestSet; do $P $T filter -input $C/CDR_Data/CDR.Corpus.v010516/CDR_${s#*:}.PubTator.txt -out $C/chemical/${s%%:*}.pubtator -types Chemical; done
L=datasets/nlm_chem; for s in train dev test; do $P $T bioc -input_dir $L/FINAL_v1/ALL -pmcids $L/FINAL_v1/pmcids_$s.txt -out $L/$s.pubtator; done
$P scripts/dict_to_pubtator.py -ctd_path datasets/CTD/CTD_diseases.tsv.gz -out datasets/CTD/ctd_disease_omim.pubtator -type Disease -omim
$P scripts/dict_to_pubtator.py -ctd_path datasets/CTD/CTD_chemicals.tsv.gz -out datasets/CTD/ctd_chemical.pubtator -type Chemical
cat $N/train.pubtator datasets/CTD/ctd_disease_omim.pubtator > $N/train_plus_ctd.pubtator
cat $B/disease/train.pubtator datasets/CTD/ctd_disease_omim.pubtator > $B/disease/train_plus_ctd.pubtator
for d in $B/chemical $C/chemical $L; do cat $d/train.pubtator datasets/CTD/ctd_chemical.pubtator > $d/train_plus_ctd.pubtator; done
```

Two more sets: BC5CDR-disease (`configs/experiments/bc5cdr.json`; `$P $T filter ... -types Disease`
into `datasets/BC5CDR/disease/`, train + `ctd_disease.pubtator` from `dict_to_pubtator.py` without `-omim`) and
MedMentions st21pv (github.com/chanzuckerberg/MedMentions, `configs/experiments/medmentions_st21pv.json`; `scripts/split_pubtator.py` with the release's
`corpus_pubtator_pmids_{trng,dev,test}.txt`, no dictionary).

## Reproduce

```bash
# one set: train + dev, then test on the same tree (eval-only)
.venv/bin/python scripts/run_experiment.py -spec configs/experiments/bc5cdr_chemical.json
.venv/bin/python scripts/run_experiment.py -spec configs/experiments/bc5cdr_chemical_test.json -xmodel_path outputs/saved_trees/<tree>

# 1-NN baselines on the tree's training rows
.venv/bin/python scripts/baselines/nn_baselines.py -xmodel_path outputs/saved_trees/<tree> \
  -test_path datasets/BC5CDR/chemical/dev.pubtator -train_path datasets/BC5CDR/chemical/train_plus_ctd.pubtator -out nn.json
```

Run directories (`outputs/runs/`) record the spec, config, git state, input hashes and metrics.
