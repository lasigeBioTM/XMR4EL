import argparse
import hashlib
import json
import os
from collections import defaultdict

"""
Usage:
  python prepare_pubtator_splits.py \
    --input MedMentions/st21pv/corpus_pubtator.txt \
    --outdir datasets/medmentions/st21pv/ \
    --train_pmids train_pmids.txt \
    --dev_pmids dev_pmids.txt \
    --test_pmids test_pmids.txt \
    --emit_jsonl
"""

def deterministic_split(pmid: str, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1) -> str:
    h = hashlib.md5(pmid.encode("utf8")).hexdigest()
    val = int(h[:8], 16) / float(0xFFFFFFFF)
    if val < train_ratio:
        return "train"
    elif val < train_ratio + dev_ratio:
        return "dev"
    else:
        return "test"

def parse_pubtator(path: str) -> dict:
    pmid_blocks = defaultdict(list)
    current_pmid = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "|t|" in line or "|a|" in line:
                pmid = line.split("|", 1)[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif "\t" in line:
                pmid = line.split("\t")[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif current_pmid:
                pmid_blocks[current_pmid].append(line)
    return pmid_blocks

def write_pubtator_file(pmid_list, pmid_blocks, out_path):
    with open(out_path, "w", encoding="utf-8") as fw:
        for pmid in pmid_list:
            if pmid in pmid_blocks:
                for line in pmid_blocks[pmid]:
                    fw.write(line + "\n")
                fw.write("\n")

def emit_jsonl_for_split(pmid_list, pmid_blocks, out_jsonl_path):
    out = []
    for pmid in pmid_list:
        lines = pmid_blocks.get(pmid, [])
        title = ""
        abstract = ""
        for l in lines:
            if "|t|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    title = parts[2]
            elif "|a|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    abstract = parts[2]
        context = " ".join(x for x in [title.strip(), abstract.strip()] if x)
        for l in lines:
            if "\t" in l:
                parts = l.split("\t")
                if len(parts) >= 6:
                    pmid_p, start, end, mention, sem_types, cui = parts[:6]
                    entry = {
                        "pmid": pmid_p,
                        "mention": mention,
                        "start": int(start),
                        "end": int(end),
                        "cui": cui,
                        "sem_types": sem_types.split(",") if sem_types else [],
                        "context": context
                    }
                    out.append(entry)
    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for e in out:
            fw.write(json.dumps(e, ensure_ascii=False) + "\n")

def load_pmids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full PubTator corpus file")
    parser.add_argument("--outdir", required=True, help="Output directory to write splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--train_pmids", help="Optional file with official train PMIDs")
    parser.add_argument("--dev_pmids", help="Optional file with official dev PMIDs")
    parser.add_argument("--test_pmids", help="Optional file with official test PMIDs")
    parser.add_argument("--emit_jsonl", action="store_true", help="Also emit JSONL for each split")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pmid_blocks = parse_pubtator(args.input)
    pmids = sorted(pmid_blocks.keys())

    splits = {"train": [], "dev": [], "test": []}

    if args.train_pmids and args.dev_pmids and args.test_pmids:
        # Use official PMID splits
        splits["train"] = load_pmids(args.train_pmids)
        splits["dev"] = load_pmids(args.dev_pmids)
        splits["test"] = load_pmids(args.test_pmids)
    else:
        # Deterministic split
        for pmid in pmids:
            s = deterministic_split(pmid, args.train_ratio, args.dev_ratio, args.test_ratio)
            splits[s].append(pmid)

    # Write PubTator files
    write_pubtator_file(splits["train"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_train.txt"))
    write_pubtator_file(splits["dev"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_dev.txt"))
    write_pubtator_file(splits["test"], pmid_blocks, os.path.join(args.outdir, "corpus_pubtator_test.txt"))

    # Optional JSONL
    if args.emit_jsonl:
        emit_jsonl_for_split(splits["train"], pmid_blocks, os.path.join(args.outdir, "train.jsonl"))
        emit_jsonl_for_split(splits["dev"], pmid_blocks, os.path.join(args.outdir, "dev.jsonl"))
        emit_jsonl_for_split(splits["test"], pmid_blocks, os.path.join(args.outdir, "test.jsonl"))

    print("Done. Split sizes: train=%d dev=%d test=%d" %
          (len(splits["train"]), len(splits["dev"]), len(splits["test"])))

if __name__ == "__main__":
    main()