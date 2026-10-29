"""Training images for the draft model, disjoint from everything the judge is evaluated or timed on.

    python experiments/speculator/split.py --pool data/items/test_stock.jsonl \
        --exclude data/items/sol_subset.jsonl data/items/quality_ood.jsonl data/items/throughput_mix.jsonl \
                  experiments/audit/results/queue.jsonl \
        --prompts-per-group 200 --out data/items/speculator_train.jsonl

An item is excluded if an excluded set has the same prompt (either image of it), the same prompt text or the
same checklist: the judge's first step depends only on the checklist, so a shared checklist would put evaluation
answers into the training data. The same number of prompts is drawn from each benchmark, both images of each.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def prompt_id(row: dict) -> str:
    return row["id"].rsplit("__", 1)[0]   # <benchmark>_<n>__original / __rewrite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--prompts-per-group", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    excluded = [row for path in args.exclude for row in read(path)]
    taken = ({prompt_id(r) for r in excluded}, {r["prompt"] for r in excluded},
             {json.dumps(r["requirements"]) for r in excluded})
    by_group = defaultdict(lambda: defaultdict(list))
    for row in read(args.pool):
        if prompt_id(row) in taken[0] or row["prompt"] in taken[1] or json.dumps(row["requirements"]) in taken[2]:
            continue
        by_group[row["group"]][prompt_id(row)].append(row)
    chosen = []
    for group in sorted(by_group):
        prompts = sorted(by_group[group])
        for pid in sorted(random.Random(args.seed).sample(prompts, min(args.prompts_per_group, len(prompts)))):
            chosen += sorted(by_group[group][pid], key=lambda r: r["variant"])
    with open(args.out, "w") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen)
    counts = {g: sum(r["group"] == g for r in chosen) for g in sorted(by_group)}
    print(f"{len(chosen)} images from {sum(len(v) for v in by_group.values())} eligible prompts: {counts}")


if __name__ == "__main__":
    main()
