"""Build a blind audit queue: images to grade by hand, with no judge verdicts attached.

Two samples, merged and shuffled so the auditor cannot tell them apart:
  random        n images per dataset, chosen with a fixed seed before any judge output is read
  disagreement  images where two judges disagree on at least one requirement (only ids are taken from the
                judge outputs; who said what is not written anywhere in the queue)

    python experiments/audit/sample.py --data items_a.jsonl items_b.jsonl --random 75 \
        --disagree results/items_a results/items_b --judges fast quality --n-disagree 50 --out audit/queue.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fastqwenjudge import metrics
from fastqwenjudge.data import load_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--random", type=int, default=75, help="random images per dataset")
    p.add_argument("--disagree", nargs="*", default=[], help="results dirs, one per dataset, same order as --data")
    p.add_argument("--judges", nargs=2, default=["fast", "quality"])
    p.add_argument("--n-disagree", type=int, default=50, help="disagreement images per dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rng = random.Random(args.seed)
    chosen, origin = {}, {}
    for k, path in enumerate(args.data):
        items = load_items(path)
        for i in rng.sample(items, args.random):
            chosen[i.id] = i
            origin.setdefault(i.id, []).append("random")
        if args.disagree:
            a, b = (metrics.load(Path(args.disagree[k]) / f"{j}.jsonl") for j in args.judges)
            differ = [i for i in items if i.id in a and i.id in b and
                      any(x["satisfied"] != y["satisfied"] for x, y in zip(a[i.id]["items"], b[i.id]["items"],
                                                                         strict=True))]
            for i in rng.sample(differ, min(args.n_disagree, len(differ))):
                chosen[i.id] = i
                origin.setdefault(i.id, []).append("disagreement")
    queue = list(chosen.values())
    rng.shuffle(queue)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in queue:
            f.write(json.dumps({"id": i.id, "prompt": i.prompt, "requirements": list(i.requirements),
                                "image": str(i.image)}, ensure_ascii=False) + "\n")
    # which sample each image came from, kept apart from the queue and read only after the audit
    with out.with_suffix(".origin.json").open("w") as f:
        json.dump(origin, f)
    print(f"{len(queue)} images in the audit queue")


if __name__ == "__main__":
    main()
