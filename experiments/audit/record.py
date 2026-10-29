"""Step through a blind audit queue and record verdicts in FastQwenJudge's results format.

    python experiments/audit/record.py next --queue audit/queue.jsonl --out audit/claude.jsonl
    python experiments/audit/record.py add --queue audit/queue.jsonl --out audit/claude.jsonl --id ID \
        --verdicts y n y --evidence "two dogs, one brown" "no frisbee visible" "grass field"

`?` marks a requirement the auditor cannot decide from the image; it is stored as not satisfied and listed in
`unsure`, so agreement can be reported with and without those requirements.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["next", "add", "status"])
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--id")
    p.add_argument("--verdicts", nargs="*")
    p.add_argument("--evidence", nargs="*")
    p.add_argument("--auditor", default="claude-opus-5.5 (manual audit)")
    args = p.parse_args()
    queue = load(args.queue)
    done = {r["id"] for r in load(args.out)}
    todo = [q for q in queue if q["id"] not in done]
    if args.action == "status":
        print(f"{len(done)} audited, {len(todo)} left")
    elif args.action == "next":
        if not todo:
            print("queue finished")
            return
        q = todo[0]
        print(json.dumps({"id": q["id"], "image": q["image"], "prompt": q["prompt"]}, ensure_ascii=False))
        for n, r in enumerate(q["requirements"], 1):
            print(f"{n}. {r}")
    else:
        q = next(q for q in queue if q["id"] == args.id)
        if q["id"] in done:
            raise SystemExit(f"{q['id']} already audited")
        if len(args.verdicts) != len(q["requirements"]) or len(args.evidence) != len(q["requirements"]):
            raise SystemExit(f"need {len(q['requirements'])} verdicts and evidence strings")
        if any(v not in ("y", "n", "?") for v in args.verdicts):
            raise SystemExit("verdicts are y, n or ?")
        items = [{"requirement": r, "evidence": e, "satisfied": v == "y"}
                 for r, v, e in zip(q["requirements"], args.verdicts, args.evidence, strict=True)]
        row = {"id": q["id"], "judge": args.auditor,
               "result": {"items": items, "score": sum(i["satisfied"] for i in items) / len(items),
                          "all_satisfied": all(i["satisfied"] for i in items),
                          "unsure": [n for n, v in enumerate(args.verdicts, 1) if v == "?"]},
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"recorded {q['id']} ({len(done) + 1}/{len(queue)})")


if __name__ == "__main__":
    main()
