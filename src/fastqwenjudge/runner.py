"""Run a judge over a dataset with resumable output.

Results are appended to a JSONL file as they complete (one line per image), so an interrupted run picks up where
it stopped. Images are started in input order or longest checklist first (`order`); with a client that admits
requests by rank, the images started first also finish first.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .client import Client, request_rank
from .data import Item, short_hash

ORDERS = ("input", "longest-first")

# How requests are ordered on the client. "makespan" (the default): the longest checklists start first and
# requests of images already in progress are admitted before new ones start, so short ones fill the end of the batch
# instead of a long chain running alone. On 600 development images it cut the batch by 4-5% (27B) and 6-13%
# (Flash-Next) and the time with the GPU not full from 13-26% to 6-10% (thinking), with verdicts changing no more
# than between two identical runs. "latency": images start in input order and requests are admitted in arrival
# order, which is fair to every image.
OBJECTIVES = {"latency": {"order": "input", "admission": "fifo"},
              "makespan": {"order": "longest-first", "admission": "rank"}}


def result_key(judge_name: str, item: Item) -> str:
    return short_hash([judge_name, item.id, list(item.requirements)])


def key_name(judge) -> str:
    """The name a judge's results are keyed under (its `key_name` if it differs from the name it reports)."""
    return getattr(judge, "key_name", judge.name)


def longest_first(items: list[Item]) -> list[Item]:
    """Largest-processing-time-first, the standard heuristic for makespan: the longest checklists start first, so
    the short ones fill the end of the batch instead of a long one finishing alone. Cost is estimated from the
    requirement count, then the checklist length; images of one checklist stay together."""
    return sorted(items, key=lambda i: (-len(i.requirements), -sum(map(len, i.requirements)), i.checklist_key))


def load_results(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "result" in row:
                        done[row["key"]] = row
    return done


class Progress:
    def __init__(self, total: int, label: str, every: int = 100):
        self.total, self.label, self.every = total, label, every
        self.done = self.failed = 0
        self.start = time.monotonic()

    def tick(self, ok: bool):
        self.done += 1
        self.failed += not ok
        if self.done % self.every == 0 or self.done == self.total:
            rate = self.done / max(time.monotonic() - self.start, 1e-9)
            eta = (self.total - self.done) / max(rate, 1e-9) / 60
            print(f"{self.label} {self.done}/{self.total} | failed {self.failed} | {rate:.2f}/s | ETA {eta:.1f} min",
                  flush=True)


async def run(judge, items: list[Item], client: Client, out: str | Path, *, order: str = "input",
              progress_every: int = 100) -> dict[str, dict]:
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_results(out)
    todo = [i for i in items if result_key(key_name(judge), i) not in done]
    if order == "longest-first":
        todo = longest_first(todo)
    print(f"{judge.name}: {len(items) - len(todo)} cached, {len(todo)} to judge", flush=True)
    with out.open("a") as sink:
        def write(item: Item, result: dict | None = None, error: str | None = None):
            row = {"key": result_key(key_name(judge), item), "id": item.id, "judge": judge.name}
            row.update({"result": result} if error is None else {"error": error})
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if error is None:
                done[row["key"]] = row

        progress = Progress(len(todo), judge.name, progress_every)

        async def one(rank: int, item: Item):
            request_rank.set(rank)   # each image runs in its own task, so this stays with its requests
            try:
                write(item, await judge.judge(item, client))
                progress.tick(True)
            except Exception as error:
                write(item, error=f"{type(error).__name__}: {str(error)[:500]}")
                progress.tick(False)

        await asyncio.gather(*(one(rank, i) for rank, i in enumerate(todo)))
    keyed = {i.id: result_key(key_name(judge), i) for i in items}
    return {i: done[k]["result"] for i, k in keyed.items() if k in done}

