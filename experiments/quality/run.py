"""Run several judges on the same images with one server, for the agreement comparison.

    python experiments/quality/run.py --profile configs/servers/tuned.yaml --data data/items/sol_subset.jsonl \
        --judges neutral single_v1 single_v2 vqascore neutral_leaky --out experiments/quality/results/sol_subset

The guard ablation needs no extra requests: neutral results keep the decide step's verdicts from before the
guards (`text_verdict_before_guards`), see analyze.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items
from fastqwenjudge.runner import run
from fastqwenjudge.server import Server, load_profile
from fastqwenjudge.throughput import GpuMonitor, Monitor, request_summary


async def run_all(server, items, names, out: Path, concurrency: int, temperature: float = 0.0, timeout: float = 600,
                  seed: int | None = 0):
    timings = {}
    for name in names:
        start = time.monotonic()
        with GpuMonitor() as gpu:
            async with Client(server.base_url, server.served_name, concurrency=concurrency,
                              temperature=temperature, timeout=timeout, seed=seed) as client, \
                    Monitor(server.base_url) as monitor:
                await run(judges.make(name), items, client, out / f"{name}.jsonl")
        timings[name] = {"wall_seconds": time.monotonic() - start, "server": monitor.summary(),
                         "gpu": gpu.summary(), "client": request_summary(client.log)}
        print(f"{name}: {timings[name]['wall_seconds']:.0f} s", flush=True)
    return timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judges", nargs="+", default=["neutral", "single_v1", "single_v2", "vqascore", "neutral_leaky"])
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.0, help="seeded sampling, e.g. for a thinking model")
    p.add_argument("--timeout", type=float, default=600, help="seconds per request")
    p.add_argument("--seed", default="0", help='sampling seed sent with every request, or "none" for unseeded')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data, limit=args.limit)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    with Server(load_profile(args.profile, overrides), model=args.model, log_dir=out / "logs") as server:
        timings = asyncio.run(run_all(server, items, args.judges, out, args.concurrency, args.temperature,
                                      args.timeout, None if args.seed == "none" else int(args.seed)))
        timings["memory"] = server.memory_report()
    (out / f"timings_{int(time.time())}.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
