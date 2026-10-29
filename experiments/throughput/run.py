"""One throughput measurement: start a server, warm it up, judge a fixed workload, record rates and latencies.

    python experiments/throughput/run.py --name L4_tuned_pipelined --profile configs/servers/tuned.yaml \
        --data data/items/throughput_mix.jsonl --judge neutral --mode pipelined --concurrency 128 --images 1000

Server start-up and the warm-up images are excluded from the measured window. Every run starts from an empty
results file, so nothing is served from cache.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items, shuffled_by_checklist
from fastqwenjudge.runner import run
from fastqwenjudge.server import Server, free_port, load_profile
from fastqwenjudge.throughput import GpuMonitor, Monitor, request_summary

HERE = Path(__file__).parent


def workload(path: str, n: int, warmup: int, seed: int = 0):
    ordered = shuffled_by_checklist(load_items(path), seed)
    return ordered[:warmup], ordered[warmup:warmup + n]


class HFServer:
    """The plain-transformers baseline, served through the same OpenAI-compatible interface."""

    def __init__(self, model: str, log_dir: Path):
        self.port, self.model, self.log_dir = free_port(), model, log_dir
        self.served_name = "judge"
        self.profile = {"name": "hf_transformers", "model": model, "args": {}}
        self.startup_seconds = None

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / f"hf_{int(time.time())}.log").open("w")
        self.process = subprocess.Popen([sys.executable, str(HERE / "hf_server.py"), "--model", self.model,
                                         "--port", str(self.port)], stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("HF server exited; see its log")
            try:
                if httpx.get(f"{self.base_url}/models", timeout=3).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        self.startup_seconds = time.monotonic() - start
        return self

    def __exit__(self, *exc):
        self.process.terminate()
        self.process.wait(timeout=60)

    def memory_report(self):
        return {"startup_seconds": self.startup_seconds}


async def measure(server, args, warm, measured, out_dir: Path) -> dict:
    judge = judges.make(args.judge)
    for name in ("warmup", "measured"):
        (out_dir / f"{args.name}.{name}.jsonl").unlink(missing_ok=True)
    async with Client(server.base_url, server.served_name, concurrency=args.concurrency) as client:
        if warm:
            await run(judge, warm, client, out_dir / f"{args.name}.warmup.jsonl", mode=args.mode)
        judge = judges.make(args.judge)   # fresh compile memo: the measured window pays for its own compiles
        start = time.monotonic()
        with GpuMonitor() as gpu:
            async with Monitor(server.base_url) as monitor:
                results = await run(judge, measured, client, out_dir / f"{args.name}.measured.jsonl",
                                    mode=args.mode)
        wall = time.monotonic() - start
    # one line per request in the measured window, for latency distributions
    with (out_dir / f"{args.name}.requests.jsonl").open("w") as f:
        for r in client.log:
            if r.start >= start:
                f.write(json.dumps({**asdict(r), "start": r.start - start}) + "\n")
    return {"name": args.name, "judge": judge.name, "mode": args.mode, "concurrency": args.concurrency,
            "images": len(measured), "judged": len(results), "wall_seconds": wall,
            "images_per_s": len(measured) / wall, "server": monitor.summary(), "gpu": gpu.summary(),
            "client": request_summary(client.log, start), "memory": server.memory_report(),
            "profile": server.profile, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--profile", help="server profile YAML, or 'hf' for the transformers baseline")
    p.add_argument("--model", help="model path (defaults to the profile's model)")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judge", default="neutral")
    p.add_argument("--mode", default="pipelined", choices=["pipelined", "staged"])
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--images", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=32)
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm, measured = workload(args.data, args.images, args.warmup)
    if args.profile == "hf":
        server = HFServer(args.model, out_dir / "logs")
    else:
        overrides = dict(kv.split("=", 1) for kv in args.set)
        profile = load_profile(args.profile, {k: yaml.safe_load(v) for k, v in overrides.items()})
        profile["name"] = args.name
        server = Server(profile, model=args.model, log_dir=out_dir / "logs")
    with server:
        stats = asyncio.run(measure(server, args, warm, measured, out_dir))
    (out_dir / f"{args.name}.json").write_text(json.dumps(stats, indent=1))
    s = stats["server"]
    print(f"{args.name}: {stats['images_per_s']:.3f} images/s, {s['prompt_tokens_per_s']:.0f} prompt + "
          f"{s['generation_tokens_per_s']:.0f} generated tok/s, p50 {stats['client']['latency_p50']:.1f} s")


if __name__ == "__main__":
    main()
