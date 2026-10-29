"""Makespan of a batch: N images submitted together, time until the last verdict. This is the number that matters
when judging is one step of a pipeline: the next step waits for the whole batch, not for any one image.

    python experiments/throughput/makespan.py --name tuned_fp8_fp8kv --profile configs/servers/tuned_fp8.yaml \
        --set kv-cache-dtype=fp8 --data data/items/throughput_mix.jsonl --images 1000 --concurrency 256

Runs on one server, in this order:

  cold     the first batch after the server reports ready, input order: one-off costs (grammar compilation for
           each output schema, kernels compiled on first use) land here. Server start-up is reported separately
           from the log: weight loading (fast if the files are in the OS page cache), torch.compile (fast if
           vLLM's compile cache hits), CUDA graph capture.
  warm-up  a separate set of images, not measured.
  warm     the same batch once per policy, each starting from an empty prefix cache; the first policy is
           repeated at the end, so drift (clocks, temperature) shows up as a difference between identical runs.

Policies (client side, any OpenAI-compatible server):
  fifo      input order, requests admitted in arrival order
  lpt       longest checklist first, arrival order
  lpt_rank  longest checklist first, and a free slot goes to the image started earliest, so images in progress
            finish before new ones start

With --replicas N the profile runs once per GPU and the client spreads the batch over them (--routing, as in
sweep.py); metrics are summed over replicas.

Results: results/makespan_<name>.json (one entry per run) and makespan_<name>.<run>.timeline.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from sweep import settle

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items, shuffled_by_checklist
from fastqwenjudge.runner import run
from fastqwenjudge.server import Replicas, Server, load_profile
from fastqwenjudge.throughput import GpuMonitor, Monitor

HERE = Path(__file__).parent
POLICIES = {"fifo": ("input", "fifo"), "lpt": ("longest-first", "fifo"), "lpt_rank": ("longest-first", "rank")}


def completion(finish: np.ndarray, q: float) -> float:
    return float(np.quantile(finish, q)) if len(finish) else float("nan")


async def batch(server: Server | Replicas, args, items, policy: str, label: str, out: Path) -> dict:
    order, admission = POLICIES[policy]
    results_file = out / f"makespan_{args.name}.{label}.jsonl"
    results_file.unlink(missing_ok=True)
    # sampling and timeout default to the judges' usual greedy setup; a thinking model sets them
    async with Client(server.base_url, server.served_name, concurrency=args.concurrency,
                      admission=admission, routing=args.routing, temperature=getattr(args, "temperature", 0.0),
                      timeout=getattr(args, "timeout", 600), seed=getattr(args, "request_seed", 0)) as client:
        with GpuMonitor() as gpu:
            async with Monitor(server.base_url, interval=1.0) as monitor:
                t0 = time.monotonic()
                await run(judges.make(args.judge), items, client, results_file, order=order,
                          progress_every=args.progress_every)
                makespan = time.monotonic() - t0
    # requests carry the rank of their image (its position in the submission order)
    per_image: dict[int, list[float]] = {}
    for r in client.log:
        span = per_image.setdefault(r.rank, [r.start, r.start + r.latency])
        span[0], span[1] = min(span[0], r.start), max(span[1], r.start + r.latency)
    finish = np.sort([end - t0 for _, end in per_image.values()])
    samples = [{"t": s["t"] - t0, **{k: s[k] for k in ("running", "waiting", "kv_cache")}} for s in monitor.samples]
    middle = [s["running"] for s in samples if 0.1 * makespan <= s["t"] <= 0.8 * makespan and s["running"]]
    saturated = float(np.median(middle)) if middle else None
    idle = 0.0
    if saturated:   # capacity left unused: 1 - running / running at saturation, integrated over the batch
        for a, b in zip(samples, samples[1:], strict=False):
            idle += max(0.0, 1 - (a["running"] or 0) / saturated) * (b["t"] - a["t"])
    t10, t80 = completion(finish, 0.1), completion(finish, 0.8)
    g = gpu.summary() or {}
    gpus = g.get("gpus") or 1
    entry = {"run": label, "policy": policy, "images": len(items), "concurrency": args.concurrency,
             "makespan_s": makespan, "images_per_s": len(items) / makespan,
             "steady_images_per_s": float(np.sum((finish >= t10) & (finish <= t80)) / (t80 - t10)),
             "done_50_s": completion(finish, 0.5), "done_90_s": completion(finish, 0.9),
             "done_99_s": completion(finish, 0.99), "tail_after_90_s": makespan - completion(finish, 0.9),
             "running_saturated": saturated, "idle_capacity_s": idle, "idle_capacity_share": idle / makespan,
             "failed_requests": sum(not r.ok for r in client.log),
             # images whose verdict failed validation after all retries (the HTTP requests themselves succeeded)
             "failed_images": sorted(row["id"] for row in map(json.loads, results_file.open()) if "error" in row),
             "server": monitor.summary(), "gpu": g,
             "usd_per_1000_images": 1000 * gpus * args.gpu_hour_usd * makespan / 3600 / len(items)}
    if g.get("power_w"):
        entry["joules_per_image"] = g["power_w"] * makespan / len(items)
    (out / f"makespan_{args.name}.{label}.timeline.json").write_text(json.dumps(
        {"server": samples, "gpu": [{**s, "t": s["t"] - t0} for s in gpu.samples],
         "images": [{"rank": k, "start": v[0] - t0, "finish": v[1] - t0} for k, v in sorted(per_image.items())]}))
    print(f"{label} ({policy}): makespan {makespan:.0f} s, {entry['images_per_s']:.3f} images/s, 90% done at "
          f"{entry['done_90_s']:.0f} s, tail after 90% {entry['tail_after_90_s']:.0f} s, unused capacity "
          f"{100 * entry['idle_capacity_share']:.0f}%", flush=True)
    return entry


async def main_async(server: Server | Replicas, args, out: Path) -> list[dict]:
    ordered = shuffled_by_checklist(load_items(args.data), args.seed)
    measured, warmup = ordered[:args.images], ordered[args.images:args.images + args.warmup_images]
    runs, path = [], out / f"makespan_{args.name}.json"

    def save():
        path.write_text(json.dumps({"name": args.name, "profile": server.profile, "memory": server.memory_report(),
                                    "runs": runs, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                   indent=1))

    if not args.skip_cold:
        runs.append(await batch(server, args, measured, args.policies[0], "cold", out))
        save()
    warmup_file = out / f"makespan_{args.name}.warmup.jsonl"
    warmup_file.unlink(missing_ok=True)
    async with Client(server.base_url, server.served_name, concurrency=args.concurrency,
                      routing=args.routing) as client:
        await run(judges.make(args.judge), warmup, client, warmup_file, progress_every=args.progress_every)
    for i, policy in enumerate(args.policies + args.policies[:1]):   # first policy again: drift check
        await settle(server.base_url)
        runs.append(await batch(server, args, measured, policy, f"warm{i}_{policy}", out))
        save()
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judge", default="neutral")
    p.add_argument("--images", type=int, default=1000)
    p.add_argument("--warmup-images", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=256, help="requests in flight from the client")
    p.add_argument("--policies", nargs="+", default=["fifo", "lpt", "lpt_rank"], choices=sorted(POLICIES))
    p.add_argument("--skip-cold", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-hour-usd", type=float, default=3.0)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--replicas", type=int, default=1, help="independent servers, one per GPU")
    p.add_argument("--routing", default="prefix", choices=["prefix", "least_outstanding"])
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    profile = load_profile(args.profile, overrides)
    profile["name"] = f"makespan_{args.name}"
    profile.setdefault("env", {})["VLLM_SERVER_DEV_MODE"] = 1   # enables POST /reset_prefix_cache
    if args.replicas > 1:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.replicas)))).split(",")
        assert len(gpus) >= args.replicas, f"{args.replicas} replicas need as many GPUs, have {gpus}"
        server = Replicas(profile, gpus[:args.replicas], model=args.model, log_dir=out / "logs")
    else:
        server = Server(profile, model=args.model, log_dir=out / "logs")
    with server:
        asyncio.run(main_async(server, args, out))


if __name__ == "__main__":
    main()
