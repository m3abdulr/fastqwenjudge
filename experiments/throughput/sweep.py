"""Latency-throughput curve: one server, closed-loop load at increasing concurrency.

    python experiments/throughput/sweep.py --name tuned --profile configs/servers/tuned.yaml \
        --data data/items/throughput_mix.jsonl --concurrency 1 2 4 8 16 32 64 128

At concurrency c, c workers each judge one image at a time and take the next image from a shuffled stream as
soon as they finish, so the server always has about c images in progress. Every level starts from the same
point of the same stream, with an empty prefix cache, so all setups see the same images at a given level.
With --replicas N the profile runs once per GPU and the client sends each request to the replica most likely to have
its prompt's beginning cached (client.PrefixRouter), or with --routing least_outstanding to the one with the fewest
in flight; metrics are summed over replicas, and each point records how the requests split between them.

A level is measured only once it is in steady state. At high concurrency an image can take minutes, much longer
than any fixed warm-up, so the warm-up lasts at least --warmup seconds and until c images have finished (one
full wave), and the window lasts at least --window seconds and until max(c, --min-images) images have finished
inside it. Throughput counts images finished inside the window; latency is per image, from its first request to
its verdict. `in_flight` (throughput times mean latency) should come out close to c (Little's law); a value well
below c means the level was cut off by --max-wait before it settled. Work still running when the window closes
is cancelled. Results are written after every level, so a crash keeps the finished levels.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items, shuffled_by_checklist
from fastqwenjudge.server import Replicas, Server, load_profile
from fastqwenjudge.throughput import GpuMonitor, Monitor, parse_metrics

HERE = Path(__file__).parent


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    return {"mean": float(np.mean(values)), **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 99)}}


async def settle(base_url: str | list[str], timeout: float = 120):
    """Wait until the cancelled requests of the previous level are gone (the server aborts a request when its
    connection closes), then empty the prefix cache so the next level starts cold."""
    for root in [u.removesuffix("/v1") for u in ([base_url] if isinstance(base_url, str) else base_url)]:
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as http:
            while time.monotonic() < deadline:
                values = parse_metrics((await http.get(f"{root}/metrics", timeout=10)).text)
                if values.get("vllm:num_requests_running", 0) + values.get("vllm:num_requests_waiting", 0) == 0:
                    break
                await asyncio.sleep(1)
            (await http.post(f"{root}/reset_prefix_cache", timeout=30)).raise_for_status()


async def until(done, at_least: float, at_most: float):
    """Sleep for at least `at_least` seconds, then until done() or `at_most` seconds have passed."""
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < at_least or (not done() and elapsed < at_most):
        await asyncio.sleep(1)


async def level(server: Server, args, items, concurrency: int):
    judge = judges.make(args.judge)   # fresh compile cache: each level pays for its own compiles
    stream = itertools.cycle(items)
    images: list[dict] = []
    began_level = time.monotonic()
    stop = asyncio.Event()
    async with Client(server.base_url, server.served_name, concurrency=2 * concurrency,
                      routing=args.routing) as client:
        # the HTTP stack may turn a cancellation into an ordinary error, so workers also check `stop`
        async def worker():
            while not stop.is_set():
                item = next(stream)
                began = time.monotonic()
                try:
                    await judge.judge(item, client)
                    ok = True
                except Exception:
                    ok = False
                images.append({"id": item.id, "start": began, "end": time.monotonic(), "ok": ok})

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await until(lambda: len(images) >= concurrency, args.warmup, args.max_wait)
        t0, cpu0 = time.monotonic(), time.process_time()
        needed = max(concurrency, args.min_images)
        with GpuMonitor() as gpu:
            async with Monitor(server.base_url) as monitor:
                await until(lambda: sum(r["end"] >= t0 for r in images) >= needed, args.window, args.max_wait)
        t1, cpu1 = time.monotonic(), time.process_time()
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=60)
    await settle(server.base_url)

    finished = [r for r in images if t0 <= r["end"] <= t1]
    ok = [r for r in finished if r["ok"]]
    requests = [r for r in client.log if r.ok and t0 <= r.start + r.latency <= t1]
    stages = {}
    for stage in sorted({r.stage for r in requests}):
        rows = [r for r in requests if r.stage == stage]
        stages[stage] = {"requests": len(rows), "latency": percentiles([r.latency for r in rows]),
                         "prompt_tokens": float(np.mean([r.prompt_tokens for r in rows])),
                         "completion_tokens": float(np.mean([r.completion_tokens for r in rows]))}
    latency = percentiles([r["end"] - r["start"] for r in ok])
    point = {"concurrency": concurrency, "warmup_seconds": t0 - began_level, "window_seconds": t1 - t0,
             "images": len(ok), "failed": len(finished) - len(ok), "images_per_s": len(ok) / (t1 - t0),
             "latency": latency, "in_flight": len(ok) / (t1 - t0) * latency.get("mean", 0), "stages": stages,
             # cores the load generator itself kept busy (image encoding, HTTP, JSON); near 1 it would be the limit
             "client_cpu_cores": (cpu1 - cpu0) / (t1 - t0),
             "requests_per_replica": np.bincount(np.array([r.server for r in requests], dtype=int),
                                                 minlength=len(client.apis)).tolist(),
             "server": monitor.summary(), "gpu": gpu.summary()}
    if point["gpu"] and ok:
        point["joules_per_image"] = point["gpu"]["power_w"] * (t1 - t0) / len(ok)
    rows = [{"concurrency": concurrency, "id": r["id"], "latency": r["end"] - r["start"], "ok": r["ok"]}
            for r in finished]
    gpu_samples = [{"concurrency": concurrency, **s, "t": s["t"] - t0} for s in gpu.samples]
    # raw Prometheus text at both ends of the window, so any server metric can be recomputed later
    snapshot = {"concurrency": concurrency, "start": monitor.start_text, "end": monitor.end_text}
    return point, rows, gpu_samples, snapshot


async def sweep(server: Server, args, out: Path) -> list[dict]:
    items = shuffled_by_checklist(load_items(args.data), args.seed)
    points = []
    await settle(server.base_url)
    for c in args.concurrency:
        point, rows, gpu_samples, snapshot = await level(server, args, items, c)
        points.append(point)
        result = {"name": args.name, "judge": args.judge, "routing": args.routing, "profile": server.profile,
                  "memory": server.memory_report(),
                  "points": points, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out.with_suffix(".json").write_text(json.dumps(result, indent=1))
        for suffix, lines in ((".images.jsonl", rows), (".gpu.jsonl", gpu_samples), (".metrics.jsonl", [snapshot])):
            with out.with_suffix(suffix).open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in lines)
        lat, s, g = point["latency"], point["server"], point["gpu"] or {}
        ttft, itl = s["histograms"]["ttft"], s["histograms"]["itl"]
        print(f"c={c}: {point['images_per_s']:.3f} images/s ({point['images']} images, {point['failed']} failed), "
              f"latency p50 {lat.get('p50', float('nan')):.1f} s p90 {lat.get('p90', float('nan')):.1f} s, "
              f"{point['in_flight']:.1f} in flight, "
              f"{s['generation_tokens_per_s']:.0f} generated tok/s, {s['mean_running'] or 0:.1f} running, "
              f"TTFT p50 {ttft['p50'] or float('nan'):.2f} s, ITL {1000 * (itl['mean'] or float('nan')):.0f} ms, "
              f"GPU {g.get('utilization', float('nan')):.0f}% busy, {g.get('power_w', float('nan')):.0f} W", flush=True)
        if not server.running():
            print("server exited; stopping the sweep", flush=True)
            break
    return points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--model", help="model path (defaults to the profile's model)")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judge", default="neutral")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--warmup", type=float, default=30, help="minimum warm-up seconds per level")
    p.add_argument("--window", type=float, default=180, help="minimum measured seconds per level")
    p.add_argument("--min-images", type=int, default=16, help="minimum images finished inside the window")
    p.add_argument("--max-wait", type=float, default=900, help="cap on the warm-up and on the window, seconds")
    p.add_argument("--replicas", type=int, default=1, help="independent servers, one per GPU")
    p.add_argument("--routing", default="prefix", choices=["prefix", "least_outstanding"],
                   help="how the client picks a replica for each request")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()
    out = Path(args.out) / f"sweep_{args.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".images.jsonl", ".gpu.jsonl", ".metrics.jsonl"):
        out.with_suffix(suffix).unlink(missing_ok=True)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    profile = load_profile(args.profile, {k: yaml.safe_load(v) for k, v in overrides.items()})
    profile["name"] = f"sweep_{args.name}"
    profile.setdefault("env", {})["VLLM_SERVER_DEV_MODE"] = 1   # enables POST /reset_prefix_cache
    if args.replicas > 1:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.replicas)))).split(",")
        assert len(gpus) >= args.replicas, f"{args.replicas} replicas need as many GPUs, have {gpus}"
        server = Replicas(profile, gpus[:args.replicas], model=args.model, log_dir=Path(args.out) / "logs")
    else:
        server = Server(profile, model=args.model, log_dir=Path(args.out) / "logs")
    with server:
        asyncio.run(sweep(server, args, out))


if __name__ == "__main__":
    main()
