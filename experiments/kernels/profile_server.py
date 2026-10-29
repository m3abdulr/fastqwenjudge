"""Where the GPU time goes inside the serving engine under real load: a torch-profiler trace of a few seconds of
closed-loop judging, summarized by kernel category.

    python experiments/kernels/profile_server.py --name tuned --profile configs/servers/tuned.yaml \
        --data data/items/throughput_mix.jsonl --concurrency 1 64

For each concurrency the script starts a fresh server with vLLM's profiler enabled (one profiling window per
server process), keeps `c` images in flight until the server is warm, records `--window` seconds, and writes
results/<name>_c<c>.json: GPU time per kernel category, the 25 most expensive kernels, and how much of the
window the GPU sat idle between kernels. With --with-stack the trace also records Python call stacks, and the summary
adds which host-side functions were running during the GPU's idle gaps (for a server that is CPU-bound).
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import gzip
import itertools
import json
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx
import yaml

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items, shuffled_by_checklist
from fastqwenjudge.server import Server, load_profile

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "throughput"))
from sweep import until  # noqa: E402

# first match wins; names are CUDA kernel symbols as the profiler records them
CATEGORIES = [
    ("attention", r"flash|fmha|attn|paged|reshape_and_cache|kernel_mha|xqa|batch(decode|prefill)"),
    ("fp8 activation quant", r"scale_1x128"),   # inside FlashInfer's small-batch FP8 GEMM, before the "gemm" rules
    ("gemm, fp8", r"cutlass.*(blockwise|fp8|e4m3|scaled)|_w8a8_block_fp8|sm90.*e4m3|deep_gemm|fp8_gemm"),
    ("gemm, bf16", r"nvjet|xmma|gemm|cublas|cutlass|matmul"),
    ("fp8 activation quant", r"per_token_group|quant"),
    ("rmsnorm", r"rms_norm|layer_norm|rmsnorm"),
    ("rotary", r"rotary|rope"),
    ("activation", r"silu|act_and_mul|gelu"),
    ("all-reduce", r"allreduce|all_reduce|nccl|cross_device"),
    ("fused by torch.compile", r"^triton_"),   # inductor kernels, e.g. residual add + RMSNorm
    ("copy / fill / pad", r"copy|fill|pad|cat|index|gather|scatter|memcpy|memset"),
    ("sampling", r"sampl|softmax|topk|argmax|logit|bitmask|penalt"),
]


def category(name: str) -> str:
    lower = name.lower()
    return next((c for c, pattern in CATEGORIES if re.search(pattern, lower)), "other")


def host_during_gaps(events: list, gaps: list[tuple[float, float]], top: int = 30) -> dict:
    """Host-side functions that were running on the thread that launches the kernels while the GPU sat idle, by idle
    time they overlap. Inclusive: a caller and its callees both count, so the list reads like a call tree flattened.
    Other threads (queues, thread pools) mostly wait, and would otherwise be counted on top. Needs a trace recorded
    with stacks."""
    launches: dict[tuple, int] = {}
    for e in events:
        if e.get("cat") == "cuda_runtime" and "Launch" in e.get("name", ""):
            launches[(e["pid"], e["tid"])] = launches.get((e["pid"], e["tid"]), 0) + 1
    main = max(launches, key=launches.get) if launches else None
    starts = [g[0] for g in gaps]
    idle: dict[str, float] = {}
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("python_function", "user_annotation") or not e.get("dur"):
            continue
        if main is not None and (e["pid"], e["tid"]) != main:
            continue
        s, t = e["ts"], e["ts"] + e["dur"]
        i, overlap = max(bisect.bisect_right(starts, s) - 1, 0), 0.0
        while i < len(gaps) and gaps[i][0] < t:
            overlap += max(0.0, min(t, gaps[i][1]) - max(s, gaps[i][0]))
            i += 1
        if overlap:
            idle[e["name"]] = idle.get(e["name"], 0.0) + overlap
    total = sum(b - a for a, b in gaps)
    return {"gaps": len(gaps), "idle_ms": total / 1000,
            "functions": [{"name": n[:200], "idle_ms": t / 1000, "share_of_idle": t / total}
                          for n, t in sorted(idle.items(), key=lambda kv: -kv[1])[:top]]}


def summarize(trace_path: Path, min_gap_us: float = 20.0) -> dict:
    with gzip.open(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("dur")), key=lambda e: e["ts"])
    if not kernels:
        return {"kernels": 0}
    span = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
    busy, end, gaps = 0.0, kernels[0]["ts"], []
    for e in kernels:   # union of kernel intervals (kernels from several streams can overlap)
        start, stop = max(e["ts"], end), e["ts"] + e["dur"]
        if e["ts"] - end >= min_gap_us:
            gaps.append((end, e["ts"]))
        if stop > start:
            busy += stop - start
        end = max(end, stop)
    by_category: dict[str, float] = {}
    by_name: dict[str, list] = {}
    for e in kernels:
        c = category(e["name"])
        by_category[c] = by_category.get(c, 0.0) + e["dur"]
        entry = by_name.setdefault(e["name"], [0, 0.0, c])
        entry[0] += 1
        entry[1] += e["dur"]
    total = sum(by_category.values())
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]
    summary = {"window_ms": span / 1000, "gpu_busy": busy / span, "kernels": len(kernels),
               "kernels_per_ms": len(kernels) / (span / 1000), "mean_kernel_us": total / len(kernels),
               "categories": {c: {"ms": t / 1000, "share": t / total}
                              for c, t in sorted(by_category.items(), key=lambda kv: -kv[1])},
               "top_kernels": [{"name": n[:160], "category": c, "calls": calls, "ms": t / 1000, "share": t / total}
                               for n, (calls, t, c) in top]}
    if any(e.get("cat") == "python_function" for e in events):
        summary["host_during_gaps"] = host_during_gaps(events, gaps)
    return summary


async def load_and_profile(server: Server, args, concurrency: int, trace_dir: Path) -> Path:
    items = shuffled_by_checklist(load_items(args.data), args.seed)
    stream = itertools.cycle(items)
    judge = judges.make(args.judge)
    done: list[float] = []
    stop = asyncio.Event()
    root = server.base_url.removesuffix("/v1")
    async with Client(server.base_url, server.served_name, concurrency=2 * concurrency) as client:
        async def worker():
            while not stop.is_set():
                try:
                    await judge.judge(next(stream), client)
                except Exception:
                    pass
                done.append(time.monotonic())

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await until(lambda: len(done) >= concurrency, args.warmup, args.max_wait)
        before = set(trace_dir.glob("*.pt.trace.json.gz"))
        async with httpx.AsyncClient(timeout=1800) as http:
            (await http.post(f"{root}/start_profile")).raise_for_status()
            await asyncio.sleep(args.window)
            (await http.post(f"{root}/stop_profile")).raise_for_status()   # returns once the trace is written
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=60)
    # one file from the API server process (CPU only) and one from the engine core (with the GPU kernels), the larger
    return max(set(trace_dir.glob("*.pt.trace.json.gz")) - before, key=lambda p: p.stat().st_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judge", default="neutral")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 64])
    p.add_argument("--warmup", type=float, default=60)
    p.add_argument("--max-wait", type=float, default=600)
    p.add_argument("--window", type=float, default=5, help="seconds recorded")
    p.add_argument("--with-stack", action="store_true", help="record Python stacks to attribute GPU idle time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()
    out = Path(args.out)
    for c in args.concurrency:
        trace_dir = out / "traces" / f"{args.name}_c{c}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
        profile = load_profile(args.profile, overrides)
        profile["name"] = f"profile_{args.name}_c{c}"
        if tuple(map(int, version("vllm").split(".")[:2])) >= (0, 12):
            profile["args"]["profiler-config"] = {"profiler": "torch", "torch_profiler_dir": str(trace_dir.resolve()),
                                                  "torch_profiler_with_stack": args.with_stack,
                                                  "torch_profiler_record_shapes": True}
        else:   # vLLM 0.11 reads the profiler settings from the environment
            profile.setdefault("env", {}).update({"VLLM_TORCH_PROFILER_DIR": str(trace_dir.resolve()),
                                                  "VLLM_TORCH_PROFILER_WITH_STACK": int(args.with_stack),
                                                  "VLLM_TORCH_PROFILER_RECORD_SHAPES": 1})
        with Server(profile, model=args.model, log_dir=out / "logs") as server:
            trace = asyncio.run(load_and_profile(server, args, c, trace_dir))
        result = {"name": args.name, "concurrency": c, "profile": profile, "trace": str(trace), **summarize(trace)}
        (out / f"{args.name}_c{c}.json").write_text(json.dumps(result, indent=1))
        print(f"{args.name} c={c}: GPU busy {100 * result['gpu_busy']:.0f}% of {result['window_ms']:.0f} ms, "
              f"{result['kernels_per_ms']:.1f} kernels/ms, mean kernel {result['mean_kernel_us']:.1f} us", flush=True)
        for cat, v in result["categories"].items():
            print(f"  {cat:24} {100 * v['share']:5.1f}%  {v['ms']:8.1f} ms", flush=True)
        if host := result.get("host_during_gaps"):
            print(f"  idle gaps >= 20 us: {host['gaps']}, {host['idle_ms']:.0f} ms; host functions running then:")
            for f in host["functions"][:15]:
                print(f"    {100 * f['share_of_idle']:5.1f}%  {f['name'][:110]}", flush=True)


if __name__ == "__main__":
    main()
