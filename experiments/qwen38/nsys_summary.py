"""Kernel time by class over a whole batch, from an Nsight Systems capture of the server (docs/roofline.md, "Where a
whole batch goes"): profile_gpu.py's summary, for every kernel of a batch instead of a few seconds of one.

The capture covers one batch: the server runs with vLLM's CUDA profiler (`profiler-config: {"profiler": "cuda"}`)
under `nsys profile --capture-range=cudaProfilerApi`, and the client brackets the batch with /start_profile and
/stop_profile (a 600-image batch per judge, after a short warm-up). CUDA graphs are traced per node, so the kernels
inside them are counted one by one.

    nsys export --type sqlite -o <capture>.sqlite <capture>.nsys-rep
    python experiments/qwen38/nsys_summary.py <capture>.sqlite --out <summary>.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_gpu import category  # noqa: E402


def summarize(db: Path, min_gap_us: float = 20.0) -> dict:
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = con.execute("SELECT start, end, deviceId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    per_device: dict[int, dict] = {}
    by_category: dict[str, float] = {}
    by_name: dict[str, list] = {}
    cat_of: dict[int, str] = {}
    n = 0
    for start, end, dev, name_id in rows:   # ns
        n += 1
        d = per_device.setdefault(dev, {"first": start, "end": start, "busy": 0, "gaps": 0, "gap_ns": 0})
        if start - d["end"] >= min_gap_us * 1000:
            d["gaps"] += 1
            d["gap_ns"] += start - d["end"]
        lo = max(start, d["end"])   # union of kernel intervals: kernels on several streams can overlap
        if end > lo:
            d["busy"] += end - lo
        d["end"] = max(d["end"], end)
        c = cat_of.get(name_id)
        if c is None:
            c = cat_of[name_id] = category(names[name_id])
        dur = end - start
        by_category[c] = by_category.get(c, 0) + dur
        e = by_name.setdefault(name_id, [0, 0, c])
        e[0] += 1
        e[1] += dur
    total = sum(by_category.values())
    devices = {dev: {"window_ms": (d["end"] - d["first"]) / 1e6, "gpu_busy": d["busy"] / (d["end"] - d["first"]),
                     "idle_gaps": d["gaps"], "idle_in_gaps_ms": d["gap_ns"] / 1e6}
               for dev, d in sorted(per_device.items())}
    first = devices[min(devices)]
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:100]
    return {"source": "nsys", "window_ms": first["window_ms"], "gpu_busy": first["gpu_busy"], "devices": devices,
            "kernels": n, "mean_kernel_us": total / max(n, 1) / 1000,
            "categories": {c: {"ms": t / 1e6, "share": t / total}
                           for c, t in sorted(by_category.items(), key=lambda kv: -kv[1])},
            "top_kernels": [{"name": names[i][:160], "category": c, "calls": calls, "ms": t / 1e6,
                             "share": t / total} for i, (calls, t, c) in top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = summarize(Path(args.sqlite))
    Path(args.out).write_text(json.dumps(s, indent=1))
    print(f"{args.sqlite}: {s['kernels']} kernels over {s['window_ms'] / 1000:.1f} s, GPU busy "
          f"{100 * s['gpu_busy']:.1f}%")
    for c, v in s["categories"].items():
        print(f"  {c:24s} {v['ms'] / 1000:8.2f} s  {100 * v['share']:5.1f}%")


if __name__ == "__main__":
    main()
