"""Throughput measurement: server-side counters from vLLM's Prometheus endpoint, GPU telemetry from nvidia-smi,
and client-side timings.

Token rates come from the server's own counters over the measured window, so they count exactly what the
engine processed (image tokens included). Gauges (running / waiting requests, KV-cache use) are sampled every
`interval` seconds. vLLM's latency histograms (time to first token, inter-token latency, queueing, prefill,
decode) and its tokens-per-engine-step histogram are differenced over the window, which gives their means and
approximate percentiles for exactly the requests that finished inside it.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import threading
import time

import httpx
import numpy as np

COUNTERS = {"prompt_tokens": "vllm:prompt_tokens_total", "generation_tokens": "vllm:generation_tokens_total",
            "prefix_queries": "vllm:prefix_cache_queries_total", "prefix_hits": "vllm:prefix_cache_hits_total",
            "preemptions": "vllm:num_preemptions_total", "requests": "vllm:request_success_total",
            "spec_drafts": "vllm:spec_decode_num_drafts_total",
            "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
            "spec_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total"}
HISTOGRAMS = {"ttft": "vllm:time_to_first_token_seconds", "itl": "vllm:inter_token_latency_seconds",
              "queue": "vllm:request_queue_time_seconds", "prefill": "vllm:request_prefill_time_seconds",
              "decode": "vllm:request_decode_time_seconds", "step_tokens": "vllm:iteration_tokens_total",
              # per request: decode time over its output tokens (vLLM's TPOT); 1 / TPOT is the request's decode speed
              "tpot": "vllm:request_time_per_output_token_seconds"}
GAUGES = {"running": ("vllm:num_requests_running",), "waiting": ("vllm:num_requests_waiting",),
          "kv_cache": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")}
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")
LE = re.compile(r'le="([^"]+)"')


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each metric over its label sets."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            values[m.group(1)] = values.get(m.group(1), 0.0) + float(m.group(3))
    return values


def parse_buckets(text: str, name: str) -> dict[float, float]:
    """Cumulative bucket counts of one histogram, keyed by upper bound and summed over all other labels."""
    buckets: dict[float, float] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m and m.group(1) == f"{name}_bucket" and (le := LE.search(m.group(2) or "")):
            bound = float(le.group(1))   # "+Inf" parses to inf
            buckets[bound] = buckets.get(bound, 0.0) + float(m.group(3))
    return buckets


def quantile(buckets: dict[float, float], q: float) -> float | None:
    """Like Prometheus' histogram_quantile: linear interpolation inside the bucket that holds the q-th
    observation. Only as precise as the bucket bounds."""
    bounds = sorted(buckets)
    if not bounds or not buckets[bounds[-1]]:
        return None
    rank = q * buckets[bounds[-1]]
    lower = below = 0.0
    for upper in bounds:
        if buckets[upper] >= rank:
            if math.isinf(upper) or buckets[upper] == below:
                return lower
            return lower + (upper - lower) * (rank - below) / (buckets[upper] - below)
        lower, below = upper, buckets[upper]
    return lower


class Monitor:
    """Counters and histograms are summed over servers (and over the engines of a data-parallel server), and so
    are the gauges: running and waiting requests in total, KV-cache use as a sum of per-replica fractions."""

    def __init__(self, base_url: str | list[str], interval: float = 2.0):
        urls = [base_url] if isinstance(base_url, str) else base_url
        self.urls = [url.removesuffix("/v1") + "/metrics" for url in urls]
        self.interval = interval
        self.samples: list[dict] = []
        self._task = None
        self.start_text = self.end_text = ""
        self.start_time = self.end_time = None

    async def _scrape(self, client: httpx.AsyncClient, timeout: float = 10) -> str:
        """The servers' metrics as one text; parse_metrics sums a metric over all its lines."""
        return "\n".join([(await client.get(url, timeout=timeout)).text for url in self.urls])

    async def _snapshot(self, attempts: int = 4) -> str:
        """The first and last scrape bound the counters, so they are retried: a server busy with hundreds of image
        requests can take longer than one scrape's timeout to answer its metrics endpoint."""
        async with httpx.AsyncClient() as client:
            for attempt in range(attempts):
                try:
                    return await self._scrape(client, timeout=30)
                except httpx.HTTPError:
                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(2)

    async def _loop(self):
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    values = parse_metrics(await self._scrape(client))
                    sample = {"t": time.monotonic()}
                    for key, names in GAUGES.items():
                        sample[key] = next((values[n] for n in names if n in values), None)
                    self.samples.append(sample)
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.interval)

    async def __aenter__(self):
        self.start_text = await self._snapshot()
        self.start_time = time.monotonic()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self.end_time = time.monotonic()
        self._task.cancel()
        self.end_text = await self._snapshot()

    def summary(self) -> dict:
        seconds = self.end_time - self.start_time
        start, end = parse_metrics(self.start_text), parse_metrics(self.end_text)
        delta = {k: end.get(n, 0.0) - start.get(n, 0.0) for k, n in COUNTERS.items()}
        queries = delta["prefix_queries"]
        out = {"seconds": seconds, "counters": delta,
               "prompt_tokens_per_s": delta["prompt_tokens"] / seconds,
               "generation_tokens_per_s": delta["generation_tokens"] / seconds,
               "prefix_cache_hit_rate": delta["prefix_hits"] / queries if queries else None}
        out["total_tokens_per_s"] = out["prompt_tokens_per_s"] + out["generation_tokens_per_s"]
        for key in GAUGES:
            values = [s[key] for s in self.samples if s.get(key) is not None]
            out[f"mean_{key}"] = float(np.mean(values)) if values else None
        out["histograms"] = {}
        for key, name in HISTOGRAMS.items():
            count = end.get(f"{name}_count", 0.0) - start.get(f"{name}_count", 0.0)
            total = end.get(f"{name}_sum", 0.0) - start.get(f"{name}_sum", 0.0)
            before = parse_buckets(self.start_text, name)
            buckets = {b: c - before.get(b, 0.0) for b, c in parse_buckets(self.end_text, name).items()}
            out["histograms"][key] = {"count": count, "mean": total / count if count else None,
                                      **{f"p{q}": quantile(buckets, q / 100) for q in (50, 90, 95, 99)}}
        out["engine_steps_per_s"] = out["histograms"]["step_tokens"]["count"] / seconds
        return out


GPU_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw", "power.limit",
              "clocks.sm", "temperature.gpu")


def _visible_gpus() -> list[str]:
    """CUDA_VISIBLE_DEVICES if it narrows what nvidia-smi lists; under a scheduler that isolates GPUs,
    nvidia-smi already lists only the job's own and nothing needs selecting."""
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    listed = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                            text=True).stdout.split()
    return visible if visible and len(visible) < len(listed) else []


def _number(field: str) -> float | None:
    try:
        return float(field)
    except ValueError:   # [N/A] or [Not Supported]
        return None


class GpuMonitor:
    """nvidia-smi sampled every `interval_ms`: utilization (share of time a kernel was running, not how much of
    the GPU it used), memory-controller utilization, memory used, power draw against the power limit, SM clock
    (drops when the GPU throttles) and temperature. Records nothing if nvidia-smi is not installed."""

    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self._process = self._thread = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            cmd = ["nvidia-smi", "--query-gpu=index," + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits",
                   f"--loop-ms={self.interval_ms}"]
            if gpus := _visible_gpus():
                cmd.append("--id=" + ",".join(gpus))
            self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        return self

    def _read(self):
        for line in self._process.stdout:
            index, *values = (_number(v) for v in line.split(","))
            if index is not None and len(values) == len(GPU_FIELDS):
                sample = dict(zip(GPU_FIELDS, values, strict=True))
                self.samples.append({"t": time.monotonic(), "gpu": int(index), **sample})

    def __exit__(self, *exc):
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._thread.join(timeout=10)

    def summary(self) -> dict | None:
        """Means over the samples; power is the mean total draw of all GPUs, so energy = power_w * seconds."""
        if not self.samples:
            return None
        gpus = sorted({s["gpu"] for s in self.samples})

        def mean(key, rows=self.samples):
            values = [s[key] for s in rows if s[key] is not None]
            return float(np.mean(values)) if values else None

        def total(key):
            means = [mean(key, [s for s in self.samples if s["gpu"] == g]) for g in gpus]
            return sum(means) if None not in means else None
        return {"gpus": len(gpus), "utilization": mean("utilization.gpu"),
                "memory_utilization": mean("utilization.memory"),
                "memory_used_gib": max(s["memory.used"] or 0 for s in self.samples) / 1024,
                "memory_total_gib": (self.samples[0]["memory.total"] or 0) / 1024,
                "power_w": total("power.draw"), "power_limit_w": total("power.limit"),
                "sm_clock_mhz": mean("clocks.sm"), "temperature_c": mean("temperature.gpu")}


def request_summary(log, start: float | None = None) -> dict:
    """Client-side latency percentiles and per-stage request counts from Client.log."""
    rows = [r for r in log if start is None or r.start >= start]
    ok = [r for r in rows if r.ok]
    lat = np.array([r.latency for r in ok]) if ok else np.array([np.nan])
    stages = {}
    for r in ok:
        s = stages.setdefault(r.stage, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        s["requests"] += 1
        s["prompt_tokens"] += r.prompt_tokens
        s["completion_tokens"] += r.completion_tokens
    return {"requests": len(rows), "failed_requests": len(rows) - len(ok),
            "latency_p50": float(np.nanpercentile(lat, 50)), "latency_p95": float(np.nanpercentile(lat, 95)),
            "stages": stages}
