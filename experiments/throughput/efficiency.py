"""Hardware efficiency of each sweep point: what share of the GPU's compute and memory bandwidth the server used.

    python experiments/throughput/efficiency.py --out experiments/throughput/results/efficiency

MFU (model FLOPs utilization) counts 2 FLOPs per language-model weight for every token the engine computed
(prompt tokens served from the prefix cache excluded), against the dense tensor-core peak of the precision the
weights run in. Attention scores and the vision encoder are left out (together about a tenth of the neutral
judge's work at these lengths), so MFU is an underestimate. With speculative decoding, draft tokens the model
rejects are computed but not counted: MFU and tokens per step count useful work only.

MBU (memory bandwidth utilization) counts the bytes every engine step has to read, all language-model weights
plus the KV cache of the running requests, times the engine steps per second (counted by vLLM, so an idle
replica adds nothing), against peak HBM bandwidth. Activations are left out. A decode step with B requests does
about 2·N·B FLOPs on 2·N bytes of BF16 weights, B FLOPs per byte, while the H100 needs about 295 to be
compute-bound: decoding is limited by memory bandwidth until B is in the hundreds, which is why MBU, not MFU, is
the number to watch for decode.

vLLM's histogram buckets are coarse (inter-token latencies of 25 ms and 49 ms share a bucket), so means are
reported for inter-token latency and request phases; percentiles only for time to first token.
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent

# H100 SXM5, nvidia.com/en-us/data-center/h100 (the page lists tensor-core FLOPS with 2:4 sparsity; dense is half)
H100 = {"bf16_flops": 989.4e12, "fp8_flops": 1978.9e12, "hbm_bytes_per_s": 3.35e12, "power_w": 700}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}


@dataclass
class Model:
    flop_params: float      # language-model weights that multiply every token (embedding table excluded)
    weight_bytes: float     # bytes of those weights, read once per engine step
    fp8: bool
    kv_bytes_per_token: dict[str, int]   # by KV cache dtype


def load_model(name_or_path: str) -> Model:
    """Parameter counts and weight bytes from the checkpoint's safetensors headers (no weights are loaded)."""
    path = Path(name_or_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(name_or_path, local_files_only=True))
    params = weight_bytes = embedding = embedding_bytes = 0
    for shard in sorted(path.glob("*.safetensors")):
        with shard.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for key, t in header.items():
            if key == "__metadata__" or ".visual." in key or key.startswith("visual."):
                continue
            numel = 1
            for d in t["shape"]:
                numel *= d
            size = numel * DTYPE_BYTES[t["dtype"]]
            if "embed_tokens" in key:
                embedding, embedding_bytes = embedding + numel, embedding_bytes + size
            elif not key.endswith(("_scale", "_scale_inv")):
                params += numel
            weight_bytes += size
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    kv = 2 * text["num_hidden_layers"] * text["num_key_value_heads"] * text["head_dim"]   # K and V values per token
    fp8 = (config.get("quantization_config") or {}).get("quant_method") == "fp8"
    return Model(flop_params=params, weight_bytes=weight_bytes - embedding_bytes, fp8=fp8,
                 kv_bytes_per_token={"auto": 2 * kv, "fp8": kv})


def layout(sweep: dict) -> tuple[int, int]:
    """(GPUs per replica, replicas): vLLM's data-parallel engines, or independent servers (sweep.py --replicas)."""
    args = sweep["profile"]["args"]
    replicas = int(args.get("data-parallel-size", 1)) * int(sweep["profile"].get("replicas", 1))
    return int(args.get("tensor-parallel-size", 1)), replicas


def efficiency(sweep: dict, point: dict, model: Model, gpu_hour_usd: float = 3.0) -> dict:
    tp, replicas = layout(sweep)
    gpus = tp * replicas
    server, gpu = point["server"], point.get("gpu") or {}
    seconds = server["seconds"]
    counters, hist = server["counters"], server["histograms"]
    # vLLM counts a prefix-cache lookup each time it tries to schedule a waiting request, so while requests queue
    # for KV cache memory the same request is looked up on every step and hits outgrow prompt tokens; the hit
    # rate stays close to the truth (re-prefills after preemption are still missed)
    queries = counters["prefix_queries"]
    hit_rate = counters["prefix_hits"] / queries if queries else 0.0
    computed = counters["prompt_tokens"] * (1 - hit_rate) + counters["generation_tokens"]
    tokens = counters["prompt_tokens"] + counters["generation_tokens"]
    flops = 2 * model.flop_params * computed / seconds
    peak_flops = H100["fp8_flops" if model.fp8 else "bf16_flops"] * gpus

    kv_dtype = str(sweep["profile"]["args"].get("kv-cache-dtype", "auto"))
    # mean_kv_cache is summed over replicas; kv_cache_tokens is the capacity of one replica
    kv_bytes = (server["mean_kv_cache"] or 0) * sweep["memory"]["kv_cache_tokens"] * model.kv_bytes_per_token[kv_dtype]
    itl = hist["itl"]["mean"]
    steps_per_s = server["engine_steps_per_s"]   # summed over replicas
    bytes_per_s = steps_per_s * (model.weight_bytes + kv_bytes / replicas) if steps_per_s else None

    out = {"concurrency": point["concurrency"], "images_per_s": point["images_per_s"],
           "images_per_s_per_gpu": point["images_per_s"] / gpus, "latency_p50": point["latency"].get("p50"),
           "latency_p90": point["latency"].get("p90"),
           "prompt_tokens_per_s": server["prompt_tokens_per_s"],
           "generated_tokens_per_s": server["generation_tokens_per_s"],
           "prefix_cache_hit_rate": server["prefix_cache_hit_rate"],
           "tokens_per_step": computed / seconds / steps_per_s if steps_per_s else None,
           "ttft_p50": hist["ttft"]["p50"], "ttft_p90": hist["ttft"]["p90"],
           "itl_mean_ms": 1000 * itl if itl else None, "queue_mean": hist["queue"]["mean"],
           "prefill_mean": hist["prefill"]["mean"], "decode_mean": hist["decode"]["mean"],
           "preemptions_per_1000_requests": 1000 * counters["preemptions"] / max(counters["requests"], 1),
           "running": (server["mean_running"] or 0) / replicas,
           "kv_cache_used": (server["mean_kv_cache"] or 0) / replicas,
           "tflops": flops / 1e12, "mfu": flops / peak_flops,
           "tb_per_s": bytes_per_s / 1e12 if bytes_per_s else None,
           "mbu": bytes_per_s / (H100["hbm_bytes_per_s"] * gpus) if bytes_per_s else None,
           "usd_per_1000_images": 1000 * gpus * gpu_hour_usd / 3600 / point["images_per_s"],
           "usd_per_million_tokens": 1e6 * gpus * gpu_hour_usd / 3600 / (tokens / seconds)}
    if counters.get("spec_drafts"):
        # every forward pass of a request yields one token plus the draft tokens accepted; passes without a
        # draft (no n-gram match) count too, so this is the factor by which decode steps shrank
        accepted = counters["spec_accepted_tokens"]
        out |= {"draft_acceptance": accepted / counters["spec_draft_tokens"],
                "tokens_per_forward": counters["generation_tokens"] / (counters["generation_tokens"] - accepted)}
    if gpu.get("power_w"):
        monitored = gpu.get("gpus") or gpus   # the GPUs nvidia-smi reported, normally the job's own
        out |= {"power_w_per_gpu": gpu["power_w"] / monitored,
                "power_share_of_limit": gpu["power_w"] / (gpu.get("power_limit_w") or H100["power_w"] * monitored),
                "sm_clock_mhz": gpu.get("sm_clock_mhz"), "joules_per_image": point.get("joules_per_image"),
                "tokens_per_joule": tokens / seconds / gpu["power_w"]}
    return out


def load_sweeps(results: Path, names=None) -> dict[str, dict]:
    sweeps = {}
    for path in sorted(results.glob("sweep_*.json")):
        s = json.loads(path.read_text())
        if s["points"] and (names is None or s["name"] in names):
            sweeps[s["name"]] = s
    return sweeps


def table(rows: list[dict], columns: list[tuple[str, str, str]]) -> list[str]:
    lines = ["| " + " | ".join(c[0] for c in columns) + " |", "|" + "---:|" * len(columns)]
    for r in rows:
        cells = []
        for _, key, fmt in columns:
            v = r.get(key)
            cells.append("—" if v is None else format(v, fmt) if fmt else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


COLUMNS = [("setup", "setup", ""), ("in flight", "concurrency", "d"), ("images/s", "images_per_s", ".3f"),
           ("per GPU", "images_per_s_per_gpu", ".3f"), ("p50 s", "latency_p50", ".1f"),
           ("prompt tok/s", "prompt_tokens_per_s", ",.0f"), ("gen tok/s", "generated_tokens_per_s", ",.0f"),
           ("TTFT p50 s", "ttft_p50", ".2f"), ("ITL ms", "itl_mean_ms", ".1f"), ("tok/step", "tokens_per_step", ".0f"),
           ("running/replica", "running", ".0f"), ("KV used", "kv_cache_used", ".0%"), ("MFU", "mfu", ".1%"),
           ("MBU", "mbu", ".0%"), ("W/GPU", "power_w_per_gpu", ".0f"), ("SM MHz", "sm_clock_mhz", ".0f"),
           ("J/image", "joules_per_image", ",.0f"), ("$/1k images", "usd_per_1000_images", ".2f"),
           ("drafts accepted", "draft_acceptance", ".0%"), ("tok/forward", "tokens_per_forward", ".2f")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", help="write <out>.md and <out>.json")
    p.add_argument("--gpu-hour-usd", type=float, default=3.0)
    p.add_argument("--checkpoint", nargs="*", default=[],
                   help="model=path for checkpoints that are not in the Hugging Face cache under their name")
    args = p.parse_args()
    paths = dict(kv.split("=", 1) for kv in args.checkpoint)
    models: dict[str, Model] = {}
    rows = []
    for name, s in load_sweeps(Path(args.results)).items():
        model_name = s["profile"]["model"]
        if model_name not in models:
            models[model_name] = load_model(paths.get(model_name, model_name))
        model = models[model_name]
        for q in s["points"]:
            if q["images"] and "histograms" in q["server"]:
                r = efficiency(s, q, model, args.gpu_hour_usd)
                r["setup"] = name
                rows.append(r)
    best = [max((r for r in rows if r["setup"] == n), key=lambda r: r["images_per_s"])
            for n in dict.fromkeys(r["setup"] for r in rows)]
    single = [r for r in rows if r["concurrency"] == 1]
    lines = ["## Highest throughput per setup\n", *table(best, COLUMNS), "\n## One image at a time\n",
             *table(single, COLUMNS), "\n## All points\n", *table(rows, COLUMNS)]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
