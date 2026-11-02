"""How much of the H100 a judging run used, and how fast the same work could go: MFU, MBU and a speed-of-light time
from an analytic model of Qwen3.8's architecture and the run's own counters (docs/serving.md, "Where the time goes,
and the speed of light").

    python experiments/qwen38/roofline.py --model Qwen/Qwen3.8-27B-FP8 --run <results>/<run> \
        --judge think_medium_strict [--profile <profiles>/<name>.json]

The model counts, per token the target model computes (prompt tokens, and every position it verifies for the
drafter: k drafts + 1 per draft round):
- FLOPs: 2 per active weight of the language model's linear layers (Flash-Next: the shared parts, the shared
  expert and 10 of 512 routed experts), the output layer for every position that needs logits (each verified and
  each drafted position, the last prompt token), attention scores (QK and AV over the context, for the model's
  attention layers and for the drafter's one, once per drafted token; Flash-Next's sparse attention over at most its
  indexer budget), the DeltaNet recurrence, the MTP drafter, and the vision tower per image (4,096 patches for a
  1024x1024 image: its linear layers and its full attention over the patches). Activations, norms and sampling are
  left out.
- Bytes, per engine step: every weight the step uses, read once (27B: all of them; Flash-Next: the experts the
  step's tokens route to, nearly all of them at 64 requests in flight), the KV cache of every running request (the
  drafter's layer once per drafted token), and the DeltaNet state: read once and written once per verified position
  (vLLM keeps one state per draft position so any number of accepted drafts can be rolled forward). Per image: the
  vision tower's weights once.
- Engine steps: vLLM's iteration counter counts what the API server receives, which can merge several engine
  steps, so a run's engine steps are estimated as its draft rounds over the mean number of requests running.

MFU = model FLOPs per second / the tensor-core peak of the precision the matmuls run in (27B: FP8; Flash-Next: BF16,
its 4-bit experts are unpacked to BF16 on H100), summed over GPUs. MBU = bytes per second / peak HBM bandwidth.
Speed of light for the run's work: the larger of all its FLOPs at peak and all its bytes at peak bandwidth, with
the batch kept at the run's cap on requests running at once (64 for the 27B; 96, or 112 for flash-fast) and the
measured draft acceptance; it is a floor no real schedule
reaches (kernels never run at peak, and prefill and decode compete for the same steps).

With --profile (experiments/qwen38/profile_gpu.py's summary at the same load), each kernel category's measured time
is set against the analytic FLOPs and bytes of the work it does, for a roofline per category.

With --stats and --kernels (a judge's .stats.json and experiments/qwen38/nsys_summary.py's summary of the same batch,
every kernel counted) and --ceilings (experiments/qwen38/ceilings.py's results), the batch's time is split class by
class (27B only): measured, at the datasheet roofline, and at the kernel ceilings measured at the batch's own shapes
(docs/roofline.md, "Where a whole batch goes").
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

# H100 SXM5 per GPU (nvidia.com/en-us/data-center/h100; the page lists tensor-core FLOPS with 2:4 sparsity, dense is
# half)
H100 = {"bf16": 989.4e12, "fp8": 1978.9e12, "hbm": 3.35e12, "watts": 700}
PATCHES_PER_IMAGE = 4096   # 1024x1024 at patch size 16 (merged 2x2 into 1,024 image tokens)


@dataclass
class Arch:
    name: str
    gpus: int
    peak_flops: float            # per GPU, the precision the matmuls run in
    linear_active: float         # weights per token in the language model's linear layers (output layer excluded)
    weight_bytes_step: float     # weight bytes one engine step at 64 requests reads, all GPUs together
    hidden: int
    vocab: int
    lm_head_bytes: float
    attn_layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    kv_bytes_per_token: float    # KV cache (and indexer keys) per context token, all attention layers
    attn_budget: int | None      # sparse attention: tokens attended per query at most
    linear_layers: int
    v_heads: int
    k_dim: int
    v_dim: int
    state_bytes: float           # one request's DeltaNet state, all layers (BF16, as served)
    mtp_active: float            # drafter weights per drafted token (output layer excluded)
    vision_params: float
    vision_bytes: float
    vision_layers: int = 27      # the vision tower's full attention over an image's patches
    vision_hidden: int = 1152
    mtp_attn_layers: int = 1     # the drafter's own attention layer, over the whole context, once per drafted token


def arch_27b() -> Arch:
    return Arch("Qwen3.8-27B (FP8)", 1, H100["fp8"], linear_active=24.35e9,
                weight_bytes_step=(22.70 + 1.18 + 0.44) * 2**30,   # FP8 linear layers, FP8 output layer, MTP layer
                hidden=5120, vocab=248320, lm_head_bytes=1.18 * 2**30,
                attn_layers=16, q_heads=24, kv_heads=4, head_dim=256, kv_bytes_per_token=16 * 2 * 4 * 256 * 1,
                attn_budget=None, linear_layers=48, v_heads=48, k_dim=128, v_dim=128,
                state_bytes=48 * 48 * 128 * 128 * 2, mtp_active=0.42e9,
                vision_params=0.46e9, vision_bytes=0.86 * 2**30)


def arch_flash() -> Arch:
    expert = 3 * 2560 * 640
    dense = (0.93 + 2.09 + 0.38 + 0.24) * 1e9            # attention, DeltaNet, dense MLP, shared expert (BF16)
    return Arch("Qwen3.8-Flash-Next (NVFP4)", 2, H100["bf16"], linear_active=dense + 10 * expert * 48,
                # all routed experts (4-bit values + FP8 scales, 63.3 GiB) are touched at 64 requests in flight,
                # plus the BF16 dense layers and the MTP head
                weight_bytes_step=(63.28 + 1.74 + 3.89 + 0.71 + 0.44 + 2.51 + 1.18) * 2**30,
                hidden=2560, vocab=248320, lm_head_bytes=1.18 * 2**30,
                attn_layers=12, q_heads=24, kv_heads=2, head_dim=256,
                kv_bytes_per_token=12 * (2 * 2 * 256 * 1 + 128 * 1 / 4), attn_budget=2048,
                linear_layers=36, v_heads=48, k_dim=128, v_dim=128, state_bytes=36 * 48 * 128 * 128 * 2,
                mtp_active=0.2e9 + 10 * expert, vision_params=0.45e9, vision_bytes=0.84 * 2**30)


ARCHS = {"Qwen/Qwen3.8-27B-FP8": arch_27b, "nvidia/Qwen3.8-Flash-Next-NVFP4": arch_flash}


def work(a: Arch, counters: dict, steps: float, k: int = 3) -> dict:
    """Analytic FLOPs and bytes of a run, by component, from vLLM's counters over the run."""
    requests = counters["requests"]
    prompt, generated = counters["prompt_tokens"], counters["generation_tokens"]
    rounds = counters.get("spec_drafts", 0.0)
    drafted = counters.get("spec_draft_tokens", 0.0)
    verified = rounds * (k + 1) if rounds else generated   # positions the target computes while decoding
    target_tokens = prompt + verified
    ctx = prompt / requests + generated / requests / 2      # mean context of a decoding request
    attended = min(ctx, a.attn_budget) if a.attn_budget else ctx
    flops = {
        "linear": 2 * a.linear_active * target_tokens,
        "output layer": 2 * a.vocab * a.hidden * (verified + drafted + requests),
        "attention": 4 * a.q_heads * a.head_dim * (a.attn_layers * (verified * attended + prompt * ctx / 2)
                                                  + a.mtp_attn_layers * drafted * attended),
        "deltanet": 6 * a.linear_layers * a.v_heads * a.k_dim * a.v_dim * target_tokens,
        "drafter": 2 * a.mtp_active * (drafted + prompt),
        "vision": 2 * a.vision_params * PATCHES_PER_IMAGE * requests,
        "vision attention": 4 * a.vision_layers * a.vision_hidden * PATCHES_PER_IMAGE**2 * requests,
    }
    running = verified / max(rounds, 1) / (k + 1) if rounds else generated / max(steps, 1)
    decoding_steps = rounds / max(running, 1) if rounds else steps
    data = {
        "weights": a.weight_bytes_step * steps,
        "kv cache": a.kv_bytes_per_token * ctx * (rounds if rounds else generated)
                    + a.kv_bytes_per_token / a.attn_layers * a.mtp_attn_layers * attended * drafted,
        "deltanet state": a.state_bytes * ((1 + (k + 1)) * rounds + 2 * requests),
        "vision": a.vision_bytes * requests,
    }
    return {"flops": flops, "bytes": data, "target_tokens": target_tokens, "context": ctx,
            "decoding_steps": decoding_steps}


def speed_of_light(a: Arch, counters: dict, cap: int = 64, k: int = 3) -> dict:
    """The run's work at peak FLOPs and peak bandwidth, with the batch at the cap and the measured acceptance."""
    rounds = counters.get("spec_drafts", 0.0)
    tokens_per_round = counters["generation_tokens"] / rounds if rounds else 1.0
    steps = counters["generation_tokens"] / tokens_per_round / cap   # every step carries `cap` decoding requests
    w = work(a, counters, steps, k)
    compute = sum(f / ((H100["bf16"] if key in BF16_WORK else a.peak_flops) * a.gpus) for key, f in w["flops"].items())
    memory = sum(w["bytes"].values()) / (H100["hbm"] * a.gpus)
    return {"steps": steps, "compute_s": compute, "memory_s": memory, "seconds": max(compute, memory),
            "bound": "compute" if compute > memory else "memory"}


def engine_steps(server: dict) -> float:
    """Engine steps of a run: draft rounds (one per running request per step) over the mean number running."""
    c = server["counters"]
    if c.get("spec_drafts") and server.get("mean_running"):
        return c["spec_drafts"] / server["mean_running"]
    return server["engine_steps_per_s"] * server["seconds"]   # API-server iterations: a lower bound


def run_cap(timing: dict, default: int = 64) -> int:
    """The server's cap on requests running at once, from the run's own record: the CLI's stats keep the server
    profile, the ladder's timings its server arguments."""
    args = (timing.get("profile") or {}).get("args") or timing.get("args") or {}
    return int(args.get("max-num-seqs") or default)


def efficiency(a: Arch, timing: dict) -> dict:
    s = timing["server"]
    seconds, counters = s["seconds"], s["counters"]
    steps = engine_steps(s)
    w = work(a, counters, steps)
    flops, data = sum(w["flops"].values()), sum(w["bytes"].values())
    gpu = timing.get("gpu") or {}
    return {"seconds": timing["wall_seconds"], "steps": steps, "steps_per_s": steps / seconds,
            "ms_per_step": 1000 * seconds / steps, "running": s["mean_running"],
            "tflops": flops / seconds / 1e12, "mfu": flops / seconds / (a.peak_flops * a.gpus),
            "tb_per_s": data / seconds / 1e12, "mbu": data / seconds / (H100["hbm"] * a.gpus),
            "flops_share": {k: v / flops for k, v in w["flops"].items()},
            "bytes_share": {k: v / data for k, v in w["bytes"].items()},
            "per_step_ms_floor": {"compute": flops / steps / (a.peak_flops * a.gpus) * 1000,
                                  "memory": data / steps / (H100["hbm"] * a.gpus) * 1000},
            "gpu_util": gpu.get("utilization"), "power_w_per_gpu": (gpu.get("power_w") or 0) / max(a.gpus, 1),
            "speed_of_light": speed_of_light(a, counters, cap=run_cap(timing)), "work": w}


# kernel categories (profile_gpu.py) and the analytic work they do
# (with FQJ_FP8_LM_HEAD the output layer runs on the FP8 matmuls; 27B's vision tower is BF16)
CATEGORY_WORK = {"gemm, fp8": (["linear", "drafter", "output layer"], ["weights"]),
                 "gemm, bf16": (["vision"], ["vision"]),
                 "attention": (["attention", "vision attention"], ["kv cache"]),
                 "deltanet": (["deltanet"], ["deltanet state"])}
BF16_WORK = {"vision", "vision attention"}   # the vision tower runs in BF16 on both models


def by_category(a: Arch, timing: dict, profile: dict) -> dict:
    """Each category's share of the profile window, scaled to the run's GPU time, against its analytic work."""
    e = efficiency(a, timing)
    gpu_seconds = timing["server"]["seconds"] * profile["gpu_busy"]
    out = {}
    for cat, (f_keys, b_keys) in CATEGORY_WORK.items():
        share = profile["categories"].get(cat, {}).get("share", 0.0)
        if not share:
            continue
        t = share * gpu_seconds
        f = sum(e["work"]["flops"][k] for k in f_keys)
        b = sum(e["work"]["bytes"][k] for k in b_keys)
        peak = H100["bf16"] if cat == "gemm, bf16" else a.peak_flops
        compute = sum(e["work"]["flops"][k] / ((H100["bf16"] if k in BF16_WORK else a.peak_flops) * a.gpus)
                      for k in f_keys)
        # the time the same work needs at the roofline: whichever of its FLOPs and its bytes takes longer at peak
        floor = max(compute, b / (H100["hbm"] * a.gpus))
        out[cat] = {"share": share, "seconds": t, "tflops": f / t / 1e12 / a.gpus, "tb_per_s": b / t / 1e12 / a.gpus,
                    "intensity": f / b if b else math.inf, "of_peak_flops": f / t / (peak * a.gpus),
                    "of_peak_bw": b / t / (H100["hbm"] * a.gpus), "roofline_seconds": floor,
                    "of_roofline": floor / t}
    return out


# the 27B's FP8 matmuls per decoder layer (ceilings.py names): 64 MLPs, 48 DeltaNet and 16 attention layers; the
# draft head is one MLP and one attention layer
# the top-k / top-p mask: vLLM's Triton kernel, or the five of kernels/topk_topp.py (one call each a step)
MASK_KERNELS = r"^_topk_topp_kernel|^_(block_max|row_bound|block_candidates|row_threshold|mask_row)$"

LAYERS_27B = {"mlp gate+up": 64, "mlp down": 64, "deltanet in (qkvz)": 48, "attention qkv+gate": 16,
              "deltanet / attention out": 64}
DRAFT_27B = {"mlp gate+up": 1, "mlp down": 1, "attention qkv+gate": 1, "deltanet / attention out": 1}


def _interp(points: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation in sorted (x, y) points, clamped at the ends."""
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def breakdown(a: Arch, stats: dict, kernels: dict, ceilings: dict, k: int = 3, chunk: int = 8192) -> dict:
    """A whole batch's GPU time by kernel class (nsys, every kernel) against its work at the datasheet roofline and
    at the measured kernel ceilings, at the batch's own shapes: decoding steps of 4 x running rows through every
    decoder layer and the output layer, k draft passes of `running` rows through the draft head and the output layer,
    prompts in `chunk`-token pieces; attention and DeltaNet bytes at the rates their kernels reach at that load."""
    server = stats["server"]
    c = server["counters"]
    running = server["mean_running"]
    steps = engine_steps(server)
    w = work(a, c, steps, k)
    f, b = w["flops"], w["bytes"]
    bw = H100["hbm"] * a.gpus
    gemm = [r for r in ceilings["gemm"]]

    def gemm_seconds(layer: str, rows: float) -> float:
        return _interp([(r["m"], r["seconds"]) for r in gemm if r["layer"] == layer], rows)

    rows = (k + 1) * running
    decode_step = sum(n * gemm_seconds(name, rows) for name, n in LAYERS_27B.items()) \
        + gemm_seconds("output layer", rows)
    draft_pass = sum(n * gemm_seconds(name, running) for name, n in DRAFT_27B.items()) \
        + gemm_seconds("output layer", running)
    big = [r for r in gemm if r["m"] == chunk and r["layer"] in LAYERS_27B]
    prefill_rate = sum(2 * r["m"] * r["n"] * r["k"] * LAYERS_27B[r["layer"]] for r in big) / \
        sum(r["seconds"] * LAYERS_27B[r["layer"]] for r in big)
    prompt_flops = 2 * a.linear_active * c["prompt_tokens"]
    gemm_ceiling = steps * (decode_step + k * draft_pass) + prompt_flops / prefill_rate

    att = [r for r in ceilings["attention"] if r["page"] == 832 and r["query_tokens"] == k + 1
           and r["splits"].startswith("cap")]
    att_rate = _interp([(r["requests"], r["tb_per_s"]) for r in att if r["context"] == 2048], running) * 1e12
    dn = ceilings["deltanet"]
    dn_rate = _interp([(r["requests"], r["tb_per_s"]) for r in dn], running) * 1e12

    # sampling: the top-k / top-p mask at the cost of one whose time does not depend on the logits
    # (kernels/topk_topp.py, timed in ceilings.py with a copy of the logits first, which is subtracted), the rest of
    # sampling (resampling for the drafts, bookkeeping) as measured
    samp = [r for r in ceilings.get("sampling", []) if r["logits"] == "zipf"]
    def samp_seconds(kernel: str) -> float:
        return _interp([(r["rows"], r["seconds"]) for r in samp if r["kernel"] == kernel], rows)
    masks = [x for x in kernels["top_kernels"] if re.search(MASK_KERNELS, x["name"])]
    calls = max((x["calls"] for x in masks), default=0)
    sampling_ceiling = (calls * (samp_seconds("fastqwenjudge three-pass") - samp_seconds("copy (floor: read + write)"))
                        + kernels["categories"].get("sampling", {}).get("ms", 0) / 1000
                        - sum(x["ms"] for x in masks) / 1000) \
        if calls and any(r["kernel"] == "fastqwenjudge three-pass" for r in samp) else None

    cats = kernels["categories"]
    t = {name: cats.get(name, {}).get("ms", 0.0) / 1000 for name in cats}
    fp8 = a.peak_flops * a.gpus
    bf16 = H100["bf16"] * a.gpus
    rows_out = [
        ("FP8 matmuls", t.get("gemm, fp8", 0),
         max((f["linear"] + f["output layer"] + f["drafter"]) / fp8, b["weights"] / bw), gemm_ceiling),
        ("DeltaNet", t.get("deltanet", 0), max(f["deltanet"] / fp8, b["deltanet state"] / bw),
         b["deltanet state"] / dn_rate),
        ("attention", t.get("attention", 0), max(f["attention"] / fp8, b["kv cache"] / bw), b["kv cache"] / att_rate),
        ("vision tower (BF16)", t.get("gemm, bf16", 0) + t.get("vision attention", 0),
         (f["vision"] + f["vision attention"]) / bf16, None),
        ("sampling", t.get("sampling", 0), 0.0, sampling_ceiling),
    ]
    modelled = {"gemm, fp8", "deltanet", "attention", "gemm, bf16", "vision attention", "sampling"}
    rest = sum(v for name, v in t.items() if name not in modelled)
    window = kernels["window_ms"] / 1000
    idle = window * (1 - kernels["gpu_busy"])
    sol = speed_of_light(a, c, cap=run_cap(stats), k=k)
    return {"rows": rows_out, "other kernels": rest, "idle": idle, "window": window,
            "wall": stats["wall_seconds"], "speed_of_light": sol, "steps": steps, "running": running,
            "decode_step_gemm_ms": 1000 * decode_step, "draft_pass_gemm_ms": 1000 * draft_pass,
            "prefill_tflops": prefill_rate / 1e12, "attention_tb_per_s": att_rate / 1e12,
            "deltanet_tb_per_s": dn_rate / 1e12}


def print_breakdown(br: dict):
    print(f"  batch {br['wall']:.0f} s (profiled window {br['window']:.0f} s); speed of light "
          f"{br['speed_of_light']['seconds']:.0f} s; {br['steps']:.0f} engine steps at {br['running']:.0f} running")
    print(f"  ceilings at this batch's shapes: matmuls {br['decode_step_gemm_ms']:.1f} ms per decoding step, "
          f"{br['draft_pass_gemm_ms']:.2f} ms per draft pass, prompts at {br['prefill_tflops']:.0f} TFLOP/s; "
          f"attention {br['attention_tb_per_s']:.2f} TB/s, DeltaNet {br['deltanet_tb_per_s']:.2f} TB/s")
    print(f"  {'class':22s} {'measured':>9s} {'roofline':>9s} {'ceiling':>9s}")
    for name, measured, roof, ceiling in br["rows"]:
        print(f"  {name:22s} {measured:8.1f}s {roof:8.1f}s " + (f"{ceiling:8.1f}s" if ceiling is not None else
                                                                 f"{'-':>9s}"))
    print(f"  {'other kernels':22s} {br['other kernels']:8.1f}s")
    print(f"  {'GPU idle':22s} {br['idle']:8.1f}s")


def latest_timing(folder: Path, judge: str) -> dict:
    for f in sorted(folder.glob("timings_*.json"), reverse=True):
        t = json.loads(f.read_text())
        if judge in t:
            return t[judge]
    raise SystemExit(f"no timings for {judge} in {folder}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(ARCHS))
    p.add_argument("--run", help="a run folder with timings_*.json")
    p.add_argument("--judge")
    p.add_argument("--profile", help="profile_gpu.py summary at the same load")
    p.add_argument("--stats", help="a judge's .stats.json (instead of --run / --judge)")
    p.add_argument("--kernels", help="nsys_summary.py's summary of the same batch (with --stats)")
    p.add_argument("--ceilings", help="ceilings.py's results (with --kernels)")
    args = p.parse_args()
    a = ARCHS[args.model]()
    if args.stats:
        timing = json.loads(Path(args.stats).read_text())
        timing.setdefault("gpu", None)
    else:
        timing = latest_timing(Path(args.run), args.judge)
    e = efficiency(a, timing)
    sol = e["speed_of_light"]
    print(f"{a.name}, {args.judge}: {e['seconds']:.0f} s, {e['ms_per_step']:.1f} ms per step at {e['running']:.0f} "
          f"running; {e['tflops']:.0f} TFLOP/s (MFU {100 * e['mfu']:.0f}%), {e['tb_per_s']:.2f} TB/s "
          f"(MBU {100 * e['mbu']:.0f}%); per-step floor {e['per_step_ms_floor']['compute']:.1f} ms compute / "
          f"{e['per_step_ms_floor']['memory']:.1f} ms memory")
    print(f"  speed of light {sol['seconds']:.0f} s ({sol['bound']}-bound: {sol['compute_s']:.0f} s of FLOPs, "
          f"{sol['memory_s']:.0f} s of bytes); measured / speed of light {e['seconds'] / sol['seconds']:.1f}x")
    print("  FLOPs: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in e["flops_share"].items()))
    print("  bytes: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in e["bytes_share"].items()))
    if args.kernels and args.ceilings:
        print_breakdown(breakdown(a, timing, json.loads(Path(args.kernels).read_text()),
                                  json.loads(Path(args.ceilings).read_text())))
    if args.profile:
        for cat, c in by_category(a, timing, json.loads(Path(args.profile).read_text())).items():
            print(f"  {cat:12s} {100 * c['share']:4.0f}% of GPU time ({c['seconds']:.0f} s): "
                  f"{c['tflops']:5.0f} TFLOP/s ({100 * c['of_peak_flops']:.0f}% of peak), {c['tb_per_s']:.2f} TB/s "
                  f"({100 * c['of_peak_bw']:.0f}% of peak), {c['intensity']:.0f} FLOP/byte; at the roofline "
                  f"{c['roofline_seconds']:.0f} s ({100 * c['of_roofline']:.0f}% of roofline speed)")


if __name__ == "__main__":
    main()
