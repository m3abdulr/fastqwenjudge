"""Tune the fused block-FP8 kernel (block_fp8.py) and time it against the FP8 linear layer of the installed vLLM, on
Qwen3-VL-32B's projections.

    python experiments/kernels/tune_block_fp8.py --checkpoint <Qwen3-VL-32B-Instruct-FP8 snapshot> \
        --out experiments/kernels/results/vllm-0.32.0/block_fp8

For every projection and token count: check each path's output against an FP32 reference, time every candidate
tile shape and K split of ours, refine warps and pipeline stages for the fastest, and time vLLM's paths the same
way (a CUDA graph over weight copies that do not fit in L2). vLLM's paths on an H100:

  vLLM 0.11  vllm        pad rows (by 4 when already a multiple of 4), quantize, CUTLASS block-scaled GEMM
             no_pad      the same, padding only when the row count is not a multiple of 4 (fixed in 0.11.1)
  vLLM 0.32  vllm        what the server dispatches: FlashInfer below 32 rows, DeepGEMM from 32 rows
             flashinfer  FlashInfer's swap-AB kernel at every row count (it quantizes the activations inside, as
                         ours does)
             deepgemm    activation quantization, then DeepGEMM, at every row count
             Both run on weights re-quantized to power-of-two block scales at load (VLLM_USE_DEEP_GEMM_E8M0, on by
             default), as the server does.

Weights are one layer of the FP8 checkpoint (--checkpoint) or random. The reference is the checkpoint's math in
FP32: activations quantized per 128-group as vLLM does, times the weights dequantized with their original scales,
so a path's error includes any re-quantization it does on top.

Writes every timing (<out>.json), the best config per shape and token count (<out>_configs.json) and per-step
totals over the 64 layers (<out>.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from importlib.metadata import version
from pathlib import Path
from statistics import median

import block_fp8
import torch
import torch.nn.functional as F
from common import BLOCK, H100, LAYERS, SHAPES, timed

TOKENS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
# checkpoint tensors fused into each projection, in vLLM's order
FUSED = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o_proj": ["self_attn.o_proj"],
         "gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]}


def vllm011_paths(w8, scale):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        apply_w8a8_block_fp8_linear,
        cutlass_scaled_mm,
        per_token_group_quant_fp8,
    )
    scale_cutlass = scale.T.contiguous()   # vLLM transposes block scales for SM90 CUTLASS at load time

    def vllm(x, w):
        return apply_w8a8_block_fp8_linear(x, w, BLOCK, scale_cutlass, cutlass_block_fp8_supported=True)

    def no_pad(x, w):
        pad = -x.shape[0] % 4
        xp = F.pad(x, (0, 0, 0, pad)) if pad else x
        q, qs = per_token_group_quant_fp8(xp, BLOCK[1], column_major_scales=True)
        out = cutlass_scaled_mm(q, w, qs, scale_cutlass, BLOCK, torch.bfloat16)
        return out[:x.shape[0]] if pad else out

    return w8, {"vllm": vllm, "no_pad": no_pad}, "no_pad"


def vllm032_paths(w8, scale):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used
    from vllm.utils.flashinfer import flashinfer_fp8_blockscale_gemm
    e8m0 = is_deep_gemm_e8m0_used()
    w8, dg_scale = deepgemm_post_process_fp8_weight_block(w8.clone(), scale.clone(), tuple(BLOCK), use_e8m0=e8m0)

    def flashinfer(x, w):
        return flashinfer_fp8_blockscale_gemm(input=x, weight=w, weight_scale=dg_scale, out_dtype=torch.bfloat16)

    def deepgemm(x, w):
        q, qs = per_token_group_quant_fp8(x, BLOCK[1], column_major_scales=True, use_ue8m0=e8m0)
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
        fp8_gemm_nt((q, qs), (w, dg_scale), out, is_deep_gemm_e8m0_used=e8m0)
        return out

    def vllm(x, w):   # the branch dynamic_flashinfer_deepgemm_blockscale_gemm takes (via torch.cond in the server)
        return flashinfer(x, w) if x.shape[0] < 32 else deepgemm(x, w)

    return w8, {"vllm": vllm, "flashinfer": flashinfer, "deepgemm": deepgemm}, "vllm"


def checkpoint_layer(path: Path, layer: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from safetensors import safe_open
    files = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def load(key):
        with safe_open(path / files[key], "pt", device="cuda") as f:
            return f.get_tensor(key)

    prefix = f"model.language_model.layers.{layer}."
    return {name: (torch.cat([load(prefix + p + ".weight") for p in parts]),
                   torch.cat([load(prefix + p + ".weight_scale_inv") for p in parts]).float())
            for name, parts in FUSED.items()}


def reference(x, w_dequant):
    """FP32 product with the activations quantized per 128-group as vLLM does (eps 1e-10, 448 = largest e4m3)."""
    g = x.float().view(x.shape[0], -1, BLOCK[1])
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    q = ((g / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).view_as(x)
    return q @ w_dequant.T


def relative_error(y, ref) -> float:
    return ((y.float() - ref).norm() / ref.norm()).item()


def tune(m: int, w8: list, scale, vllm_w8: list, paths: dict, w_dequant, rep: int) -> dict:
    n, k = w8[0].shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = reference(x, w_dequant)
    row = {"n": n, "k": k, "m": m,
           "roofline_us": 1e6 * max((n * k + scale.numel() * 4 + 2 * m * (k + n)) / H100["bytes_per_s"],
                                    2 * m * n * k / H100["fp8"])}
    for name, fn in paths.items():
        row[f"{name}_us"] = timed(lambda fn=fn: [fn(x, w) for w in vllm_w8], len(vllm_w8), rep)
        row[f"{name}_error"] = relative_error(fn(x, vllm_w8[0]), ref)

    def time_config(cfg):
        try:
            err = relative_error(block_fp8.linear(x, w8[0], scale, cfg), ref)
            if err > 1e-2:
                print(f"  {cfg}: wrong output (relative error {err:.3g}), skipped", flush=True)
                return None
            return timed(lambda: [block_fp8.linear(x, w, scale, cfg) for w in w8], len(w8), rep)
        except Exception as e:   # e.g. out of shared memory for large tiles with many stages
            print(f"  {cfg}: {type(e).__name__}", flush=True)
            return None

    timings = {}
    for cfg in block_fp8.candidates(m, n):
        timings[cfg] = time_config(cfg)
    coarse = min((c for c in timings if timings[c] is not None), key=timings.get)
    for cfg in block_fp8.refinements(coarse):
        if cfg not in timings:
            timings[cfg] = time_config(cfg)
    best = min((c for c in timings if timings[c] is not None), key=timings.get)
    row |= {"ours_us": timings[best], "config": dataclasses.asdict(best),
            "ours_error": relative_error(block_fp8.linear(x, w8[0], scale, best), ref),
            "configs_timed": sum(t is not None for t in timings.values())}
    return row


def summary(rows: list[dict], paths: list[str], deployed: str) -> list[str]:
    """Per-step linear time over the 64 layers; `deployed` is the vLLM path ours would replace where it is faster."""
    head = " | ".join(f"{p} ms" for p in paths)
    lines = [f"| tokens in step | {head} | fused kernel ms | faster of ours and {deployed} ms | speed-up over vllm "
             f"| FP8 roofline ms |", "|" + "---:|" * (len(paths) + 5)]
    for m in sorted({r["m"] for r in rows}):
        at = [r for r in rows if r["m"] == m]
        total = {key: LAYERS * sum(r[key] for r in at) / 1000
                 for key in [f"{p}_us" for p in paths] + ["ours_us", "roofline_us"]}
        either = LAYERS * sum(min(r["ours_us"], r[f"{deployed}_us"]) for r in at) / 1000
        cells = " | ".join(f"{total[f'{p}_us']:.2f}" for p in paths)
        lines.append(f"| {m} | {cells} | {total['ours_us']:.2f} | {either:.2f} | {total['vllm_us'] / either:.2f}x "
                     f"| {total['roofline_us']:.2f} |")
    lines += ["", "Relative error against the FP32 reference, median over token counts:", "",
              "| projection | " + " | ".join(paths) + " | fused kernel |", "|---|" + "---:|" * (len(paths) + 1)]
    for name, *_ in SHAPES:
        errors = {key: median(r[key] for r in rows if r["layer"] == name)
                  for key in [f"{p}_error" for p in paths] + ["ours_error"]}
        lines.append(f"| {name} | " + " | ".join(f"{errors[f'{p}_error']:.2e}" for p in paths)
                     + f" | {errors['ours_error']:.2e} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/kernels/results/block_fp8")
    p.add_argument("--checkpoint", type=Path, help="FP8 checkpoint directory; random weights if not given")
    p.add_argument("--layer", type=int, default=20, help="checkpoint layer whose weights are used")
    p.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    p.add_argument("--rep", type=int, default=100, help="ms of replays per timing")
    args = p.parse_args()
    torch.manual_seed(0)
    vllm_version = version("vllm")
    make_paths = vllm011_paths if vllm_version.startswith("0.11") else vllm032_paths
    print(torch.cuda.get_device_name(), "vLLM", vllm_version, flush=True)
    layer = checkpoint_layer(args.checkpoint, args.layer) if args.checkpoint else {}
    rows, deployed, path_names = [], None, []
    for name, n, k in SHAPES:
        if name in layer:
            w, scale = layer[name]
        else:
            w = (torch.randn(n, k, device="cuda") / k ** 0.5).to(torch.float8_e4m3fn)
            scale = torch.rand(n // BLOCK[0], k // BLOCK[1], device="cuda") * 1e-2 + 1e-3
        assert w.shape == (n, k), (name, w.shape)
        w_dequant = w.float() * scale.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)
        vllm_w, paths, deployed = make_paths(w, scale)
        path_names = list(paths)
        copies = max(1, math.ceil(256e6 / (n * k)))
        w8 = [w.clone() for _ in range(copies)]
        vllm_w8 = [vllm_w.clone() for _ in range(copies)]
        for m in args.tokens:
            row = {"layer": name, **tune(m, w8, scale, vllm_w8, paths, w_dequant, args.rep)}
            rows.append(row)
            print(json.dumps({key: round(v, 4) if isinstance(v, float) else v for key, v in row.items()}),
                  flush=True)
        del w8, vllm_w8, w_dequant
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"vllm": vllm_version, "checkpoint_layer": args.layer
                                                    if args.checkpoint else None, "rows": rows}, indent=1))
    configs = {f"{r['n']}x{r['k']}": {} for r in rows}
    for r in rows:
        configs[f"{r['n']}x{r['k']}"][r["m"]] = r["config"] | {"faster_than_vllm": r["ours_us"] < r[f"{deployed}_us"]}
    out.with_name(out.name + "_configs.json").write_text(json.dumps(configs, indent=1))
    text = "\n".join(summary(rows, path_names, deployed))
    out.with_suffix(".md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
