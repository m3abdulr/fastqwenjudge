"""The judge model's linear layers in isolation: time per call against the H100 roofline, for every batch size the
server runs (1 token per request while decoding, up to 8,192 tokens in a prefill step).

    python experiments/kernels/bench_linear.py --out experiments/kernels/results/bench_linear

Shapes are Qwen3-VL-32B's per-layer projections (64 layers each): fused QKV, attention output, fused gate/up and
down, plus the BF16 output head (once per step). Each FP8 layer is timed as vLLM 0.11 runs it on an H100 and in
parts:

  bf16         torch linear (cuBLAS), the BF16 checkpoint
  fp8_vllm     vLLM's block-FP8 linear: pad rows to a multiple of 4, quantize activations per 128-group,
               CUTLASS block-scaled GEMM, slice the padding off
  fp8_pad      the padding copy alone (vLLM pads 4 rows even when the row count is already a multiple of 4)
  fp8_quant    the activation quantization alone
  fp8_gemm     the CUTLASS GEMM alone, on inputs quantized beforehand
  fp8_triton   vLLM's Triton block-FP8 kernel with its default tile config (no config ships for these shapes)

Every timing replays a CUDA graph (as the server does while decoding) over enough copies of the weight that
nothing stays in the 50 MB L2 cache between calls, like the 64 different layers of a real step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from common import BLOCK, H100, HEAD, LAYERS, SHAPES, timed
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_w8a8_block_fp8_linear,
    cutlass_scaled_mm,
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)

BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192]


def bench_shape(name: str, n: int, k: int, batches: list[int], fp8: bool) -> list[dict]:
    copies_bf16 = max(1, math.ceil(256e6 / (n * k * 2)))
    w16 = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5 for _ in range(copies_bf16)]
    w8 = scale = None
    if fp8:
        copies_fp8 = max(1, math.ceil(256e6 / (n * k)))
        w8 = [w16[i % copies_bf16].to(torch.float8_e4m3fn) for i in range(copies_fp8)]
        scale = torch.rand(n // BLOCK[0], k // BLOCK[1], device="cuda", dtype=torch.float32) * 1e-2 + 1e-3
    rows = []
    for m in batches:
        row = {"layer": name, "n": n, "k": k, "m": m, **bench_batch(m, n, k, w16, w8, scale)}
        rows.append(row)
        print(json.dumps({key: round(v, 2) if isinstance(v, float) else v for key, v in row.items()}), flush=True)
    return rows


def bench_batch(m: int, n: int, k: int, w16: list, w8: list | None, scale) -> dict:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    flops = 2 * m * n * k
    act = m * k * 2 + m * n * 2
    row = {"bf16_us": timed(lambda: [F.linear(x, w) for w in w16], len(w16)),
           "bf16_roofline_us": 1e6 * max((n * k * 2 + act) / H100["bytes_per_s"], flops / H100["bf16"])}
    if w8 is None:
        return row
    scale_cutlass = scale.T.contiguous()   # vLLM transposes block scales for SM90 CUTLASS at load time
    pad = 4 - m % 4                        # as vLLM 0.11 computes it: 4 rows when m is already a multiple of 4
    xp = F.pad(x, (0, 0, 0, pad))
    q, qs = per_token_group_quant_fp8(xp, BLOCK[1], column_major_scales=True)
    q_rows, qs_rows = per_token_group_quant_fp8(x, BLOCK[1], column_major_scales=False)
    row["fp8_vllm_us"] = timed(lambda: [apply_w8a8_block_fp8_linear(x, w, BLOCK, scale_cutlass,
                                                                    cutlass_block_fp8_supported=True)
                                        for w in w8], len(w8))
    row["fp8_pad_us"] = timed(lambda: F.pad(x, (0, 0, 0, pad)), 1)
    row["fp8_quant_us"] = timed(lambda: per_token_group_quant_fp8(xp, BLOCK[1], column_major_scales=True), 1)
    row["fp8_gemm_us"] = timed(lambda: [cutlass_scaled_mm(q, w, qs, scale_cutlass, BLOCK, torch.bfloat16)
                                        for w in w8], len(w8))
    row["fp8_triton_us"] = timed(lambda: [w8a8_block_fp8_matmul(q_rows, w, qs_rows, scale, BLOCK, torch.bfloat16)
                                          for w in w8], len(w8))
    fp8_bytes = n * k + scale.numel() * 4
    row["fp8_roofline_us"] = 1e6 * max((fp8_bytes + act) / H100["bytes_per_s"], flops / H100["fp8"])
    return row


def check_numerics():
    """The three FP8 paths agree with each other and with a BF16 reference on dequantized weights."""
    n, k, m = 5120, 8192, 37
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
    w8 = w.to(torch.float8_e4m3fn)
    scale = torch.ones(n // 128, k // 128, device="cuda", dtype=torch.float32)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w8.to(torch.bfloat16))
    a = apply_w8a8_block_fp8_linear(x, w8, BLOCK, scale.T.contiguous(), cutlass_block_fp8_supported=True)
    b = apply_w8a8_block_fp8_linear(x, w8, BLOCK, scale, cutlass_block_fp8_supported=False)
    rel = lambda y: ((y.float() - ref.float()).norm() / ref.float().norm()).item()  # noqa: E731
    print(f"relative error vs BF16 on the FP8 weights: cutlass {rel(a):.4f}, triton {rel(b):.4f}")
    assert rel(a) < 0.05 and rel(b) < 0.05


def summary(rows: list[dict]) -> list[str]:
    """Linear-layer time per engine step (64 layers of the four projections, plus the output head once)."""
    lines = ["| tokens in step | BF16 ms | FP8 as vLLM runs it ms | of which pad + quant ms | FP8 GEMM only ms "
             "| FP8 Triton ms | FP8 roofline ms | BF16 roofline ms |", "|" + "---:|" * 8]
    for m in sorted({r["m"] for r in rows}):
        layer = [r for r in rows if r["m"] == m and r["layer"] != HEAD[0]]
        head = next((r for r in rows if r["m"] == m and r["layer"] == HEAD[0]), None)
        head_us = head["bf16_us"] if head else 0
        s = {key: LAYERS * sum(r[key] for r in layer) / 1000 for key in (
            "bf16_us", "fp8_vllm_us", "fp8_pad_us", "fp8_quant_us", "fp8_gemm_us", "fp8_triton_us",
            "fp8_roofline_us", "bf16_roofline_us")}
        lines.append(f"| {m} | {s['bf16_us'] + head_us / 1000:.2f} | {s['fp8_vllm_us'] + head_us / 1000:.2f} | "
                     f"{s['fp8_pad_us'] + s['fp8_quant_us']:.2f} | {s['fp8_gemm_us']:.2f} | "
                     f"{s['fp8_triton_us']:.2f} | {s['fp8_roofline_us']:.2f} | {s['bf16_roofline_us']:.2f} |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/kernels/results/bench_linear")
    p.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    args = p.parse_args()
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(), flush=True)
    check_numerics()
    rows = []
    for name, n, k in SHAPES:
        rows += bench_shape(name, n, k, args.batches, fp8=True)
    # the output head runs once per step on the last token of each request; it stays BF16 in the FP8 checkpoint
    rows += bench_shape(*HEAD, [m for m in args.batches if m <= 512], fp8=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    text = "\n".join(summary(rows))
    out.with_suffix(".md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
