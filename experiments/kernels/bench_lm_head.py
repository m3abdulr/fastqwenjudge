"""The output layer as served (BF16) against the plugin's block-FP8 version (fastqwenjudge/kernels/qwen3vl.py,
FQJ_FP8_LM_HEAD=1): time per call and how far the logits move, on the checkpoint's own output-layer weights.

    python experiments/kernels/bench_lm_head.py --checkpoint <Qwen3-VL-32B-Instruct-FP8 snapshot> \
        --out experiments/kernels/results/vllm-0.32.0/lm_head

Inputs are random rows with the RMS of the final norm's output (unit RMS times its weight), at the row counts the
server sees: one row per running request, plus the guessed tokens while it speculates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from common import timed

from fastqwenjudge.kernels import qwen3vl

TOKENS = [1, 4, 16, 31, 32, 64, 128, 256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rep", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from safetensors import safe_open
    path = Path(args.checkpoint)
    files = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def load(key):
        with safe_open(path / files[key], "pt", device="cuda") as f:
            return f.get_tensor(key)
    w, norm_w = load("lm_head.weight"), load("model.language_model.norm.weight")
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    head = qwen3vl._Fp8LMHead(layer)
    rows, lines = [], ["| rows | BF16 us | FP8 us | speedup | logits rel. error | same top token |",
                       "|---:|---:|---:|---:|---:|---:|"]
    for m in TOKENS:
        gen = torch.Generator(device="cuda").manual_seed(m)
        x = (torch.randn(m, w.shape[1], device="cuda", generator=gen) * norm_w.float()).bfloat16()
        ref, out = F.linear(x, w), head.apply(layer, x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        t_bf16 = timed(lambda x=x: F.linear(x, w), 1, args.rep)
        t_fp8 = timed(lambda x=x: head.apply(layer, x), 1, args.rep)
        rows.append({"rows": m, "bf16_us": t_bf16, "fp8_us": t_fp8, "rel_error": rel, "same_top": top1})
        lines.append(f"| {m} | {t_bf16:.1f} | {t_fp8:.1f} | {t_bf16 / t_fp8:.2f}x | {rel:.1e} | {100 * top1:.1f}% |")
        print(lines[-1], flush=True)
    print("\n".join(lines))
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
