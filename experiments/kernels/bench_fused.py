"""Time and check the fused norm/SiLU + FP8 kernels (fastqwenjudge/kernels/fused_quant.py) against what vLLM 0.32
runs, on Qwen3-VL-32B's shapes and layer-20 weights.

    python experiments/kernels/bench_fused.py --checkpoint <Qwen3-VL-32B-Instruct-FP8 snapshot> \
        --out experiments/kernels/results/vllm-0.32.0/fused_quant

Three paths at each of the two sites, per token count:
  vllm        what the server runs: torch.compile's residual add + RMSNorm (or SiLU x up), then vLLM's per-128-group
              FP8 conversion with power-of-two scales (the input DeepGEMM gets from 32 rows up; below 32 rows FlashInfer
              converts inside its own matmul, so there is nothing to fuse)
  vllm_fused  vLLM's fused CUDA kernels (rms_norm_per_block_quant, silu_and_mul_per_block_quant), which its fusion
              passes use when custom ops are on
  ours        fastqwenjudge/kernels/fused_quant.py

Checks against the vllm path: identical FP8 bytes, scales and residual, and the relative difference of the next
matmul's output (layer 20's QKV after the norm, its down projection after SiLU) when DeepGEMM is fed each path's
FP8 activations. Times each path in a CUDA graph, then projects the saving per engine step over 64 layers at the
step sizes the server runs at 64 and 256 images in flight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from common import BLOCK, H100, LAYERS, timed
from tune_block_fp8 import checkpoint_layer

from fastqwenjudge.kernels import fused_quant

TOKENS = [32, 64, 128, 224, 256, 512, 1024, 2048, 4096, 8192]
HIDDEN, INTER, EPS = 5120, 25600, 1e-6
# tokens per engine step and step time, FlashInfer + n-gram sweep (sweep_tuned_fp8_fp8kv_flashinfer_ngram_dynamic):
# (computed prompt + generated tokens/s) / engine steps/s
STEPS = {"64 in flight": (141, 1000 / 39.2), "256 in flight": (587, 1000 / 15.0)}


def add_rms_norm_ref(x, residual, weight, eps):
    """vllm/ir/ops/layernorm.py fused_add_rms_norm, the function torch.compile turns into one Triton kernel."""
    h = x.float() + residual.float()
    y = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(weight.dtype) * weight
    return y.to(x.dtype), h.to(x.dtype)


def silu_mul_ref(gate_up):
    d = gate_up.shape[-1] // 2
    return F.silu(gate_up[:, :d]) * gate_up[:, d:]


def deepgemm_matmul(w8, scale):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used
    e8m0 = is_deep_gemm_e8m0_used()
    w, ws = deepgemm_post_process_fp8_weight_block(w8.clone(), scale.clone(), tuple(BLOCK), use_e8m0=e8m0)

    def mm(q, s):
        out = torch.empty(q.shape[0], w.shape[0], dtype=torch.bfloat16, device=q.device)
        fp8_gemm_nt((q, s), (w, ws), out, is_deep_gemm_e8m0_used=e8m0)
        return out
    return mm


def dequantized(w8, scale):
    """FP32 weights from the checkpoint's FP8 blocks and their original (not power-of-two) scales."""
    s = scale.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)
    return w8.float() * s[:w8.shape[0], :w8.shape[1]]


def same(a, b) -> float:
    return (a.view(torch.uint8) == b.view(torch.uint8)).float().mean().item() if a.dtype == torch.float8_e4m3fn \
        else (a == b).float().mean().item()


def rel(y, ref) -> float:
    return ((y.float() - ref.float()).norm() / ref.float().norm()).item()


def fmt(diff) -> str:
    return "failed" if diff is None else f"{diff:.1e}"


def power_of_two(s) -> float:
    return (torch.frexp(s)[0].abs() == 0.5).float().mean().item()


def bench(m, ckpt_layer, norm_w, rep):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    fp8 = torch.float8_e4m3fn
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", generator=gen).bfloat16()
    res = (2 * torch.randn(m, HIDDEN, device="cuda", generator=gen)).bfloat16()
    gu = torch.cat([1.5 * torch.randn(m, INTER, device="cuda", generator=gen),
                    torch.randn(m, INTER, device="cuda", generator=gen)], 1).bfloat16()
    norm_c, silu_c = torch.compile(add_rms_norm_ref), torch.compile(silu_mul_ref)
    qkv, down = deepgemm_matmul(*ckpt_layer["qkv"]), deepgemm_matmul(*ckpt_layer["down"])
    # what every path approximates: the next matmul on the activations computed in FP64, unquantized
    h = x.double() + res.double()
    norm_true = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * norm_w.double()).float()
    g, u = gu[:, :INTER].double(), gu[:, INTER:].double()
    truth = {"norm": norm_true @ dequantized(*ckpt_layer["qkv"]).T,
             "silu": (g * torch.sigmoid(g) * u).float() @ dequantized(*ckpt_layer["down"]).T}

    def quant(y):
        return per_token_group_quant_fp8(y, BLOCK[1], column_major_scales=True, use_ue8m0=True)

    def norm_vllm():
        y, r = norm_c(x, res, norm_w, EPS)
        return (*quant(y), r)

    scratch = res.clone()

    def norm_vllm_fused(residual=scratch):   # updates the residual in place
        q, s = ops.rms_norm_per_block_quant(x, norm_w, EPS, fp8, [1, BLOCK[1]], residual=residual,
                                            is_scale_transposed=True)
        return q, s, residual

    def norm_ours():
        return fused_quant.add_rms_norm_fp8(x, res, norm_w, EPS)

    def silu_vllm():
        return quant(silu_c(gu))

    def silu_vllm_fused():
        return ops.silu_and_mul_per_block_quant(gu, BLOCK[1], fp8, is_scale_transposed=True)

    def silu_ours():
        return fused_quant.silu_mul_fp8(gu)

    # site -> path -> (call for the checks, call for timing); the in-place fused norm is checked on a fresh residual
    sites = {"norm": ({"vllm": (norm_vllm, norm_vllm),
                       "vllm_fused": (lambda: norm_vllm_fused(res.clone()), norm_vllm_fused),
                       "ours": (norm_ours, norm_ours)}, qkv),
             "silu": ({"vllm": (silu_vllm, silu_vllm), "vllm_fused": (silu_vllm_fused, silu_vllm_fused),
                       "ours": (silu_ours, silu_ours)}, down)}
    rows = []
    for site, (paths, mm) in sites.items():
        out = {name: check() for name, (check, _) in paths.items()}
        base = out["vllm"]
        y_base = mm(base[0], base[1])
        row = {"site": site, "tokens": m}
        for name, o in out.items():
            try:
                y = mm(o[0], o[1])
                diff, err = rel(y, y_base), rel(y, truth[site])
            except RuntimeError as e:   # DeepGEMM refuses scales it cannot use
                diff = err = None
                print(f"{site} {name} m={m}: next matmul failed: {e}", flush=True)
            row[name] = {"us": timed(paths[name][1], 1, rep), "fp8_same": same(o[0], base[0]),
                         "scales_same": same(o[1], base[1]), "scales_pow2": power_of_two(o[1]),
                         "next_matmul_rel_diff": diff, "next_matmul_err_vs_fp64": err}
            if site == "norm":
                row[name]["residual_same"] = same(o[2], base[2])
        # bytes ours moves: the norm reads x and the residual and writes the new residual and FP8 (7 bytes per
        # value); SiLU reads gate and up and writes FP8 (5 bytes per output value)
        moved = m * (7 * HIDDEN if site == "norm" else 5 * INTER)
        row["ours"]["bandwidth_share"] = moved / (row["ours"]["us"] * 1e-6) / H100["bytes_per_s"]
        rows.append(row)
    return rows


def small_rows(norm_w, rep) -> list[str]:
    """Below 32 rows the plugin's kernels write BF16 for FlashInfer: same values as torch.compile's, in one kernel."""
    lines = ["| tokens | norm same | norm us: vllm / ours | SiLU same | SiLU us: vllm / ours |",
             "|---:|---:|---:|---:|---:|"]
    norm_c, silu_c = torch.compile(add_rms_norm_ref), torch.compile(silu_mul_ref)
    for m in (1, 4, 8, 16, 31):
        gen = torch.Generator(device="cuda").manual_seed(m)
        x = torch.randn(m, HIDDEN, device="cuda", generator=gen).bfloat16()
        res = (2 * torch.randn(m, HIDDEN, device="cuda", generator=gen)).bfloat16()
        gu = torch.cat([1.5 * torch.randn(m, INTER, device="cuda", generator=gen),
                        torch.randn(m, INTER, device="cuda", generator=gen)], 1).bfloat16()
        (y, r), (y2, r2) = norm_c(x, res, norm_w, EPS), fused_quant.add_rms_norm(x, res, norm_w, EPS)
        norm_same = min(same(y, y2), same(r, r2))
        silu_same = same(silu_c(gu), fused_quant.silu_mul(gu))
        t = [timed(f, 1, rep) for f in (lambda x=x, res=res: norm_c(x, res, norm_w, EPS),
                                        lambda x=x, res=res: fused_quant.add_rms_norm(x, res, norm_w, EPS),
                                        lambda gu=gu: silu_c(gu), lambda gu=gu: fused_quant.silu_mul(gu))]
        lines.append(f"| {m} | {100 * norm_same:.2f}% | {t[0]:.1f} / {t[1]:.1f} | {100 * silu_same:.2f}% | "
                     f"{t[2]:.1f} / {t[3]:.1f} |")
    return lines


def show_inductor(norm_w, m=256):
    """Print the Triton kernel torch.compile writes for the reference norm, and how often its BF16 output equals the
    same function run eagerly (the compiled kernel drops the cast to BF16 before the weight multiply)."""
    from torch._inductor.utils import run_and_get_code
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", generator=gen).bfloat16()
    res = (2 * torch.randn(m, HIDDEN, device="cuda", generator=gen)).bfloat16()
    (y, _), code = run_and_get_code(torch.compile(add_rms_norm_ref), x, res, norm_w, EPS)
    for source in code:
        start = source.find("@triton.jit")
        print(source[start:source.find("''', device_str", start)] if start >= 0 else source[:3000])
    print(f"compiled vs eager BF16 output identical: {100 * same(y, add_rms_norm_ref(x, res, norm_w, EPS)[0]):.2f}%")


def project(rows) -> list[str]:
    def at(site, path, tokens):   # linear interpolation between measured token counts
        pts = sorted((r["tokens"], r[path]["us"]) for r in rows if r["site"] == site)
        for (m0, t0), (m1, t1) in zip(pts, pts[1:], strict=False):
            if m0 <= tokens <= m1:
                return t0 + (t1 - t0) * (tokens - m0) / (m1 - m0)
        return pts[-1][1]
    lines = ["| load | tokens/step | step ms | " + " | ".join(f"{p} saves per step" for p in ("vllm_fused", "ours")) +
             " |", "|---|---:|---:|---:|---:|"]
    for load, (tokens, step_ms) in STEPS.items():
        cells = []
        for path in ("vllm_fused", "ours"):
            # two norm sites (before attention, before the MLP) and one SiLU site per layer
            saved = LAYERS * sum(k * (at(site, "vllm", tokens) - at(site, path, tokens))
                                 for site, k in (("norm", 2), ("silu", 1))) / 1000
            cells.append(f"{saved:.2f} ms ({100 * saved / step_ms:+.1f}%)")
        lines.append(f"| {load} | {tokens} | {step_ms:.1f} | " + " | ".join(cells) + " |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rep", type=int, default=200)
    p.add_argument("--out", required=True)
    p.add_argument("--show-inductor", action="store_true", help="print torch.compile's norm kernel, then stop")
    args = p.parse_args()
    from safetensors import safe_open
    path = Path(args.checkpoint)
    files = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.post_attention_layernorm.weight"
    with safe_open(path / files[key], "pt", device="cuda") as f:
        norm_w = f.get_tensor(key)
    if args.show_inductor:
        show_inductor(norm_w)
        return
    layer = checkpoint_layer(path, args.layer)
    rows = []
    for m in TOKENS:
        rows += bench(m, layer, norm_w, args.rep)
        for r in rows[-2:]:
            print(f"{r['site']:4} m={m:5}: " + "  ".join(
                f"{n} {r[n]['us']:7.1f} us (fp8 same {100 * r[n]['fp8_same']:5.1f}%, next matmul diff "
                f"{fmt(r[n]['next_matmul_rel_diff'])})" for n in ("vllm", "vllm_fused", "ours")), flush=True)
    lines = ["| site | tokens | vllm us | vllm_fused us | ours us | ours: FP8 same | ours: next-matmul diff | "
             "vllm_fused: FP8 same | vllm_fused: next-matmul diff | vllm_fused scales power of two | "
             "next-matmul error vs FP64: vllm / vllm_fused / ours |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        o, f = r["ours"], r["vllm_fused"]
        lines.append(f"| {r['site']} | {r['tokens']} | {r['vllm']['us']:.1f} | {f['us']:.1f} | {o['us']:.1f} | "
                     f"{100 * o['fp8_same']:.2f}% | {fmt(o['next_matmul_rel_diff'])} | {100 * f['fp8_same']:.2f}% | "
                     f"{fmt(f['next_matmul_rel_diff'])} | {100 * f['scales_pow2']:.0f}% | "
                     + " / ".join(fmt(r[n]["next_matmul_err_vs_fp64"]) for n in ("vllm", "vllm_fused", "ours")) + " |")
    lines += ["", "Projected saving per engine step (64 layers, two norm sites and one SiLU site each):", ""]
    lines += project(rows)
    lines += ["", "Under 32 rows (BF16 out, for FlashInfer's matmul):", ""] + small_rows(norm_w, args.rep)
    print("\n".join(lines))
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
