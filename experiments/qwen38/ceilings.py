"""What each kernel class can reach on one H100 at Qwen3.8-27B's shapes, measured in isolation with the same kernels
vLLM 0.32 runs in the server (docs/roofline.md, "Check 2"). The roofline (roofline.py) sets the
run's work against the datasheet peaks; this sets each kernel against what it achieves on its own at the shapes the
server gives it, so the gap between a run and its speed of light splits into what the kernels cannot do at these
shapes and what the run loses around them.

    python experiments/qwen38/ceilings.py --out <results>/ceilings_<node>.json [--only hbm,gemm,attention,...]

- hbm: a read-only Triton kernel and a device copy over 4 GiB (the achievable bandwidth).
- gemm: the 27B's block-FP8 linear layers (and the FP8 output layer) at 1 to 16,384 rows, on the matmuls vLLM
  picks (FlashInfer's BF16-input kernel below 32 rows, DeepGEMM with FP8 activations from 32), plus the activation
  quantization DeepGEMM needs. Weights rotate through enough copies that none is served from L2.
- attention: FlashAttention 3 over the paged FP8 KV cache as vLLM lays it out (pages of 832 tokens, K and V
  interleaved per head), 24 query heads on 4 KV heads of 256, 1 or 4 query tokens per request (4 = one decoding
  step with 3 drafts), with vLLM's CUDA-graph split cap (32) and with FA3's own heuristic; also smaller pages.
- attention_flashinfer: FlashInfer's paged prefill (FA2 and FA3 backends) and decode kernels at the same shapes.
- deltanet: vLLM's fused CUDA kernel for a speculative decoding step (reads each request's 16-bit state, writes one
  state per verified position).
- sampling: vLLM's Triton top-k / top-p masking (top_k 20, top_p 0.95, the quality modes' settings) over the
  248,320-token vocabulary on logits of several shapes (including the answer's JSON grammar, which sets the
  tokens it does not allow to -inf), against FlashInfer's top-k mask, torch.topk and this package's three-pass mask
  (kernels/topk_topp.py). The two masking kernels are timed with a copy of the logits first, which "copy" times
  alone.

Every case is captured in a CUDA graph and replayed (launch overhead excluded, as in the server's decode steps),
timed with CUDA events, and reported as the median of several replays, with the SM clock and board power sampled
over the case.
"""
# ruff: noqa: B023  (each closure below is timed inside the loop that defines it)
from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path

import torch

H100 = {"bf16": 989.4e12, "fp8": 1978.9e12, "hbm": 3.35e12}
L2_BYTES = 50 * 2**20
HIDDEN = 5120
# (name, N, K) of the 27B's FP8 linear layers; the output layer is FP8 with FQJ_FP8_LM_HEAD
GEMMS = [("mlp gate+up", 34816, 5120), ("mlp down", 5120, 17408), ("deltanet in (qkvz)", 16384, 5120),
         ("attention qkv+gate", 14336, 5120), ("deltanet / attention out", 5120, 6144),
         ("output layer", 248320, 5120)]
ROWS = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


class Clocks:
    """SM clock (MHz) and board power (W) sampled every 20 ms while a case runs."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        self.samples, self.on = [], False

    def __enter__(self):
        self.samples, self.on = [], True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        n = self.nvml
        while self.on:
            self.samples.append((n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_SM),
                                 n.nvmlDeviceGetPowerUsage(self.h) / 1000))
            time.sleep(0.02)

    def __exit__(self, *exc):
        self.on = False
        self.t.join()

    def summary(self):
        if not self.samples:
            return {}
        sm, w = zip(*self.samples, strict=True)
        return {"sm_mhz": round(statistics.mean(sm)), "sm_mhz_min": min(sm), "watts": round(statistics.mean(w))}


def timed(fn, inner: int = 20, reps: int = 7) -> float:
    """Seconds per call: `inner` calls captured in one CUDA graph, replayed `reps` times, median."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
        run = graph.replay
    except Exception as e:   # not capturable: eager loop (adds launch overhead; reported)
        print(f"    (no graph: {type(e).__name__}: {str(e)[:80]})", flush=True)

        def run():
            for _ in range(inner):
                fn()
    run()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / 1000 / inner)
    return statistics.median(out)


def case(clocks, fn, **kw):
    with clocks:
        s = timed(fn, **kw)
    return s, clocks.summary()


# ---------------------------------------------------------------------------------------------------------- hbm
def bench_hbm(clocks):
    import triton
    import triton.language as tl

    @triton.jit
    def read_kernel(x, out, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        base = pid * BLOCK * ITERS
        for i in range(ITERS):
            offs = base + i * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(x + offs, mask=offs < n, other=0.0).to(tl.float32)
        tl.store(out + pid, tl.sum(acc))

    n = 2 * 2**30   # 4 GiB of BF16
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    rows = []
    for block, iters in [(4096, 16), (8192, 8), (8192, 32)]:
        grid = triton.cdiv(n, block * iters)
        out = torch.empty(grid, dtype=torch.float32, device="cuda")
        s, c = case(clocks, lambda: read_kernel[(grid,)](x, out, n, BLOCK=block, ITERS=iters), inner=5)
        rows.append({"kernel": f"triton read {block}x{iters}", "seconds": s, "tb_per_s": 2 * n / s / 1e12, **c})
    s, c = case(clocks, lambda: y.copy_(x), inner=5)
    rows.append({"kernel": "copy (read + write)", "seconds": s, "tb_per_s": 2 * 2 * n / s / 1e12, **c})
    for r in rows:
        r["of_peak"] = r["tb_per_s"] * 1e12 / H100["hbm"]
        print(f"  hbm {r['kernel']:22s} {r['tb_per_s']:.2f} TB/s ({100 * r['of_peak']:.0f}% of 3.35) "
              f"{r.get('sm_mhz')} MHz", flush=True)
    x = y = None
    torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------------------------------------- gemm
def bench_gemm(clocks, rows=ROWS):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used, per_block_cast_to_fp8
    from vllm.utils.flashinfer import flashinfer_fp8_blockscale_gemm
    e8m0 = is_deep_gemm_e8m0_used()
    out_rows = []
    for name, n_out, k in GEMMS:
        w_bytes = n_out * k
        copies = max(1, math.ceil(3 * L2_BYTES / w_bytes))
        weights = []
        for _ in range(copies):
            w = torch.randn(n_out, k, device="cuda", dtype=torch.bfloat16) / k**0.5
            w8, sc = per_block_cast_to_fp8(w, [128, 128], use_ue8m0=False)
            weights.append(deepgemm_post_process_fp8_weight_block(w8, sc, (128, 128), use_e8m0=e8m0))
            del w
        for m in rows:
            if name == "output layer" and m > 1024:   # it sees one row per verified or drafted position
                continue
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n_out, device="cuda", dtype=torch.bfloat16)
            it = {"i": 0}
            if m < 32:   # vLLM: FlashInfer's kernel takes the BF16 activations directly
                def fn():
                    w8, sc = weights[it["i"] % copies]
                    it["i"] += 1
                    flashinfer_fp8_blockscale_gemm(input=x, weight=w8, weight_scale=sc, out_dtype=torch.bfloat16)
                quant_s = 0.0
                kernel = "flashinfer bf16-in"
            else:
                q, s = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=e8m0)

                def fn():
                    w8, sc = weights[it["i"] % copies]
                    it["i"] += 1
                    fp8_gemm_nt((q, s), (w8, sc), out, is_deep_gemm_e8m0_used=e8m0)
                quant_s = timed(lambda: per_token_group_quant_fp8(x, 128, column_major_scales=True,
                                                                  use_ue8m0=e8m0))
                kernel = "deepgemm"
            try:
                sec, c = case(clocks, fn, inner=max(copies, 10) if m <= 4096 else copies)
            except Exception as e:
                print(f"  gemm {name} m={m}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            flops = 2 * m * n_out * k
            data = w_bytes + m * k * (2 if m < 32 else 1) + 2 * m * n_out
            roof = max(flops / H100["fp8"], data / H100["hbm"])
            r = {"layer": name, "n": n_out, "k": k, "m": m, "kernel": kernel, "seconds": sec,
                 "quant_seconds": quant_s, "tflops": flops / sec / 1e12, "tb_per_s": data / sec / 1e12,
                 "of_fp8_peak": flops / sec / H100["fp8"], "of_hbm_peak": data / sec / H100["hbm"],
                 "roofline_seconds": roof, "of_roofline": roof / sec, **c}
            out_rows.append(r)
            print(f"  gemm {name:26s} m={m:6d} {kernel:18s} {1e6 * sec:8.1f} us  {r['tflops']:6.0f} TFLOP/s "
                  f"({100 * r['of_fp8_peak']:3.0f}%)  {r['tb_per_s']:.2f} TB/s ({100 * r['of_hbm_peak']:3.0f}%)  "
                  f"roofline {100 * r['of_roofline']:3.0f}%  quant {1e6 * quant_s:.1f} us  {c.get('sm_mhz')} MHz "
                  f"{c.get('watts')} W", flush=True)
        weights = None
        torch.cuda.empty_cache()
    return out_rows


# ---------------------------------------------------------------------------------------------------- attention
def bench_attention(clocks):
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
    fp8 = torch.float8_e4m3fn
    hq, hkv, d = 24, 4, 256
    out_rows = []
    configs = [(b, ql, ctx, page) for page in (832, 64) for b in (16, 32, 64) for ql in (1, 4)
               for ctx in (1024, 2048, 4096, 8192)]
    for b, ql, ctx, page in configs:
        pages_per_req = math.ceil((ctx + ql) / page)
        copies = max(1, math.ceil(3 * L2_BYTES / (b * ctx * hkv * 2 * d)))
        num_blocks = b * pages_per_req * copies
        # vLLM's layout: (blocks, kv heads, page, 2 * head_dim), K and V interleaved per token and head
        cache = torch.randn(num_blocks, hkv, page, 2 * d, device="cuda", dtype=torch.bfloat16).to(fp8)
        k_cache, v_cache = cache.transpose(1, 2).split(d, dim=-1)
        tables = [torch.arange(i * b * pages_per_req, (i + 1) * b * pages_per_req, device="cuda",
                               dtype=torch.int32).view(b, pages_per_req) for i in range(copies)]
        q = torch.randn(b * ql, hq, d, device="cuda", dtype=torch.bfloat16).to(fp8)
        out = torch.empty(b * ql, hq, d, device="cuda", dtype=torch.bfloat16)
        cu_q = torch.arange(0, b * ql + 1, ql, device="cuda", dtype=torch.int32)
        seqused = torch.full((b,), ctx + ql, device="cuda", dtype=torch.int32)
        one = torch.ones(b, hkv, device="cuda", dtype=torch.float32)
        for splits_name, splits in [("cap 32 (graphs)", 32), ("heuristic", 0)]:
            meta = None
            if splits:
                meta = get_scheduler_metadata(batch_size=b, max_seqlen_q=ql, max_seqlen_k=ctx + ql,
                                              num_heads_q=hq, num_heads_kv=hkv, headdim=d, cache_seqlens=seqused,
                                              qkv_dtype=fp8, cu_seqlens_q=cu_q, page_size=page, causal=True,
                                              num_splits=splits)
            it = {"i": 0}

            def fn():
                t = tables[it["i"] % copies]
                it["i"] += 1
                flash_attn_varlen_func(q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q, max_seqlen_q=ql,
                                       seqused_k=seqused, max_seqlen_k=ctx + ql, softmax_scale=d**-0.5,
                                       causal=True, block_table=t, scheduler_metadata=meta, fa_version=3,
                                       q_descale=one, k_descale=one, v_descale=one, num_splits=splits)
            try:
                sec, c = case(clocks, fn, inner=max(copies, 10))
            except Exception as e:
                print(f"  attention b={b} q={ql} ctx={ctx} page={page} {splits_name}: FAILED "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            data = b * (ctx + ql) * hkv * 2 * d + b * ql * hq * d * 3
            flops = 4 * b * ql * hq * d * (ctx + ql)
            roof = max(flops / H100["fp8"], data / H100["hbm"])
            r = {"requests": b, "query_tokens": ql, "context": ctx, "page": page, "splits": splits_name,
                 "seconds": sec, "tb_per_s": data / sec / 1e12, "of_hbm_peak": data / sec / H100["hbm"],
                 "roofline_seconds": roof, "of_roofline": roof / sec, **c}
            out_rows.append(r)
            print(f"  attention b={b:2d} q={ql} ctx={ctx:5d} page={page:3d} {splits_name:15s} {1e6 * sec:7.1f} us "
                  f"{r['tb_per_s']:.2f} TB/s ({100 * r['of_hbm_peak']:3.0f}% of HBM peak)", flush=True)
        cache = k_cache = v_cache = tables = None
        torch.cuda.empty_cache()
    return out_rows


def bench_attention_flashinfer(clocks):
    """FlashInfer's paged kernels over an FP8 KV cache in its own layout (HND, K and V apart), BF16 queries, at the
    shapes above where the cache is larger than L2 (so no copies are needed): the prefill wrapper (FA2 and FA3
    backends) for 1 or 4 query tokens, the decode wrapper on tensor cores for 1."""
    import flashinfer
    fp8 = torch.float8_e4m3fn
    hq, hkv, d = 24, 4, 256
    ws = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    out_rows = []
    configs = [(b, ql, ctx, page) for page in (16, 64) for b, ctx in ((16, 4096), (16, 8192), (64, 1024),
                                                                       (64, 2048), (64, 4096))
               for ql in (1, 4)]
    for b, ql, ctx, page in configs:
        pages = math.ceil((ctx + ql) / page)
        k = torch.randn(b * pages, hkv, page, d, device="cuda", dtype=torch.bfloat16).to(fp8)
        v = torch.randn(b * pages, hkv, page, d, device="cuda", dtype=torch.bfloat16).to(fp8)
        indptr = torch.arange(0, b * pages + 1, pages, device="cuda", dtype=torch.int32)
        indices = torch.arange(b * pages, device="cuda", dtype=torch.int32)
        last = torch.full((b,), ctx + ql - (pages - 1) * page, device="cuda", dtype=torch.int32)
        qo = torch.arange(0, b * ql + 1, ql, device="cuda", dtype=torch.int32)
        q16 = torch.randn(b * ql, hq, d, device="cuda", dtype=torch.bfloat16)
        q8 = q16.to(fp8)
        variants = []
        for backend in ("fa2", "fa3"):
            for q in (q16, q8):
                def make(backend=backend, q=q):
                    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=backend)
                    w.plan(qo, indptr, indices, last, hq, hkv, d, page, causal=True, q_data_type=q.dtype,
                           kv_data_type=fp8, o_data_type=torch.bfloat16)
                    return lambda: w.run(q, (k, v))
                variants.append((f"prefill {backend} q {'fp8' if q is q8 else 'bf16'}", make))
        q = q16
        if ql == 1:
            def make_decode():
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "HND", use_tensor_cores=True)
                w.plan(indptr, indices, last, hq, hkv, d, page, q_data_type=torch.bfloat16, kv_data_type=fp8)
                return lambda: w.run(q, (k, v))
            variants.append(("decode tensor cores", make_decode))
        for name, make in variants:
            try:
                sec, c = case(clocks, make(), inner=10)
            except Exception as e:
                msg = str(e).replace("\n", " | ")
                print(f"  flashinfer b={b} q={ql} ctx={ctx} page={page} {name}: FAILED {type(e).__name__}: "
                      f"{msg[:200]} ... {msg[-600:]}", flush=True)
                continue
            data = b * (ctx + ql) * hkv * 2 * d + b * ql * hq * d * 4
            r = {"kernel": f"flashinfer {name}", "requests": b, "query_tokens": ql, "context": ctx, "page": page,
                 "seconds": sec, "tb_per_s": data / sec / 1e12, "of_hbm_peak": data / sec / H100["hbm"], **c}
            out_rows.append(r)
            print(f"  flashinfer b={b:2d} q={ql} ctx={ctx:5d} page={page:3d} {name:20s} {1e6 * sec:7.1f} us "
                  f"{r['tb_per_s']:.2f} TB/s ({100 * r['of_hbm_peak']:3.0f}% of HBM peak)", flush=True)
        k = v = None
        torch.cuda.empty_cache()
    return out_rows


# ----------------------------------------------------------------------------------------------------- deltanet
def bench_deltanet(clocks):
    from vllm import _custom_ops as ops
    hk, hv, dk, dv, spec = 16, 48, 128, 128, 3
    out_rows = []
    for b in (1, 16, 32, 64, 96):
        t = b * (spec + 1)
        state_bytes = hv * dv * dk * 2
        copies = max(1, math.ceil(3 * L2_BYTES / (b * (spec + 1) * state_bytes)))
        slots = b * (spec + 1) * copies + 1
        state = torch.randn(slots, hv, dv, dk, device="cuda", dtype=torch.bfloat16) * 0.01
        mixed = torch.randn(t, 2 * hk * dk + hv * dv, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(t, hv, device="cuda", dtype=torch.bfloat16)
        bb = torch.randn(t, hv, device="cuda", dtype=torch.bfloat16)
        a_log = torch.randn(hv, device="cuda", dtype=torch.float32)
        dt_bias = torch.ones(hv, device="cuda", dtype=torch.float32)
        gate = torch.randn(t, hv, dv, device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(dv, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(gate)
        cu = torch.arange(0, t + 1, spec + 1, device="cuda", dtype=torch.int32)
        accepted = torch.full((b,), 2, device="cuda", dtype=torch.int32)
        idx = [1 + torch.arange(i * b * (spec + 1), (i + 1) * b * (spec + 1), device="cuda",
                                dtype=torch.int32).view(b, spec + 1) for i in range(copies)]
        it = {"i": 0}

        def fn():
            s = idx[it["i"] % copies]
            it["i"] += 1
            ops.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed, a=a, b=bb, A_log=a_log, dt_bias=dt_bias,
                                               state_indices=s, cu_seqlens=cu, num_accepted_tokens=accepted,
                                               state=state, output_gate=gate, norm_weight=norm_w, out=out)
        try:
            sec, c = case(clocks, fn, inner=max(copies, 10))
        except Exception as e:
            print(f"  deltanet b={b}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        # one state read and spec + 1 written per request, plus the step's inputs and outputs
        data = b * (1 + spec + 1) * state_bytes + t * (mixed.shape[1] + 2 * hv + 2 * hv * dv) * 2
        r = {"requests": b, "positions": spec + 1, "seconds": sec, "tb_per_s": data / sec / 1e12,
             "of_hbm_peak": data / sec / H100["hbm"], "roofline_seconds": data / H100["hbm"],
             "state_writes_share": b * (spec + 1) * state_bytes / data, **c}
        out_rows.append(r)
        print(f"  deltanet b={b:3d} x {spec + 1} positions {1e6 * sec:7.1f} us {r['tb_per_s']:.2f} TB/s "
              f"({100 * r['of_hbm_peak']:3.0f}% of HBM peak; state writes {100 * r['state_writes_share']:.0f}% of "
              f"bytes)", flush=True)
        state = None
        torch.cuda.empty_cache()
    return out_rows


# ----------------------------------------------------------------------------------------------------- sampling
def logits_of(kind: str, rows: int, vocab: int, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if kind == "gaussian":
        return torch.randn(rows, vocab, device=dev, generator=gen) * 2
    if kind == "peaked":   # the earlier test's rows: a Gaussian floor and 8 spikes
        x = torch.randn(rows, vocab, device=dev, generator=gen) * 2
        peak = torch.randint(0, vocab, (rows, 8), device=dev, generator=gen)
        x.scatter_add_(1, peak, torch.rand(rows, 8, device=dev, generator=gen) * 25)
        return x
    if kind == "zipf":     # a language model's shape: log-probabilities falling with log rank, shuffled per row
        rank = torch.arange(1, vocab + 1, device=dev, dtype=torch.float32)
        base = -1.1 * torch.log(rank) + 0.3 * torch.randn(rows, vocab, device=dev, generator=gen)
        perm = torch.argsort(torch.rand(rows, vocab, device=dev, generator=gen), dim=1)
        return torch.gather(base, 1, perm)
    if kind == "confident":   # most decoding steps: one token holds nearly all the mass, then a Zipf tail
        x = logits_of("zipf", rows, vocab, gen)
        top = x.argmax(1, keepdim=True)
        x.scatter_add_(1, top, torch.full((rows, 1), 12.0, device=dev))
        return x
    if kind.startswith("grammar"):   # the answer's JSON grammar: tokens it does not allow are set to -inf
        allowed = int(kind.split("_")[1])
        x = logits_of("zipf", rows, vocab, gen)
        keep = torch.rand(rows, vocab, device=dev, generator=gen).argsort(dim=1)[:, :allowed]
        masked = torch.full_like(x, float("-inf"))
        return masked.scatter_(1, keep, torch.gather(x, 1, keep))
    if kind == "mixed":   # a step with half the rows reasoning (no grammar) and half answering (20 tokens allowed)
        x = logits_of("zipf", rows, vocab, gen)
        x[rows // 2:] = logits_of("grammar_20", rows - rows // 2, vocab, gen)
        return x
    raise ValueError(kind)


def bench_sampling(clocks):
    import flashinfer.sampling as fis
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    from fastqwenjudge.kernels.topk_topp import topk_topp
    vocab = 248320
    gen = torch.Generator(device="cuda").manual_seed(0)
    out_rows = []
    for rows in (64, 128, 256):
        k = torch.full((rows,), 20, device="cuda", dtype=torch.int32)
        p = torch.full((rows,), 0.95, device="cuda", dtype=torch.float32)
        for kind in ("gaussian", "peaked", "zipf", "confident", "grammar_1", "grammar_20", "grammar_2000",
                     "grammar_124160", "mixed"):
            x = logits_of(kind, rows, vocab, gen)
            work = x.clone()
            fns = {"vllm triton top-k/top-p": lambda: apply_top_k_top_p(work.copy_(x), k, p),
                   "fastqwenjudge three-pass": lambda: topk_topp(work.copy_(x), k, p),
                   "flashinfer top-k mask": lambda: fis.top_k_mask_logits(x, 20),
                   "flashinfer top-k mask + top-p": lambda: fis.top_p_renorm_probs(
                       torch.softmax(fis.top_k_mask_logits(x, 20), dim=-1), 0.95),
                   "torch.topk(20)": lambda: torch.topk(x, 20, dim=-1),
                   "copy (floor: read + write)": lambda: work.copy_(x)}
            for name, fn in fns.items():
                try:
                    sec, c = case(clocks, fn, inner=10)
                except Exception as e:
                    print(f"  sampling {rows} {kind} {name}: FAILED {type(e).__name__}: {str(e)[:120]}",
                          flush=True)
                    continue
                r = {"rows": rows, "logits": kind, "kernel": name, "seconds": sec,
                     "read_floor_seconds": rows * vocab * 4 / H100["hbm"], **c}
                out_rows.append(r)
                print(f"  sampling rows={rows:3d} {kind:9s} {name:28s} {1e6 * sec:8.1f} us", flush=True)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="hbm,gemm,attention,attention_flashinfer,deltanet,sampling")
    args = ap.parse_args()
    torch.cuda.init()
    clocks = Clocks()
    props = torch.cuda.get_device_properties(0)
    res = {"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
           "peaks": H100, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    benches = {"hbm": bench_hbm, "gemm": bench_gemm, "attention": bench_attention,
               "attention_flashinfer": bench_attention_flashinfer, "deltanet": bench_deltanet,
               "sampling": bench_sampling}
    for name in args.only.split(","):
        print(f"== {name}", flush=True)
        t0 = time.time()
        try:
            res[name] = benches[name](clocks)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"   ({time.time() - t0:.0f} s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
