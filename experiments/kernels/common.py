"""Shapes, hardware peaks and timing shared by the kernel benchmarks."""
from __future__ import annotations

import torch
from triton.testing import do_bench_cudagraph

H100 = {"bf16": 989.4e12, "fp8": 1978.9e12, "bytes_per_s": 3.35e12}   # SXM, dense
LAYERS = 64
# (name, N, K): output and input features of each projection in Qwen3-VL-32B's language model
SHAPES = [("qkv", 10240, 5120), ("o_proj", 5120, 8192), ("gate_up", 51200, 5120), ("down", 5120, 25600)]
HEAD = ("lm_head", 151936, 5120)
BLOCK = [128, 128]


def timed(fn, copies: int, rep: int = 200) -> float:
    """Microseconds per call, replaying a CUDA graph of `copies` calls (one per weight copy, so that nothing stays
    in the 50 MB L2 cache between calls, like the 64 different layers of a real step)."""
    fn()   # compile and allocate outside the graph
    torch.cuda.synchronize()
    return 1000 * do_bench_cudagraph(fn, rep=rep) / copies
