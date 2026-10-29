"""Block-FP8 linear layer for Qwen-style checkpoints (128x128 weight blocks, dynamic per-token activation
groups of 128), as one Triton kernel.

vLLM 0.11 runs such a layer on an H100 as three steps: pad the activation rows to a multiple of 4 (by 4 rows
when they already are), quantize the activations to FP8 in a separate kernel, then a CUTLASS block-scaled GEMM.
Here each program quantizes the activation tile it is about to multiply, inside the GEMM's K loop: one K step
is exactly one 128-wide quantization group, so the group's absmax, scale and FP8 cast stay in registers and the
quantized activations never go to memory. Quantizing again for every output tile costs ALU time that grows with
the number of tokens, so this pays off for decode-sized batches; large prefill steps keep vLLM's path.

Two layouts of the same product:
  rows   tokens on the MMA's M side: y[m, n] tiles of BLOCK_M x BLOCK_N
  swap   weights on the MMA's M side (y^T = W x^T), so a 64- or 128-row MMA tile is filled by output features
         even when only a few tokens are in the step
SPLIT_K > 1 splits the K loop across programs (more programs for the 5,120-wide projections, which give only
40 tiles of 128 for 132 SMs); partial sums go to an FP32 buffer and are added up by a second, small kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

GROUP = 128   # activation quantization group = weight block = one K step (hard-coded as 128 in the kernel)


@dataclass(frozen=True)
class Config:
    block_m: int        # tokens per tile
    block_n: int        # output features per tile
    split_k: int = 1
    swap: bool = False
    num_warps: int = 4
    num_stages: int = 4


@triton.jit
def _kernel(x_ptr, w_ptr, ws_ptr, out_ptr, M, N, K,
            stride_xm, stride_wn, stride_ws_n, stride_ws_k, stride_os, stride_om,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr, SWAP: tl.constexpr,
            NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_N)
    # consecutive programs share the same token tile and walk along N, so that tile stays in L2
    pid_m, pid_n = pid // tiles_n, pid % tiles_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    row_ok = offs_m < M

    steps = K // 128
    per_split = tl.cdiv(steps, SPLIT_K)
    first = split * per_split
    last = tl.minimum(first + per_split, steps)

    x_tile = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_tile = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    ws_col = ws_ptr + (offs_n // 128) * stride_ws_n
    if SWAP:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for step in tl.range(first, last, num_stages=NUM_STAGES):
        k0 = step * 128
        x = tl.load(x_tile + k0, mask=row_ok[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_tile + k0)
        w_scale = tl.load(ws_col + step * stride_ws_k)
        # dynamic per-token quantization of this 128-wide group, as vLLM's per_token_group_quant_fp8 does it
        # (eps 1e-10, 448 = largest finite float8_e4m3fn)
        x_scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10) / 448.0
        q = tl.clamp(x / x_scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
        if SWAP:
            acc += tl.dot(w, tl.trans(q)) * w_scale[:, None] * x_scale[None, :]
        else:
            acc += tl.dot(q, tl.trans(w)) * x_scale[:, None] * w_scale[None, :]

    out = out_ptr + split * stride_os
    if SWAP:
        tl.store(out + offs_m[None, :] * stride_om + offs_n[:, None], acc.to(out_ptr.dtype.element_ty),
                 mask=row_ok[None, :])
    else:
        tl.store(out + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask=row_ok[:, None])


def linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, config: Config,
           out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """x [M, K] (BF16) times weight [N, K] (float8_e4m3fn, 128x128 blocks with scales weight_scale [N/128, K/128],
    any strides, e.g. the transposed copy vLLM keeps for CUTLASS seen through .T)."""
    m, k = x.shape
    n = weight.shape[0]
    assert x.stride(1) == 1 and weight.stride(1) == 1, "K must be the contiguous dimension"
    assert k % GROUP == 0 and n % config.block_n == 0 and weight_scale.shape == (n // GROUP, k // GROUP)
    split = config.split_k
    out = torch.empty((split, m, n) if split > 1 else (m, n), device=x.device,
                      dtype=torch.float32 if split > 1 else out_dtype)
    grid = (triton.cdiv(m, config.block_m) * (n // config.block_n), split)
    _kernel[grid](x, weight, weight_scale, out, m, n, k,
                  x.stride(0), weight.stride(0), weight_scale.stride(0), weight_scale.stride(1),
                  out.stride(0) if split > 1 else 0, out.stride(-2),
                  BLOCK_M=config.block_m, BLOCK_N=config.block_n, SPLIT_K=split, SWAP=config.swap,
                  NUM_STAGES=config.num_stages, num_warps=config.num_warps)
    return out.sum(0).to(out_dtype) if split > 1 else out


def candidates(m: int, n: int) -> list[Config]:
    """Tile shapes and K splits worth timing for one shape, at 4 warps and 4 stages (tuned further afterwards)."""
    out = []
    largest_m = max(16, triton.next_power_of_2(m))
    for swap in (False, True):
        for block_m in (16, 32, 64, 128):
            if block_m > largest_m:
                continue
            for block_n in (64, 128) if swap else (64, 128, 256):
                tiles = triton.cdiv(m, block_m) * (n // block_n)
                for split in (1, 2, 4):
                    if split == 1 or tiles * split <= 4 * 132:   # splitting only while it adds waves of programs
                        out.append(Config(block_m, block_n, split, swap))
    return out


def refinements(best: Config) -> list[Config]:
    return [Config(best.block_m, best.block_n, best.split_k, best.swap, warps, stages)
            for warps in (4, 8) for stages in (3, 4, 5, 6)]
