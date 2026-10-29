"""Fused Triton kernels for the elementwise work in front of Qwen3-VL-32B's FP8 matmuls.

On vLLM 0.32 each of these is two kernels: torch.compile's residual add + RMSNorm (or SiLU x up), then vLLM's
per-128-group FP8 conversion for DeepGEMM. Here each is one pass, so the BF16 activation is never written out and
read back:

  add_rms_norm_fp8   x + residual -> new residual (BF16); RMSNorm(new residual) * weight -> FP8 and scales
  silu_mul_fp8       silu(gate) * up -> FP8 and scales

add_rms_norm and silu_mul are the same kernels writing BF16 instead, for batches under 32 rows, where vLLM's
FlashInfer matmul takes BF16 and converts it itself.

The arithmetic follows the two-kernel path as torch.compile generates it: BF16 roundings in the same places, one
scale per 128 values rounded up to a power of two (UE8M0, what DeepGEMM uses on H100), scales stored column-major as
DeepGEMM reads them. With power-of-two scales the FP8 value is the BF16 value times an exact power of two, and the
outputs are bit-identical to the server's (experiments/kernels/bench_fused.py checks this).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0   # largest e4m3 value
GROUP = 128


@triton.jit
def _to_fp8(y, q_ptr, s_ptr, offsets, mask, groups, group_mask, M, GROUP: tl.constexpr, FP8_MAX: tl.constexpr):
    """y is [groups, GROUP] FP32 values already rounded to BF16; one power-of-two scale per group."""
    scale = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10) * (1.0 / FP8_MAX))))
    q = tl.clamp(y / scale[:, None], -FP8_MAX, FP8_MAX).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(s_ptr + groups * M + tl.program_id(0).to(tl.int64), scale, mask=group_mask)


@triton.jit
def _add_rms_norm(x_ptr, res_ptr, w_ptr, res_out_ptr, y_ptr, s_ptr, M, eps,
                  N: tl.constexpr, BLOCK: tl.constexpr, GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
                  QUANT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    h = (tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
         + tl.load(res_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32))
    tl.store(res_out_ptr + row * N + cols, h.to(tl.bfloat16), mask=mask)
    # vLLM's norm casts to BF16 before the weight multiply, but torch.compile drops that cast (the product stays
    # FP32 until the end); keeping the cast changes ~2% of the FP8 values
    y = h * tl.rsqrt(tl.sum(h * h, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (y * w.to(tl.float32)).to(tl.bfloat16)
    if QUANT:
        gcols = tl.reshape(cols, (BLOCK // GROUP, GROUP))
        groups = tl.arange(0, BLOCK // GROUP)
        _to_fp8(tl.reshape(y.to(tl.float32), (BLOCK // GROUP, GROUP)), y_ptr, s_ptr, row * N + gcols, gcols < N,
                groups, groups < N // GROUP, M, GROUP, FP8_MAX)
    else:
        tl.store(y_ptr + row * N + cols, y, mask=mask)


@triton.jit
def _silu_mul(gu_ptr, y_ptr, s_ptr, M,
              D: tl.constexpr, BLOCK: tl.constexpr, GROUP: tl.constexpr, FP8_MAX: tl.constexpr, QUANT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(gu_ptr + row * 2 * D + cols).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * D + D + cols).to(tl.float32)
    y = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    if QUANT:
        gcols = tl.reshape(cols, (BLOCK // GROUP, GROUP))
        groups = tl.program_id(1) * (BLOCK // GROUP) + tl.arange(0, BLOCK // GROUP)
        _to_fp8(tl.reshape(y.to(tl.float32), (BLOCK // GROUP, GROUP)), y_ptr, s_ptr, row * D + gcols, gcols < D,
                groups, groups < D // GROUP, M, GROUP, FP8_MAX)
    else:
        tl.store(y_ptr + row * D + cols, y)


def column_major_scales(m: int, k: int, device) -> torch.Tensor:
    """[m, k / 128] float32 scales laid out column-major, as per_token_group_quant_fp8(column_major_scales=True)."""
    return torch.empty(k // GROUP, m, dtype=torch.float32, device=device).T


def _norm(x, residual, weight, eps, quant, num_warps):
    m, n = x.shape
    assert n % GROUP == 0 and x.is_contiguous() and residual.is_contiguous()
    y = torch.empty(m, n, dtype=torch.float8_e4m3fn if quant else x.dtype, device=x.device)
    s = column_major_scales(m, n, x.device) if quant else y.new_empty(0, dtype=torch.float32)
    res_out = torch.empty_like(residual)
    _add_rms_norm[(m,)](x, residual, weight, res_out, y, s, m, eps, N=n, BLOCK=triton.next_power_of_2(n),
                        GROUP=GROUP, FP8_MAX=FP8_MAX, QUANT=quant, num_warps=num_warps)
    return y, s, res_out


def add_rms_norm_fp8(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                     num_warps: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (FP8 activations, their scales, new residual)."""
    return _norm(x, residual, weight, eps, True, num_warps)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                 num_warps: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (BF16 normalized activations, new residual)."""
    y, _, res_out = _norm(x, residual, weight, eps, False, num_warps)
    return y, res_out


def _silu(gate_up, quant, block, num_warps):
    m, d = gate_up.shape[0], gate_up.shape[1] // 2
    assert d % block == 0 and block % GROUP == 0 and gate_up.is_contiguous()
    y = torch.empty(m, d, dtype=torch.float8_e4m3fn if quant else gate_up.dtype, device=gate_up.device)
    s = column_major_scales(m, d, gate_up.device) if quant else y.new_empty(0, dtype=torch.float32)
    _silu_mul[(m, d // block)](gate_up, y, s, m, D=d, BLOCK=block, GROUP=GROUP, FP8_MAX=FP8_MAX, QUANT=quant,
                               num_warps=num_warps)
    return y, s


def silu_mul_fp8(gate_up: torch.Tensor, block: int = 1024, num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_up is [m, 2d] with the gate half first, as vLLM's merged gate_up projection writes it."""
    return _silu(gate_up, True, block, num_warps)


def silu_mul(gate_up: torch.Tensor, block: int = 1024, num_warps: int = 4) -> torch.Tensor:
    return _silu(gate_up, False, block, num_warps)[0]
