"""vLLM plugin: Qwen3-VL-specific kernels, each behind its own environment variable.

Registered as a vLLM general plugin (pyproject entry point), so vLLM imports it in every process; it changes
nothing unless the server's environment sets one of these:

FQJ_FUSED_KERNELS=1  In every decoder layer after the first, three pairs of kernels become one each: the residual
    add + RMSNorm before the QKV projection, the same before gate_up, and SiLU x up before down, each fused with the
    FP8 conversion its matmul needs. vLLM's FP8 matmul switches at 32 rows, and so do these: from 32 rows the fused
    kernel hands DeepGEMM FP8 activations and scales; below 32, FlashInfer takes BF16 and converts it inside its own
    kernel, so the norm or SiLU runs alone, still as one kernel. The matmuls are vLLM's own calls on vLLM's
    processed weights. The activations match vLLM's up to summation order (identical on random inputs up to 2,048
    rows, a handful of values differ beyond), so greedy outputs can differ the way they do between attention
    kernels.

FQJ_FP8_LM_HEAD=1  The output layer (5120 -> 151,936), which the FP8 checkpoint keeps in BF16 and vLLM never
    quantizes, is converted at load to the same 128x128-block FP8 as the decoder's linear layers and runs on the same
    two matmuls. Halves its weight read and frees ~0.8 GB for the KV cache. A speculative drafter's output layer is
    left in BF16.

Single GPU only (tensor parallel falls back to vLLM's path).
"""
from __future__ import annotations

import functools
import os

import torch

SMALL = 32   # rows below which vLLM uses FlashInfer's BF16-input matmul (dynamic_flashinfer_deepgemm_blockscale_gemm)
BLOCK = (128, 128)
# layers whose input norm stays vLLM's (comma-separated indices)
KEEP_INPUT_NORM = {int(i) for i in os.environ.get("FQJ_FUSED_KEEP_INPUT_NORM", "").split(",") if i}


def register():
    if os.environ.get("FQJ_FUSED_KERNELS") == "1":
        _register_fused_layers()
    if os.environ.get("FQJ_FP8_LM_HEAD") == "1":
        _register_fp8_lm_head()


def _register_fused_layers():
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.models import qwen3
    from vllm.model_executor.models.utils import extract_layer_index
    if getattr(qwen3.Qwen3DecoderLayer.forward, "fused", False):
        return
    _define_ops()
    original = qwen3.Qwen3DecoderLayer.forward

    def forward(self, positions, hidden_states, residual):
        if (residual is None or get_tensor_model_parallel_world_size() > 1
                or self.self_attn.qkv_proj.weight.dtype != torch.float8_e4m3fn):
            return original(self, positions, hidden_states, residual)
        keep_input_norm = extract_layer_index(self.self_attn.attn.layer_name) in KEEP_INPUT_NORM
        return _fused_forward(self, positions, hidden_states, residual, keep_input_norm)

    forward.fused = True
    qwen3.Qwen3DecoderLayer.forward = forward
    print("fastqwenjudge: fused norm/SiLU + FP8 kernels on for Qwen3 decoder layers", flush=True)


def _weights(linear) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP8 weight and block scales vLLM's own matmul uses (already re-quantized for DeepGEMM at load)."""
    scale = linear.weight_scale_inv if hasattr(linear, "weight_scale_inv") else linear.weight_scale
    return linear.weight, scale


def _fused_forward(layer, positions, hidden_states, residual, keep_input_norm: bool):
    attn, mlp = layer.self_attn, layer.mlp
    if keep_input_norm:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        qkv, _ = attn.qkv_proj(hidden_states)
    else:
        qkv, residual = torch.ops.fastqwenjudge.norm_linear(
            hidden_states, residual, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon,
            *_weights(attn.qkv_proj))
    # the rest of Qwen3Attention.forward, unchanged
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.view(*q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim)).view(q.shape)
    k = attn.k_norm(k.view(*k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim)).view(k.shape)
    q, k = attn.rotary_emb(positions, q, k)
    hidden_states, _ = attn.o_proj(attn.attn(q, k, v))
    gate_up, residual = torch.ops.fastqwenjudge.norm_linear(
        hidden_states, residual, layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon, *_weights(mlp.gate_up_proj))
    return torch.ops.fastqwenjudge.silu_linear(gate_up, *_weights(mlp.down_proj)), residual


@functools.cache
def _matmuls():
    """vLLM 0.32's two block-FP8 matmuls: DeepGEMM on FP8 activations (32+ rows), FlashInfer on BF16 (fewer)."""
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used
    from vllm.utils.flashinfer import flashinfer_fp8_blockscale_gemm
    e8m0 = is_deep_gemm_e8m0_used()

    def matmul_fp8(q, s, weight, scale):
        out = torch.empty(q.shape[0], weight.shape[0], dtype=torch.bfloat16, device=q.device)
        fp8_gemm_nt((q, s), (weight, scale), out, is_deep_gemm_e8m0_used=e8m0)
        return out

    def matmul_bf16(y, weight, scale):
        return flashinfer_fp8_blockscale_gemm(input=y, weight=weight, weight_scale=scale, out_dtype=torch.bfloat16)
    return matmul_fp8, matmul_bf16, e8m0


def _define_ops():
    from fastqwenjudge.kernels import fused_quant
    matmul_fp8, matmul_bf16, _ = _matmuls()

    # opaque to torch.compile (it sees one node per op), so the row-count branch below runs at execution time
    @torch.library.custom_op("fastqwenjudge::norm_linear", mutates_args=())
    def norm_linear(x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                    weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] < SMALL:
            y, new_residual = fused_quant.add_rms_norm(x, residual, norm_weight, eps)
            return matmul_bf16(y, weight, scale), new_residual
        q, s, new_residual = fused_quant.add_rms_norm_fp8(x, residual, norm_weight, eps)
        return matmul_fp8(q, s, weight, scale), new_residual

    @norm_linear.register_fake
    def _(x, residual, norm_weight, eps, weight, scale):
        return x.new_empty(x.shape[0], weight.shape[0]), torch.empty_like(residual)

    @torch.library.custom_op("fastqwenjudge::silu_linear", mutates_args=())
    def silu_linear(gate_up: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if gate_up.shape[0] < SMALL:
            return matmul_bf16(fused_quant.silu_mul(gate_up), weight, scale)
        return matmul_fp8(*fused_quant.silu_mul_fp8(gate_up), weight, scale)

    @silu_linear.register_fake
    def _(gate_up, weight, scale):
        return gate_up.new_empty(gate_up.shape[0], weight.shape[0])


class _Fp8LMHead:
    """Replaces the output layer's quant method: its weight in 128x128-block FP8, its matmul vLLM's FP8 pair."""

    def __init__(self, layer):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
        from vllm.utils.deep_gemm import per_block_cast_to_fp8
        _, _, e8m0 = _matmuls()
        # the checkpoint's FP8 layers went through the same two steps: FP8 blocks with float scales, then DeepGEMM's
        # power-of-two scales and layout
        w8, scale = per_block_cast_to_fp8(layer.weight.data, list(BLOCK), use_ue8m0=False)
        self.weight, self.scale = deepgemm_post_process_fp8_weight_block(w8, scale, BLOCK, use_e8m0=e8m0)
        layer.weight.data = torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device)   # free BF16 copy
        torch.cuda.empty_cache()

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
        matmul_fp8, matmul_bf16, e8m0 = _matmuls()
        x = x.contiguous()
        if x.shape[0] < SMALL:
            out = matmul_bf16(x, self.weight, self.scale)
        else:
            q, s = per_token_group_quant_fp8(x, BLOCK[1], column_major_scales=True, use_ue8m0=e8m0)
            out = matmul_fp8(q, s, self.weight, self.scale)
        return out if bias is None else out + bias


def _register_fp8_lm_head():
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
    original = UnquantizedEmbeddingMethod.process_weights_after_loading
    if getattr(original, "fp8_lm_head", False):
        return

    def process_weights_after_loading(self, layer):
        from vllm.compilation import backends
        original(self, layer)
        w = layer.weight
        # the judge model's output layer only: vLLM loads a speculative drafter (EAGLE-3's own 32,000-token head)
        # under another model tag, and quantizing that head costs draft acceptance
        if (isinstance(layer, ParallelLMHead) and backends.model_tag == "backbone" and layer.tp_size == 1
                and w.dtype == torch.bfloat16 and w.shape[0] % BLOCK[0] == 0 and w.shape[1] % BLOCK[1] == 0):
            shape = tuple(w.shape)   # the BF16 weight is freed below
            layer.quant_method = _Fp8LMHead(layer)
            print(f"fastqwenjudge: output layer {shape} in block FP8", flush=True)

    process_weights_after_loading.fp8_lm_head = True
    UnquantizedEmbeddingMethod.process_weights_after_loading = process_weights_after_loading
