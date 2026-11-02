"""vLLM plugin: the output layer in block FP8 (FQJ_FP8_LM_HEAD=1).

Registered as a vLLM general plugin (pyproject entry point), so vLLM imports it in every process; it changes
nothing unless the server's environment sets FQJ_FP8_LM_HEAD=1.

Qwen's FP8 checkpoints keep the output layer in BF16 and vLLM never quantizes it. At load it is converted to the
same 128x128-block FP8 as the decoder's linear layers and runs on vLLM's own two block-FP8 matmuls. On
Qwen3.8-27B (248,320 x 5,120, run once for the model and once per draft position) that is 4% off both judges'
batch time, with verdicts moving about as much as between two identical runs (docs/serving.md). Single GPU only: with
tensor parallelism the layer stays in BF16.
"""
from __future__ import annotations

import functools
import os

import torch

SMALL = 32   # rows below which vLLM's block-FP8 matmul takes BF16 input (FlashInfer) instead of FP8 (DeepGEMM)
BLOCK = (128, 128)


def register():
    if os.environ.get("FQJ_FP8_LM_HEAD") == "1":
        _register_fp8_lm_head()


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
        # the judge model's own output layer only: a separately loaded drafter's layer comes under another model tag
        if (isinstance(layer, ParallelLMHead) and backends.model_tag == "backbone"
                and layer.tp_size == 1
                and w.dtype == torch.bfloat16 and w.shape[0] % BLOCK[0] == 0 and w.shape[1] % BLOCK[1] == 0):
            shape = tuple(w.shape)   # the BF16 weight is freed below
            layer.quant_method = _Fp8LMHead(layer)
            print(f"fastqwenjudge: output layer {shape} in block FP8", flush=True)

    process_weights_after_loading.fp8_lm_head = True
    UnquantizedEmbeddingMethod.process_weights_after_loading = process_weights_after_loading

