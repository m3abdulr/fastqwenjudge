"""vLLM plugin (FQJ_TOPK_TOPP=1): top-k / top-p masking in three Triton passes, whatever the logits look like.

vLLM 0.32's model runner V2 masks the logits for top-k and top-p with a Triton kernel (Qrita) that searches for each
row's thresholds with pivots chosen from the row's statistics. On logits the answer's JSON grammar has masked (every
token it does not allow set to -inf), the search takes many more rounds: 2.5 ms for 64 rows with 20 tokens allowed,
against 0.3 ms unmasked (experiments/qwen38/ceilings.py), and in the 27B quality judge's profile it is 2.2-3.0 ms a
step. Here the cost does not depend on the values:

1. each block of a row (4,096 logits) writes its maximum; the k-th largest block maximum is a lower bound on the
   row's k-th largest logit (k elements are at least that large);
2. each block lists its logits at or above that bound (on real logits a handful per row, most blocks none);
3. per row, the k largest candidates give the top-k threshold, and top-p within them (vLLM's order: top-k first,
   then top-p over the k kept, by probability) the final one;
4. one pass per row, in token order, sets every logit below the threshold to -inf, in place, and of the logits
   equal to it keeps as many as the cut allows, lowest token ids first (vLLM's rule for ties).

Three reads and one write of the logits.
Used only when every row's top_k is at most 32 (the quality modes use 20) or the row is greedy with top-p off (its
top-1 token always stays, so its argmax is unchanged); any other batch goes to vLLM's kernel. The check runs on the
host arrays vLLM already keeps, so it adds no GPU sync. Against vLLM's kernel on synthetic, grammar-masked and tied
logits (a GPU test, not in the repository): the same tokens kept, except where a row's cumulative probability lands
within rounding of top_p (one token in a few hundred rows) and where thousands of tokens tie exactly at the top-p cut
(vLLM keeps them all). In the 27B quality judge's server (profiles/27b.yaml turns it on): 0.38 ms a step against 2.7
ms, 7% more images per second in steady state, verdicts within rerun noise (docs/roofline.md). Not on for Flash-Next:
no gain beyond noise.
"""
from __future__ import annotations

import os

import torch

K_MAX = 32       # candidates per block, and the largest top_k this path takes
BLOCK = 4096
_CALLED: list = []


def register():
    if os.environ.get("FQJ_TOPK_TOPP") == "1":
        _register()


def _register():
    """Model runner V2 masks in two places: the sampler calls apply_top_k_top_p itself, and the sampling states'
    method calls it for the logits it processes; both get the top-k / top-p tensors from
    SamplingStates.get_top_k_top_p first. That method is wrapped to mark, from the host arrays, whether this batch
    can take the fast path; the function is replaced in both modules and reads the mark."""
    import numpy as np
    from vllm.v1.worker.gpu.sample import sampler, states

    if getattr(states.apply_top_k_top_p, "fqj", False):
        return
    original = states.apply_top_k_top_p
    get_top_k_top_p = states.SamplingStates.get_top_k_top_p

    def marked_get_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np):
        top_k, top_p = get_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np)
        if top_k is not None:
            k = self.top_k.np[idx_mapping_np]
            greedy = self.temperature.np[idx_mapping_np] == 0.0
            top_k.fqj_small = bool(np.all((k <= K_MAX) | (greedy & (self.top_p.np[idx_mapping_np] >= 1.0))))
        return top_k, top_p

    def apply_top_k_top_p(logits, k, p, *args, **kwargs):
        if (k is None or not getattr(k, "fqj_small", False) or args or kwargs or logits.dtype != torch.float32
                or logits.stride(-1) != 1):
            return original(logits, k, p, *args, **kwargs)
        if not _CALLED:
            _CALLED.append(True)
            print("fastqwenjudge: three-pass top-k / top-p masking in use", flush=True)
        return topk_topp(logits, k, p)

    apply_top_k_top_p.fqj = True
    states.SamplingStates.get_top_k_top_p = marked_get_top_k_top_p
    states.apply_top_k_top_p = apply_top_k_top_p
    sampler.apply_top_k_top_p = apply_top_k_top_p


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _block_max(x_ptr, bmax_ptr, vocab, row_stride, n_blocks, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * row_stride + offs, mask=offs < vocab, other=float("-inf"))
        tl.store(bmax_ptr + row * n_blocks + blk, tl.max(x, axis=0))

    @triton.jit
    def _row_bound(bmax_ptr, k_ptr, bound_ptr, n_blocks, K: tl.constexpr, NB: tl.constexpr):
        # the k-th largest block maximum: k elements of the row are at least this, so the row's k largest are too
        row = tl.program_id(0)
        offs = tl.arange(0, NB)
        b = tl.load(bmax_ptr + row * n_blocks + offs, mask=offs < n_blocks, other=float("-inf"))
        k = tl.minimum(tl.load(k_ptr + row), K)
        bound = float("inf")
        for i in tl.static_range(K):
            m, j = tl.max(b, axis=0, return_indices=True)
            bound = tl.where(i == k - 1, m, bound)
            b = tl.where(offs == j, float("-inf"), b)
        tl.store(bound_ptr + row, bound)

    @triton.jit
    def _block_candidates(x_ptr, bound_ptr, cand_ptr, vocab, row_stride, n_blocks, K: tl.constexpr,
                          BLOCK: tl.constexpr):
        # each block's values at or above the bound, at most K of them, largest first (most blocks have none)
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * row_stride + offs, mask=offs < vocab, other=float("-inf"))
        bound = tl.load(bound_ptr + row)
        x = tl.where((x >= bound) & (x > float("-inf")), x, float("-inf"))
        n = tl.minimum(tl.sum((x > float("-inf")).to(tl.int32), axis=0), K)
        out = cand_ptr + (row * n_blocks + blk) * K
        for i in range(n):
            m, j = tl.max(x, axis=0, return_indices=True)
            tl.store(out + i, m)
            x = tl.where(tl.arange(0, BLOCK) == j, float("-inf"), x)   # one occurrence at a time: ties counted

    @triton.jit
    def _row_threshold(cand_ptr, k_ptr, p_ptr, thr_ptr, ties_ptr, n_cand, HAS_P: tl.constexpr, K: tl.constexpr,
                       NC: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, NC)
        c = tl.load(cand_ptr + row * n_cand + offs, mask=offs < n_cand, other=float("-inf"))
        k = tl.minimum(tl.load(k_ptr + row), K)
        idx = tl.arange(0, K)
        vals = tl.full([K], float("-inf"), tl.float32)
        for i in tl.static_range(K):   # the K largest candidates, descending, with multiplicity
            m, j = tl.max(c, axis=0, return_indices=True)
            vals = tl.where(idx == i, m, vals)
            c = tl.where(offs == j, float("-inf"), c)
        keep = (idx < k) & (vals > float("-inf"))
        if HAS_P:
            p = tl.load(p_ptr + row)
            top = tl.max(vals, axis=0)
            e = tl.where(keep, tl.exp(vals - top), 0.0)
            probs = e / tl.sum(e, axis=0)
            above = tl.cumsum(probs, axis=0) - probs   # probability of the tokens ranked above this one
            keep = keep & ((above < p) | (idx == 0))
        n_keep = tl.sum(keep.to(tl.int32), axis=0)
        thr = tl.min(tl.where(keep, vals, float("inf")), axis=0)
        # how many tokens equal to the threshold stay: vLLM keeps exactly the count, lowest token ids first
        ties = n_keep - tl.sum((keep & (vals > thr)).to(tl.int32), axis=0)
        tl.store(thr_ptr + row, thr)
        tl.store(ties_ptr + row, ties)

    @triton.jit
    def _mask_row(x_ptr, thr_ptr, ties_ptr, vocab, row_stride, n_blocks, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        t = tl.load(thr_ptr + row)
        ties = tl.load(ties_ptr + row)
        seen = tl.zeros((), dtype=tl.int32)
        for blk in range(n_blocks):   # in token order, so ties keep the lowest ids
            offs = blk * BLOCK + tl.arange(0, BLOCK)
            ptr = x_ptr + row * row_stride + offs
            x = tl.load(ptr, mask=offs < vocab, other=float("-inf"))
            eq = (x == t) & (offs < vocab)
            rank = tl.cumsum(eq.to(tl.int32), axis=0) + seen
            keep = (x > t) | (eq & (rank <= ties))
            tl.store(ptr, tl.where(keep, x, float("-inf")), mask=offs < vocab)
            seen += tl.sum(eq.to(tl.int32), axis=0)
except ImportError:   # pragma: no cover - the plugin is only enabled inside vLLM, which has Triton
    triton = None


def topk_topp(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor:
    """vLLM's apply_top_k_top_p for top_k <= K_MAX: the logits masked with -inf, in place."""
    rows, vocab = logits.shape
    n_blocks = triton.cdiv(vocab, BLOCK)
    dev = logits.device
    k = k.to(torch.int32)
    bmax = torch.empty(rows, n_blocks, device=dev, dtype=torch.float32)
    bound = torch.empty(rows, device=dev, dtype=torch.float32)
    cand = torch.full((rows, n_blocks * K_MAX), float("-inf"), device=dev, dtype=torch.float32)
    thr = torch.empty(rows, device=dev, dtype=torch.float32)
    ties = torch.empty(rows, device=dev, dtype=torch.int32)
    stride = logits.stride(0)
    _block_max[(rows, n_blocks)](logits, bmax, vocab, stride, n_blocks, BLOCK=BLOCK, num_warps=8)
    _row_bound[(rows,)](bmax, k, bound, n_blocks, K=K_MAX, NB=triton.next_power_of_2(n_blocks), num_warps=2)
    _block_candidates[(rows, n_blocks)](logits, bound, cand, vocab, stride, n_blocks, K=K_MAX, BLOCK=BLOCK,
                                        num_warps=8)
    _row_threshold[(rows,)](cand, k, p if p is not None else thr, thr, ties, n_blocks * K_MAX, HAS_P=p is not None,
                            K=K_MAX, NC=triton.next_power_of_2(n_blocks * K_MAX), num_warps=8)
    _mask_row[(rows,)](logits, thr, ties, vocab, stride, n_blocks, BLOCK=BLOCK, num_warps=8)
    return logits
