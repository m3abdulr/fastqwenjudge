Images/s, MFU, MBU, power, J/image and $/1k images are at the steady-state rate with that many requests in flight, at $3 per H100-hour; "1,000 images s" is a 1,000-image batch from the throughput mix, timed to its last verdict. Agreement counts an image when the judge and the reference agree that every requirement holds, with the neutral judge as it was before round 2 and before the presence fix: against Sol on 2,106 GenEval and GenAI-Bench images with Sol labels (all but two in the final evaluation set), and against the 242 images of the first audit (150 drawn at random, 100 where two judges disagreed, 8 of them both), so the audit column compares steps rather than estimating accuracy. Five rows reuse another run's verdicts instead of judging again: vLLM defaults uses the tuned BF16 server's (same weights and precision), and four speculation rows use the same setup's verdicts without drafts or with another draft schedule or head, because drafted tokens are kept only when the model would have produced them. Every 1,000-image batch left 5 or 6 images without a verdict after three attempts: 4 whose plans kept failing the compile step's checks and 1 or 2 long checklists that ran past the neutral judge's output limit, then a fixed 2,048 tokens.

| step | images/s | in flight | x vLLM defaults | one image p50 s | 1,000 images s | MFU | MBU | W | J/image | $/1k images | vs Sol, full image | vs audit, full image | same verdict as BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM defaults | 0.57 | 32 | 1.0 | 35.1 | — | 9% | 70% | 568 | 1,000 | 1.47 | 83.0% | 84.7% | — |
| tuned server, BF16 | 0.83 | 32 | 1.5 | 35.4 | — | 15% | 68% | 597 | 721 | 1.01 | 83.0% | 84.7% | — |
| + FP8 weights | 2.02 | 128 | 3.6 | 21.6 | — | 19% | 49% | 692 | 343 | 0.41 | 83.5% | 85.1% | 92.8% |
| + FP8 KV cache | 2.74 | 256 | 4.8 | 17.6 | 466 | 25% | 29% | 690 | 252 | 0.30 | 82.8% | 82.6% | 91.5% |
| + FlashInfer attention | 3.12 | 256 | 5.5 | 18.6 | 424 | 28% | 33% | 692 | 222 | 0.27 | 83.2% | 84.7% | 92.3% |
| + n-gram speculation while <= 96 requests run | 3.11 | 256 | 5.5 | 12.6 | 383 | 29% | 32% | 691 | 222 | 0.27 | 83.2% | 84.7% | 92.3% |
| + fused norm/SiLU + FP8 kernels (Qwen3-VL plugin) | 3.14 | 256 | 5.5 | 11.6 | 377 | 29% | 33% | 692 | 220 | 0.27 | 83.5% | 84.7% | 91.8% |
| + FP8 output layer (plugin) | 3.17 | 256 | 5.6 | 12.3 | 375 | 29% | 34% | 692 | 218 | 0.26 | 83.6% | 82.2% | 91.7% |

Not adopted:

| step | images/s | in flight | x vLLM defaults | one image p50 s | 1,000 images s | MFU | MBU | W | J/image | $/1k images | vs Sol, full image | vs audit, full image | same verdict as BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n-gram speculation, always on | 2.36 | 128 | 4.2 | 9.3 | 454 | 22% | 21% | 689 | 292 | 0.35 | 83.7% | 84.3% | 92.1% |
| n-gram speculation while <= 96 requests run | 2.77 | 256 | 4.9 | 9.3 | 421 | 25% | 29% | 691 | 249 | 0.30 | 83.7% | 84.3% | 92.1% |
| EAGLE-3 head trained on judge traces, always on | 2.62 | 128 | 4.6 | 6.4 | — | 26% | 16% | 671 | 256 | 0.32 | 83.5% | 84.7% | 92.6% |
| FlashInfer + EAGLE-3 head while <= 128 requests run | 2.17 | 128 | 3.8 | 7.7 | 608 | 21% | 14% | 601 | 277 | 0.38 | 83.3% | 87.2% | 92.5% |
| FlashInfer + EAGLE-3, full-attention head | 2.70 | 256 | 4.8 | 7.4 | 415 | 25% | 30% | 691 | 256 | 0.31 | 83.3% | 87.2% | 92.5% |
| EAGLE-3, full-attention head (FlashAttention) | 2.84 | 128 | 5.0 | 6.6 | — | 27% | 17% | 692 | 244 | 0.29 | 83.5% | 84.7% | 92.6% |
