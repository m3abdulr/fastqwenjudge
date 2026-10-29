"""The optimization ladder in one table: each serving step's speed, hardware efficiency, cost and accuracy.

    python experiments/ladder.py --throughput experiments/throughput/results/vllm-0.32.0 \
        --quality experiments/quality/results/vllm-0.32.0 --sol data/references/sol.jsonl \
        --audit experiments/audit/results/claude.jsonl --out docs/ladder

Every step adds one change to the one before, on one H100. Speed and efficiency come from the steady-state
sweep (throughput/sweep.py, highest-throughput point and one image at a time) and the 1,000-image batch
(throughput/makespan.py); accuracy is the neutral judge's full-success agreement with the GPT-6 Sol labels
(sol_subset) and with the blind audit (its images in sol_subset and quality_ood), plus how often the step's
verdicts match the BF16 step's. vLLM's defaults and the tuned server run the same BF16 weights, so they share
the BF16 accuracy run. Missing runs show as a dash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fastqwenjudge import metrics

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "throughput"))
from efficiency import efficiency, load_model, load_sweeps, table  # noqa: E402

# (label, sweep and makespan name, quality run)
RUNGS = [("vLLM defaults", "stock", "bf16"),
         ("tuned server, BF16", "tuned", "bf16"),
         ("+ FP8 weights", "tuned_fp8", "fp8"),
         ("+ FP8 KV cache", "tuned_fp8_fp8kv", "fp8_fp8kv"),
         ("+ FlashInfer attention", "tuned_fp8_fp8kv_flashinfer", "fp8_fp8kv_flashinfer"),
         # speculation only proposes tokens the model then verifies, so the step above's verdicts stand for it
         ("+ n-gram speculation while <= 96 requests run", "tuned_fp8_fp8kv_flashinfer_ngram_dynamic",
          "fp8_fp8kv_flashinfer"),
         ("+ fused norm/SiLU + FP8 kernels (Qwen3-VL plugin)", "tuned_fp8_fp8kv_flashinfer_ngram_dynamic_fused",
          "fp8_fp8kv_flashinfer_ngram_dynamic_fused"),
         ("+ FP8 output layer (plugin)", "tuned_fp8_fp8kv_flashinfer_ngram_dynamic_fused_fp8head",
          "fp8_fp8kv_flashinfer_ngram_dynamic_fused_fp8head")]
# speculation variants measured but not adopted (FA3 attention unless named)
ALTERNATIVES = [("n-gram speculation, always on", "tuned_fp8_fp8kv_ngram", "fp8_fp8kv_ngram"),
                ("n-gram speculation while <= 96 requests run", "tuned_fp8_fp8kv_ngram_dynamic", "fp8_fp8kv_ngram"),
                ("EAGLE-3 head trained on judge traces, always on", "tuned_fp8_fp8kv_eagle3_k3",
                 "fp8_fp8kv_eagle3_k3"),
                ("FlashInfer + EAGLE-3 head while <= 128 requests run", "tuned_fp8_fp8kv_flashinfer_eagle3_dynamic",
                 "fp8_fp8kv_flashinfer_eagle3_dynamic"),
                # the same head with full attention instead of its 2,048-token window (one KV cache group, not 65);
                # speculation does not change outputs, so the windowed head's verdicts stand for it
                ("FlashInfer + EAGLE-3, full-attention head", "tuned_fp8_fp8kv_flashinfer_eagle3full_dynamic",
                 "fp8_fp8kv_flashinfer_eagle3_dynamic"),
                ("EAGLE-3, full-attention head (FlashAttention)", "tuned_fp8_fp8kv_eagle3full_dynamic",
                 "fp8_fp8kv_eagle3_k3")]
COLUMNS = [("step", "step", ""), ("images/s", "images_per_s", ".2f"), ("in flight", "concurrency", "d"),
           ("x vLLM defaults", "speedup", ".1f"), ("one image p50 s", "single_p50", ".1f"),
           ("1,000 images s", "makespan_s", ".0f"), ("MFU", "mfu", ".0%"), ("MBU", "mbu", ".0%"),
           ("W", "power_w_per_gpu", ".0f"), ("J/image", "joules_per_image", ",.0f"),
           ("$/1k images", "usd_per_1000_images", ".2f"), ("vs Sol, full image", "vs_sol", ".1%"),
           ("vs audit, full image", "vs_audit", ".1%"), ("same verdict as BF16", "vs_bf16", ".1%")]
NOTE = ("Images/s, MFU, MBU, power, J/image and $/1k images are at the steady-state rate with that many requests in "
        "flight, at $3 per H100-hour; \"1,000 images s\" is a 1,000-image batch from the throughput mix, timed to its "
        "last verdict. Agreement counts an image when the judge and the reference agree that every requirement holds, "
        "with the neutral judge as it was before round 2 and before the presence fix: against Sol on 2,106 GenEval and "
        "GenAI-Bench images with Sol labels (all but two in the final evaluation set), and against the 242 images of "
        "the first audit (150 drawn at random, 100 where two judges disagreed, 8 of them both), so the audit column "
        "compares steps rather than estimating accuracy. Five rows reuse another run's verdicts instead of judging "
        "again: vLLM defaults uses the tuned BF16 server's (same weights and precision), and four speculation rows "
        "use the same setup's verdicts without drafts or with another draft schedule or head, because drafted tokens "
        "are kept only when the model would have produced them. Every 1,000-image batch left 5 or 6 images without a "
        "verdict after three attempts: 4 whose plans kept failing the compile step's checks and 1 or 2 long "
        "checklists that ran past the neutral judge's output limit, then a fixed 2,048 tokens.")


def full_success_agreement(judge: dict, reference: dict) -> float | None:
    ids = [i for i in judge if i in reference]
    if not ids:
        return None
    return float(np.mean([judge[i]["all_satisfied"] == reference[i]["all_satisfied"] for i in ids]))


def neutral(quality: Path, run: str) -> dict:
    """The neutral judge's verdicts of one quality run, both datasets."""
    out = {}
    for ds in ("sol_subset", "ood"):
        path = quality / f"{ds}_{run}" / "neutral.jsonl"
        if path.exists():
            out |= metrics.load(path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--throughput", required=True)
    p.add_argument("--quality", required=True)
    p.add_argument("--sol", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--checkpoint", nargs="*", default=[],
                   help="model=path for checkpoints that are not in the Hugging Face cache under their name")
    p.add_argument("--out", help="write <out>.md and <out>.json")
    args = p.parse_args()
    throughput, quality = Path(args.throughput), Path(args.quality)
    sweeps = load_sweeps(throughput)
    sol, audit = metrics.load(args.sol), metrics.load(args.audit)
    bf16 = neutral(quality, "bf16")
    paths = dict(kv.split("=", 1) for kv in args.checkpoint)
    models = {}

    def step(label: str, name: str, run: str) -> dict:
        row = {"step": label}
        if name in sweeps:
            s = sweeps[name]
            model_name = s["profile"]["model"]
            if model_name not in models:
                models[model_name] = load_model(paths.get(model_name, model_name))
            model = models[model_name]
            points = [efficiency(s, q, model) for q in s["points"] if q["images"]]
            best = max(points, key=lambda r: r["images_per_s"])
            row |= {k: best.get(k) for k in ("images_per_s", "concurrency", "mfu", "mbu", "power_w_per_gpu",
                                             "joules_per_image", "usd_per_1000_images")}
            row["single_p50"] = next((r["latency_p50"] for r in points if r["concurrency"] == 1), None)
        makespan = throughput / f"makespan_{name}.json"
        if makespan.exists():   # the warm runs in input order
            runs = [r for r in json.loads(makespan.read_text())["runs"] if r["policy"] == "fifo" and r["run"] != "cold"]
            row["makespan_s"] = float(np.mean([r["makespan_s"] for r in runs])) if runs else None
        verdicts = neutral(quality, run)
        if verdicts:
            row |= {"vs_sol": full_success_agreement(verdicts, sol),
                    "vs_audit": full_success_agreement(verdicts, audit),
                    "vs_bf16": full_success_agreement(verdicts, bf16) if run != "bf16" else None}
        return row

    ladder = [step(*r) for r in RUNGS]
    alternatives = [step(*r) for r in ALTERNATIVES]
    base = ladder[0].get("images_per_s")
    for row in ladder + alternatives:
        row["speedup"] = row["images_per_s"] / base if base and row.get("images_per_s") else None
    text = "\n".join([NOTE, ""] + table(ladder, COLUMNS) + ["", "Not adopted:", ""] + table(alternatives, COLUMNS))
    print(text)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps({"ladder": ladder, "alternatives": alternatives},
                                                                  indent=1))


if __name__ == "__main__":
    main()
