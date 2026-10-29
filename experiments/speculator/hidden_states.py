"""Hidden states of the served model for every training conversation, for offline draft-model training.

    python experiments/speculator/hidden_states.py --profile configs/servers/tuned_fp8.yaml --set kv-cache-dtype=fp8 \
        --conversations experiments/speculator/data/conversations.jsonl --out /scratch/$USER/fastqwenjudge/eagle3

Starts the model with vLLM's hidden-state extraction: the outputs of layers 2, 32 and 61 of 64 (EAGLE-3's inputs)
and of the last layer (its training target), written to disk per request by vLLM's example hidden-states
connector. Two steps of the speculators library (--speculators, its CLI) then run against that server:

  prepare-data           renders each conversation with the server's chat template into token ids and a loss
                         mask that is 1 on the assistant's answer, and counts the answer tokens -> <out>/data
  generate-offline-data  one prefill per conversation, images included   -> <out>/data/hidden_states

The server runs with the weights and KV-cache precision the draft model will serve with, so it learns the
hidden states it will see. At about 40 KB per token (4 layers x 5,120 values in BF16), the judge's ~3,600
tokens per image come to ~150 MB per image: put --out on a scratch file system.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from fastqwenjudge.server import Server, load_profile

LAYERS = [2, 32, 61, 64]   # vLLM's EAGLE-3 defaults for 64 layers (2, n/2, n-3) and the final hidden state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--conversations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-length", type=int, help="tokens per conversation (default: the server's max-model-len)")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--speculators", default="speculators", help="path to the speculators CLI")
    args = p.parse_args()
    out = Path(args.out)
    data, spool = out / "data", out / "spool"
    spool.mkdir(parents=True, exist_ok=True)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    profile = load_profile(args.profile, overrides)
    profile["name"] = "extract_hidden_states"
    profile["args"] |= {
        "speculative-config": {"method": "extract_hidden_states", "num_speculative_tokens": 1,
                               "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": LAYERS}}},
        "kv-transfer-config": {"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer",
                               "kv_connector_extra_config": {"shared_storage_path": str(spool.resolve())}},
        "allowed-local-media-path": "/",   # conversations refer to images by file path
        "enable-scale-out": True,          # registers /render, which prepare-data tokenizes with
        "renderer-num-workers": 2,
    }
    model = args.model or profile["model"]
    # the render endpoint refuses to truncate beyond the server's context, and every conversation fits in it anyway
    seq_length = args.seq_length or int(profile["args"].get("max-model-len", 8192))
    with Server(profile, model=args.model, log_dir=out / "logs") as server:
        root = server.base_url.removesuffix("/v1")
        # token frequencies (<out>/data/token_freq.pt) pick the draft model's reduced vocabulary at training time
        subprocess.run([args.speculators, "prepare-data", "--model", model, "--data", args.conversations,
                        "--render-endpoint", root, "--output", str(data), "--seq-length", str(seq_length),
                        "--no-skip-token-freq"], check=True)
        subprocess.run([args.speculators, "generate-offline-data", "--endpoint", server.base_url,
                        "--preprocessed-data", str(data), "--output", str(data / "hidden_states"),
                        "--concurrency", str(args.concurrency), "--validate-outputs"], check=True)


if __name__ == "__main__":
    main()
