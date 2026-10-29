"""Start the server once per configuration and record whether it comes up and how much KV cache it gets.

    python experiments/throughput/startup.py --configs experiments/throughput/startup_configs.yaml

No requests are sent. With ~62.6 GiB of weights on an 80 GB card, the KV cache left over after start-up decides
how many requests can be in flight, so this is the first thing to look at when tuning a profile.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

from fastqwenjudge.server import Server, load_profile

HERE = Path(__file__).parent


def failure_reason(log: str) -> str:
    """The root cause, not the "engine core initialization failed" wrapper raised after it."""
    for kind in ("ValueError", "torch.OutOfMemoryError", "AssertionError", "RuntimeError"):
        errors = re.findall(re.escape(kind) + r": ([^\n]+)", log)
        if errors:
            return errors[-1].strip()
    return log.strip().splitlines()[-1][:300]


def wait_for_free_gpu(timeout: float = 120):
    """The engine process can outlive the API server by a few seconds; the next start-up must see an empty GPU."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if all(int(u) < 1024 for u in used):
            return
        time.sleep(2)
    raise RuntimeError("GPU memory was not released")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=str(HERE / "startup_configs.yaml"))
    p.add_argument("--model")
    p.add_argument("--only", nargs="*", help="run only these config names")
    p.add_argument("--out", default=str(HERE / "results" / "startup.json"))
    args = p.parse_args()
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else {}
    for config in yaml.safe_load(Path(args.configs).read_text()):
        if args.only and config["name"] not in args.only:
            continue
        profile = load_profile(config["profile"], config.get("set"))
        profile["name"] = re.sub(r"\W+", "_", config["name"]).strip("_")
        server = Server(profile, model=args.model, log_dir=out.parent / "logs")
        row = {"profile": config["profile"], "set": config.get("set", {}),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            with server:
                row.update(started=True, **server.memory_report())
        except (RuntimeError, TimeoutError):
            log = re.sub(r"\x1b\[[0-9;]*m", "", server.log_path.read_text(errors="replace"))
            row.update(started=False, reason=failure_reason(log), **server.memory_report())
        wait_for_free_gpu()
        rows[config["name"]] = row
        out.write_text(json.dumps(rows, indent=1))
        kv = f"{row['kv_cache_tokens']:,.0f} KV tokens" if row.get("kv_cache_tokens") else row.get("reason", "")
        print(f"{config['name']}: {'up' if row['started'] else 'FAILED'}, {kv}", flush=True)


if __name__ == "__main__":
    main()
