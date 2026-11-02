"""Serving ladder: several server settings ("rungs") started one after another in one job, each judging the same
images with the same judges, so node, data and client stay fixed (docs/serving.md).

    python experiments/qwen38/ladder.py --ladder experiments/qwen38/ladders/27b_c.yaml --model <checkpoint> \
        --items <items folder> --out <results>/27b_c

Ladder file: {profile, judges, data, rungs: [{name, set?, env?, profile?, model?, data?, judges?, max_concurrency?,
seed?, objective?, chunks?}]}. `profile` is a YAML path or a bundled name; `set` overrides its flags (null drops
one), `env` adds environment variables, and a rung can replace the profile, the checkpoint, the item file (a name in
--items or a path) and the judges; `seed: null` sends requests without a per-request seed; `steady: {concurrency,
warmup, seconds}` (ladder or rung) measures steady-state throughput (run.py's run_steady) instead of one batch.
`objective: makespan` starts the longest checklists first; `chunks: n` judges the items as n consecutive batches.
Each rung writes <out>/<rung>/ (verdicts, timings, server log). A rung whose timings already exist is skipped; a
rung that fails is recorded in error.txt and the next one starts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import JUDGES, run_all, run_steady  # noqa: E402

from fastqwenjudge.data import load_items  # noqa: E402
from fastqwenjudge.server import Server, load_profile  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", required=True)
    p.add_argument("--model", help="the checkpoint for rungs that do not name their own")
    p.add_argument("--items", default=".", help="folder the ladder's item file names are looked up in")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", help="run only these rungs")
    args = p.parse_args()
    spec = yaml.safe_load(Path(args.ladder).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.yaml").write_text(Path(args.ladder).read_text())
    for rung in spec["rungs"]:
        if args.only and rung["name"] not in args.only:
            continue
        folder = out / rung["name"]
        if list(folder.glob("timings_*.json")):
            print(f"{rung['name']}: done already", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        judges = rung.get("judges", spec["judges"])
        assert all(j in JUDGES for j in judges), judges
        data = Path(rung.get("data", spec.get("data", "")))
        items = load_items(data if data.is_absolute() or data.exists() else Path(args.items) / data)
        profile = load_profile(rung.get("profile", spec["profile"]), rung.get("set", {}))
        profile["args"] = {k: v for k, v in profile["args"].items() if v is not None}
        profile.setdefault("env", {}).update(rung.get("env", {}))
        profile["name"] = rung["name"]
        print(f"=== {rung['name']}: {json.dumps(profile['args'])} env {json.dumps(profile['env'])}", flush=True)
        try:
            with Server(profile, model=rung.get("model", args.model), log_dir=folder / "logs",
                        startup_timeout=3600) as server:
                timings = {"rung": rung, "args": profile["args"], "startup_seconds": server.startup_seconds}
                steady = rung.get("steady", spec.get("steady"))
                if steady:   # {concurrency, warmup, seconds}: steady-state throughput instead of one batch
                    timings.update(asyncio.run(run_steady(server, items, judges, seed=rung.get("seed", 0),
                                                          **steady)))
                else:
                    timings.update(asyncio.run(run_all(server, items, judges, folder, rung.get("max_concurrency"),
                                                       rung.get("seed", 0), rung.get("objective", "latency"),
                                                       rung.get("chunks", 1))))
                timings["memory"] = server.memory_report()
            (folder / f"timings_{int(time.time())}.json").write_text(json.dumps(timings, indent=1))
            print(f"{rung['name']}: " + ", ".join(
                f"{j} {timings[j]['images_per_s']:.2f} images/s" if timings[j].get("steady") else
                f"{j} {timings[j]['wall_seconds']:.0f} s" for j in judges), flush=True)
        except Exception:
            (folder / "error.txt").write_text(traceback.format_exc())
            print(f"{rung['name']}: FAILED\n{traceback.format_exc()[-2000:]}", flush=True)


if __name__ == "__main__":
    main()
