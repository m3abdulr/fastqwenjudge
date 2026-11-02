"""fastqwenjudge judge | serve | checklist | compare

    fastqwenjudge judge --data items.jsonl --out results.jsonl                  # fast mode, starts vLLM
    fastqwenjudge judge --mode quality --data items.jsonl --out results.jsonl
    fastqwenjudge judge --mode flash-quality --data items.jsonl --out results.jsonl   # two GPUs
    fastqwenjudge judge --gpus 0,1,2,3 --data items.jsonl --out results.jsonl  # one server per GPU (per pair: flash)
    fastqwenjudge serve --mode fast --port 8000                                 # or keep a server running ...
    fastqwenjudge judge --base-url http://127.0.0.1:8000/v1 --data items.jsonl --out results.jsonl
    fastqwenjudge judge --mode quality --base-url http://127.0.0.1:8000/v1 ... # ... the same server: both modes
    fastqwenjudge checklist --data prompts.jsonl --out items.jsonl              # prompts without a checklist
    fastqwenjudge compare --reference sol.jsonl results.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from . import judges, metrics
from .checklists import write_checklists
from .client import Client
from .data import load_items
from .modes import MODES, make_judge, mode_profile
from .runner import OBJECTIVES, run
from .server import Replicas, Server, load_profile
from .throughput import Monitor, request_summary


def api_key(args) -> str:
    """OPENAI_API_KEY for an endpoint given with --base-url that needs one; a server this command starts needs none."""
    return os.environ.get("OPENAI_API_KEY", "EMPTY") if args.base_url else "EMPTY"


async def judge_with(args, base_url: str | list[str], model: str, server: Server | Replicas | None = None) -> dict:
    items = load_items(args.data, groups=args.groups, limit=args.limit)
    judge = judges.make(args.judge) if args.judge else make_judge(args.mode)
    started = time.monotonic()
    schedule = OBJECTIVES[args.objective]
    async with Client(base_url, model, concurrency=args.concurrency, admission=schedule["admission"],
                      timeout=args.timeout, api_key=api_key(args)) as client, Monitor(base_url) as monitor:
        results = await run(judge, items, client, args.out, order=schedule["order"])
    stats = {"judge": judge.name, "mode": args.mode, "objective": args.objective, "concurrency": args.concurrency,
             "images": len(items), "judged": len(results), "wall_seconds": time.monotonic() - started,
             "server": monitor.summary(), "client": request_summary(client.log)}
    stats["images_per_s"] = len(items) / stats["wall_seconds"]
    if server is not None:
        stats["memory"] = server.memory_report()
        stats["profile"] = server.profile
    Path(str(args.out) + ".stats.json").write_text(json.dumps(stats, indent=1))
    s = stats["server"]
    print(f"{len(results)}/{len(items)} judged in {stats['wall_seconds']:.0f} s ({stats['images_per_s']:.2f} "
          f"images/s); {s['prompt_tokens_per_s']:.0f} prompt + {s['generation_tokens_per_s']:.0f} generated tokens/s",
          flush=True)
    return stats


def servers(profile: dict, gpus: list[str] | None, model: str | None, logs: Path) -> Server | Replicas:
    return Replicas(profile, gpus, model=model, log_dir=logs) if gpus else Server(profile, model=model, log_dir=logs)


def cmd_judge(args):
    mode = MODES[args.mode]
    gpus = args.gpus.split(",") if args.gpus else None
    profile = load_profile(args.server) if args.server else mode_profile(args.mode)
    replicas = len(gpus) // int(profile["args"].get("tensor-parallel-size", 1)) if gpus else 1
    args.concurrency = args.concurrency or mode.concurrency * max(replicas, 1)
    args.timeout = args.timeout or mode.timeout
    if args.base_url:
        asyncio.run(judge_with(args, args.base_url, args.served_name))
        return
    with servers(profile, gpus, args.model, Path(args.out).parent / "logs") as server:
        asyncio.run(judge_with(args, server.base_url, server.served_name, server))


def cmd_serve(args):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # leave through the with block, which stops vLLM
    profile = load_profile(args.server) if args.server else mode_profile(args.mode)
    with Server(profile, model=args.model, port=args.port) as server:
        print(json.dumps(server.memory_report(), indent=1))
        try:
            server.process.wait()
        except KeyboardInterrupt:
            pass


def cmd_checklist(args):
    rows = [json.loads(line) for line in open(args.data) if line.strip()]

    async def go(url: str, name: str, extra_body: dict | None) -> int:
        async with Client(url, name, concurrency=args.concurrency, timeout=600, api_key=api_key(args)) as client:
            return await write_checklists(rows, client, args.out, extra_body)

    if args.base_url:
        failed = asyncio.run(go(args.base_url, args.served_name, None))
    else:   # the fast modes' server, thinking off and greedy
        extra_body = {**judges.GREEDY, "chat_template_kwargs": {"enable_thinking": False}}
        with Server(load_profile(args.server or MODES["fast"].profile), model=args.model,
                    log_dir=Path(args.out).parent / "logs") as server:
            failed = asyncio.run(go(server.base_url, server.served_name, extra_body))
    print(f"{args.out}: {len(rows) - failed} of {len(rows)} rows written", flush=True)


def cmd_compare(args):
    reference = metrics.load(args.reference)
    print(f"reference: {args.reference} ({len(reference)} images)")
    for path in args.results:
        r = metrics.compare(metrics.load(path), reference)
        fs, req = r["full_success"], r.get("requirements")
        line = (f"{Path(path).stem:28s} n={r['images']:5d}  full {fs['judge']:.3f} vs {fs['reference']:.3f}  "
                f"agree {fs['agreement'][0]:.3f}  kappa {fs['kappa']:.3f}  leniency {fs['leniency'][0]:+.3f}")
        if req:
            line += f"  | requirements agree {req['agreement'][0]:.3f} kappa {req['kappa']:.3f}"
        elif "auroc" in fs:
            line += f"  | AUROC {fs['auroc']:.3f}"
        print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fastqwenjudge")
    sub = parser.add_subparsers(dest="command", required=True)

    j = sub.add_parser("judge", help="judge a dataset")
    j.add_argument("--data", required=True, help="items JSONL")
    j.add_argument("--out", required=True, help="results JSONL (appended; reruns skip finished images)")
    j.add_argument("--mode", default="fast", choices=list(MODES),
                   help="fast / quality: Qwen3.8-27B on one GPU, thinking off / on; flash-fast / flash-quality: "
                        "Qwen3.8-Flash-Next on two GPUs")
    j.add_argument("--judge", choices=sorted(judges.JUDGES), help="override the mode's judge")
    target = j.add_mutually_exclusive_group()
    target.add_argument("--server", help="server profile (YAML path or bundled name) instead of the mode's")
    target.add_argument("--base-url", help="an already running OpenAI-compatible endpoint")
    j.add_argument("--gpus", help="comma-separated GPU ids: one server per GPU, or per pair for the flash modes "
                                  "(default: one server on the visible GPUs)")
    j.add_argument("--model", help="model path or id (overrides the profile)")
    j.add_argument("--served-name", default="judge", help="model name at --base-url")
    j.add_argument("--concurrency", type=int, help="requests in flight (default: the mode's)")
    j.add_argument("--timeout", type=float, help="seconds per request (default: the mode's)")
    j.add_argument("--objective", default="makespan", choices=sorted(OBJECTIVES),
                   help="makespan: longest checklists first (shortest batch); latency: input order, first come "
                        "first served")
    j.add_argument("--groups", nargs="*")
    j.add_argument("--limit", type=int)
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("serve", help="run a vLLM server from a profile until interrupted")
    s.add_argument("--mode", default="fast", choices=list(MODES), help="the mode the server is for (its model and "
                                                                       "server settings; fast and quality share them)")
    s.add_argument("--server", help="server profile (YAML path or bundled name) instead of the mode's")
    s.add_argument("--model")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    k = sub.add_parser("checklist", help="write checklists for prompts that have none (docs/checklists.md)")
    k.add_argument("--data", required=True, help="JSONL rows with id and prompt (other fields are copied)")
    k.add_argument("--out", required=True, help="items JSONL (appended; reruns skip finished ids)")
    target = k.add_mutually_exclusive_group()
    target.add_argument("--server", help="server profile instead of the fast modes' Qwen3.8-27B")
    target.add_argument("--base-url", help="an already running OpenAI-compatible endpoint (OPENAI_API_KEY if it "
                                           "needs a key)")
    k.add_argument("--model", help="model path or id (overrides the profile)")
    k.add_argument("--served-name", default="judge", help="model name at --base-url")
    k.add_argument("--concurrency", type=int, default=64)
    k.set_defaults(fn=cmd_checklist)

    c = sub.add_parser("compare", help="agreement of result files with a reference")
    c.add_argument("--reference", required=True)
    c.add_argument("results", nargs="+")
    c.set_defaults(fn=cmd_compare)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
