"""fastqwenjudge judge | serve | compare

    fastqwenjudge judge --data items.jsonl --out results.jsonl                  # fast mode, starts vLLM
    fastqwenjudge judge --mode quality --data items.jsonl --out results.jsonl
    fastqwenjudge judge --mode quality --thinking-budget -1 ...                 # let the model reason to the limit
    fastqwenjudge judge --gpus 0,1,2,3 --data items.jsonl --out results.jsonl  # one server per GPU
    fastqwenjudge serve --mode fast --port 8000                                 # or keep a server running ...
    fastqwenjudge judge --base-url http://127.0.0.1:8000/v1 --data items.jsonl --out results.jsonl
    fastqwenjudge compare --reference sol.jsonl results.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from . import judges, metrics
from .client import Client
from .data import load_items
from .modes import MODES, make_judge
from .runner import run
from .server import Replicas, Server, load_profile
from .throughput import Monitor, request_summary

# How requests are ordered on the client. "latency": images start in input order and requests are admitted in
# arrival order, which is fair to every image. "makespan": the longest checklists start first and requests of
# images already in progress are admitted before new images start. That shortens the batch's tail after 90% of it
# is done, but in the fast mode the whole batch took 3-9% longer than in input order (docs/serving.md).
OBJECTIVES = {"latency": {"order": "input", "admission": "fifo"},
              "makespan": {"order": "longest-first", "admission": "rank"}}


async def judge_with(args, base_url: str | list[str], model: str, server: Server | Replicas | None = None) -> dict:
    items = load_items(args.data, groups=args.groups, limit=args.limit)
    judge = judges.make(args.judge) if args.judge else make_judge(args.mode, args.thinking_budget)
    started = time.monotonic()
    schedule = OBJECTIVES[args.objective]
    async with Client(base_url, model, concurrency=args.concurrency, admission=schedule["admission"],
                      timeout=args.timeout) as client, Monitor(base_url) as monitor:
        results = await run(judge, items, client, args.out, mode=args.execution, order=schedule["order"])
    stats = {"judge": judge.name, "mode": args.mode, "thinking_budget": getattr(judge, "thinking_budget", None),
             "execution": args.execution, "objective": args.objective,
             "concurrency": args.concurrency, "images": len(items),
             "judged": len(results), "wall_seconds": time.monotonic() - started,
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


def cmd_judge(args):
    mode = MODES[args.mode]
    gpus = args.gpus.split(",") if args.gpus else None
    args.concurrency = args.concurrency or mode.concurrency * (len(gpus) if gpus else 1)
    args.timeout = args.timeout or mode.timeout
    if args.base_url:
        asyncio.run(judge_with(args, args.base_url, args.served_name))
        return
    profile, logs = load_profile(args.server or mode.profile), Path(args.out).parent / "logs"
    with (Replicas(profile, gpus, model=args.model, log_dir=logs) if gpus
          else Server(profile, model=args.model, log_dir=logs)) as server:
        asyncio.run(judge_with(args, server.base_url, server.served_name, server))


def cmd_serve(args):
    with Server(load_profile(args.server or MODES[args.mode].profile), model=args.model, port=args.port) as server:
        print(json.dumps(server.memory_report(), indent=1))
        try:
            server.process.wait()
        except KeyboardInterrupt:
            pass


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
    j.add_argument("--mode", default="fast", choices=sorted(MODES),
                   help="fast: neutral judge on the Instruct model; quality: the Thinking model in one call")
    j.add_argument("--judge", choices=sorted(judges.JUDGES), help="override the mode's judge")
    j.add_argument("--thinking-budget", type=int,
                   help="quality mode: thinking tokens per image before the server ends the reasoning (default "
                        "8192; -1 for none)")
    target = j.add_mutually_exclusive_group()
    target.add_argument("--server", help="server profile (YAML path or bundled name) instead of the mode's")
    target.add_argument("--base-url", help="an already running OpenAI-compatible endpoint")
    j.add_argument("--gpus", help="comma-separated GPU ids: one server per GPU (default: one server, GPU 0)")
    j.add_argument("--model", help="model path or id (overrides the profile)")
    j.add_argument("--served-name", default="judge", help="model name at --base-url")
    j.add_argument("--execution", default="pipelined", choices=["pipelined", "staged"],
                   help="pipelined: each image moves on as soon as its previous request returns")
    j.add_argument("--concurrency", type=int, help="requests in flight (default: the mode's)")
    j.add_argument("--timeout", type=float, help="seconds per request (default: the mode's)")
    j.add_argument("--objective", default="latency", choices=sorted(OBJECTIVES),
                   help="latency: input order, first come first served (faster in our tests); makespan: longest "
                        "checklists first, a shorter tail")
    j.add_argument("--groups", nargs="*")
    j.add_argument("--limit", type=int)
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("serve", help="run a vLLM server from a profile until interrupted")
    s.add_argument("--mode", default="fast", choices=sorted(MODES))
    s.add_argument("--server", help="server profile (YAML path or bundled name) instead of the mode's")
    s.add_argument("--model")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    c = sub.add_parser("compare", help="agreement of result files with a reference")
    c.add_argument("--reference", required=True)
    c.add_argument("results", nargs="+")
    c.set_defaults(fn=cmd_compare)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
