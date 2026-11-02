"""Judge variants on Qwen3.8 for the agreement experiments: the four modes and every variant behind the numbers in
docs/results.md (baselines, prompts, sampling, reasoning effort, the 0-10 judge), each judging the same images on
one server, one after another.

    python experiments/qwen38/run.py --profile 27b --model <checkpoint> --data <items>/q38_dev.jsonl \
        --judges off_greedy think_medium_strict --out <results>/<run>

Every request says whether the model thinks (the checkpoints do by default). Before each judge, one request checks
that thinking really is off or on. Timings (batch time, server metrics, GPU use, client latencies) go to
<out>/timings_<time>.json; verdicts to <out>/<judge>.jsonl.

Sampling: the model cards' settings unless a variant says otherwise (thinking off: temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5; thinking on: temperature 1.0, top_p 0.95, top_k 20, effort xhigh by default), and
greedy for the "greedy" variants. Variants with p_true record the probability of "true" at each verdict, for the
cut-off chosen on development images (the fast modes apply it inside the judge).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from fastqwenjudge import prompts
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items, shuffled_by_checklist
from fastqwenjudge.judges import GREEDY, THINKING, SingleCallJudge
from fastqwenjudge.judges.base import verdicts
from fastqwenjudge.judges.single import user_text
from fastqwenjudge.runner import OBJECTIVES, run
from fastqwenjudge.server import Server, load_profile
from fastqwenjudge.throughput import GpuMonitor, Monitor, request_summary

HERE = Path(__file__).resolve().parent
CARD_OFF = {"top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


class LocalSingleJudge(SingleCallJudge):
    """The single call with a prompt from experiments/qwen38/prompts/ when the package has none of that name:
    "look" (describe the image in a "description" field before the verdicts), "doubt" (an "against" field per
    verdict, what does not match, before the decision), "v2" (the Qwen3-VL release's rules for counts, negation and
    "every")."""

    def __init__(self, prompt: str, **kwargs):
        local = HERE / "prompts" / f"single_{prompt}.txt"
        super().__init__("v1" if local.exists() else prompt, **kwargs)
        if local.exists():
            self.system = local.read_text()
        self.describe, self.doubt = prompt == "look", prompt == "doubt"
        if self.describe:
            self.schema = prompts.obj({"description": {"type": "string"},
                                       "items": prompts.VERDICTS["properties"]["items"]})
        elif self.doubt:
            self.schema = prompts.array_of("items", {"id": {"type": "integer"}, "evidence": {"type": "string"},
                                                     "against": {"type": "string"},
                                                     "satisfied": {"type": "boolean"}})

    def validate(self, answer: dict, requirements) -> dict:
        out = verdicts(answer, requirements)
        if self.doubt:
            for kept, raw in zip(out["items"], answer["items"], strict=True):
                kept["against"] = raw.get("against", "")
        if self.describe:
            out["description"] = answer.get("description", "")
        return out


SCORE10 = prompts.obj({"problems": {"type": "array", "items": {"type": "string"}},
                       "score": {"type": "integer", "enum": list(range(11))}})


class Score10Judge:
    """No checklist: the prompt and the image only (the usual rate-it-out-of-10 baseline). The model lists what the
    image gets wrong, then scores it 0-10 (prompts/score10.txt). The result keeps the problems and the score, not
    the checklist output format; a pass mark is applied when scoring (pass = score >= a mark set on development
    images)."""

    def __init__(self, name: str, thinking: bool, effort: str | None = None, max_tokens: int = 8192,
                 temperature: float = 0.0, sampling: dict | None = None):
        self.name = name
        self.system = (HERE / "prompts" / "score10.txt").read_text()
        self.max_tokens, self.temperature = max_tokens, temperature
        switch = {"enable_thinking": thinking, **({"reasoning_effort": effort} if effort else {})}
        self.extra_body = {**(sampling or {}), "chat_template_kwargs": switch}

    async def judge(self, item, client: Client) -> dict:
        content = [{"type": "text", "text": "PROMPT:\n" + item.prompt + "\n\nIMAGE:"},
                   {"type": "image_url", "image_url": {"url": item.image_url()}}]

        def validate(answer: dict) -> dict:
            if not isinstance(answer.get("score"), int) or not 0 <= answer["score"] <= 10:
                raise ValueError(f"score must be an integer 0-10, got {answer.get('score')!r}")
            return {"problems": [str(p) for p in answer.get("problems", [])], "score": answer["score"]}

        reply = await client.chat_json(self.name, self.system, content, SCORE10, max_tokens=self.max_tokens,
                                       validate=validate, temperature=self.temperature, extra_body=self.extra_body)
        return {**reply.answer, "requests": reply.attempts}


@dataclass
class Variant:
    make: Callable[[], object]
    concurrency: int    # requests in flight from the client (the shipped modes': 256 thinking off, 128 on)
    timeout: float      # seconds per request


def off(name, prompt="v1", greedy=True, sampling=None, p_true=None):
    """Thinking off: greedy (with p_true) or the card's sampling."""
    if greedy:
        return Variant(lambda: LocalSingleJudge(prompt, thinking=False, temperature=0.0, sampling=GREEDY,
                                                p_true=True if p_true is None else p_true, name=name), 256, 600)
    return Variant(lambda: LocalSingleJudge(prompt, thinking=False, temperature=0.7, sampling=sampling or CARD_OFF,
                                            p_true=bool(p_true), name=name), 256, 600)


def think(name, prompt="v1", effort="xhigh", p_true=False, budget=None):
    return Variant(lambda: LocalSingleJudge(prompt, thinking=True, effort=effort, temperature=1.0, sampling=THINKING,
                                            max_tokens=32768, p_true=p_true, thinking_budget=budget, name=name),
                   128, 3600)


JUDGES = {
    # the baselines scored on the held-out set on 2026-10-29 (the neutral pipeline baseline ran the Qwen3-VL
    # release's three-request judge, tag qwen3vl-v1)
    "single_off": off("single_off", greedy=False),
    "think_on": think("think_on"),
    # development variants, thinking off
    "off_nopres": off("off_nopres", greedy=False, sampling={**CARD_OFF, "presence_penalty": 0.0}),
    "off_greedy": off("off_greedy"),
    "off_greedy_v2": off("off_greedy_v2", "v2"),
    "off_greedy_look": off("off_greedy_look", "look"),
    "off_card_look": off("off_card_look", "look", greedy=False),
    "off_greedy_strict": off("off_greedy_strict", "strict"),
    "off_greedy_doubt": off("off_greedy_doubt", "doubt"),
    # development variants, thinking on: reasoning effort, and the near-miss prompts
    "think_medium": think("think_medium", effort="medium"),
    "think_low": think("think_low", effort="low"),
    **{f"think_{effort}_{prompt}": think(f"think_{effort}_{prompt}", prompt, effort, p_true=True)
       for effort in ("xhigh", "medium") for prompt in ("strict", "doubt")},
    # the shipped quality judges send no log-probability request (the variants above do), with and without the
    # thinking budget: runaway chains showed up only in runs without it (2026-10-31)
    "think_medium_strict_nolp": think("think_medium_strict_nolp", "strict", "medium"),
    "think_medium_strict_budget": think("think_medium_strict_budget", "strict", "medium", budget=8192),
    # no checklist, a score out of 10
    "score10_off": Variant(lambda: Score10Judge("score10_off", thinking=False, sampling=GREEDY), 256, 600),
    "score10_medium": Variant(lambda: Score10Judge("score10_medium", thinking=True, effort="medium",
                                                   max_tokens=32768, temperature=1.0, sampling=THINKING), 128, 3600),
}


async def preflight(server, item, judge) -> dict:
    """One request with the judge's sampling and switches; fails loudly if thinking is not what was asked for."""
    switch = judge.extra_body["chat_template_kwargs"]
    api = AsyncOpenAI(base_url=server.base_url, api_key="EMPTY", timeout=3600)
    try:
        r = await api.chat.completions.create(
            model=server.served_name, max_tokens=32768, seed=0, temperature=judge.temperature,
            extra_body=judge.extra_body, messages=[
                {"role": "system", "content": prompts.load("single_v1.txt")},
                {"role": "user", "content": [{"type": "text", "text": user_text(item.prompt, item.requirements)},
                                             {"type": "image_url", "image_url": {"url": item.image_url()}}]}])
    finally:
        await api.close()
    message = r.choices[0].message.model_dump()
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    check = {"thinking": switch["enable_thinking"], "effort": switch.get("reasoning_effort"),
             "reasoning_chars": len(reasoning), "completion_tokens": r.usage.completion_tokens,
             "finish_reason": r.choices[0].finish_reason}
    print("preflight", json.dumps(check), flush=True)
    if switch["enable_thinking"] != bool(reasoning) or "<think>" in (message.get("content") or ""):
        raise RuntimeError(f"thinking={switch['enable_thinking']} but the reply has {len(reasoning)} reasoning chars")
    return check


def throttled(gpu: dict) -> bool:
    """Under a full batch an H100 SXM here averaged 1,690-1,980 MHz (the 27B at full load sits near 1,700 MHz, held
    there by the 700 W limit; short kernel benchmarks dipped to 1,365 MHz). A hot or power-capped one runs far lower,
    and every speed number from it is off (one node ran at 990 MHz and 86 C, 35-55% slower than the same settings
    elsewhere)."""
    return (gpu.get("sm_clock_mhz") or 2000) < 1500 or (gpu.get("temperature_c") or 0) > 84


async def run_all(server, items, names, out: Path, max_concurrency: int | None, seed: int | None = 0,
                  objective: str = "latency", chunks: int = 1) -> dict:
    """Each judge over the items as one batch, or as `chunks` consecutive batches each run to its end (as a caller
    with small batches sees it; each one's time recorded). `objective` as the CLI's: latency (input order, first
    come first served) or makespan (longest checklists first, admitted by rank). Every request's start (seconds
    from the judge's start), time and tokens go to <out>/<judge>_requests.jsonl, for the batch's tail."""
    schedule = OBJECTIVES[objective]
    timings = {}
    for name in names:
        variant = JUDGES[name]
        judge = variant.make()
        concurrency = min(variant.concurrency, max_concurrency or variant.concurrency)
        check = await preflight(server, items[0], judge)
        start = time.monotonic()
        size, chunk_seconds = -(-len(items) // chunks), []
        with GpuMonitor() as gpu:
            async with Client(server.base_url, server.served_name, concurrency=concurrency, timeout=variant.timeout,
                              seed=seed, admission=schedule["admission"]) as client, \
                    Monitor(server.base_url) as monitor:
                for first in range(0, len(items), size):
                    begin = time.monotonic()
                    await run(judge, items[first:first + size], client, out / f"{name}.jsonl",
                              order=schedule["order"], progress_every=25)
                    chunk_seconds.append(time.monotonic() - begin)
        with (out / f"{name}_requests.jsonl").open("w") as sink:
            for r in client.log:
                sink.write(json.dumps({"start": r.start - start, "seconds": r.latency, "rank": r.rank, "ok": r.ok,
                                       "prompt_tokens": r.prompt_tokens,
                                       "completion_tokens": r.completion_tokens}) + "\n")
        timings[name] = {"wall_seconds": time.monotonic() - start, "chunk_seconds": chunk_seconds,
                         "objective": objective, "concurrency": concurrency,
                         "temperature": judge.temperature, "sampling": judge.extra_body, "seed": seed,
                         "p_true": getattr(judge, "p_true", False), "preflight": check, "server": monitor.summary(),
                         "gpu": gpu.summary(), "client": request_summary(client.log)}
        g = timings[name]["gpu"] or {}
        timings[name]["throttled"] = throttled(g)
        warning = (f" (GPU THROTTLED: {g.get('sm_clock_mhz', 0):.0f} MHz, {g.get('temperature_c', 0):.0f} C)"
                   if timings[name]["throttled"] else "")
        print(f"{name}: {timings[name]['wall_seconds']:.0f} s{warning}", flush=True)
    return timings


async def _steady_one(server, judge, timeout: float, items, concurrency: int, warmup: float, seconds: float,
                      seed: int | None) -> tuple[int, float, dict, dict]:
    stream = itertools.cycle(items)
    done: list[float] = []
    stop = asyncio.Event()
    async with Client(server.base_url, server.served_name, concurrency=concurrency, timeout=timeout,
                      seed=seed) as client:
        async def worker():
            while not stop.is_set():
                try:
                    await judge.judge(next(stream), client)
                    done.append(time.monotonic())
                except Exception:
                    pass

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.sleep(warmup)
        begin = time.monotonic()
        with GpuMonitor() as gpu:
            async with Monitor(server.base_url) as monitor:
                await asyncio.sleep(seconds)
        end = time.monotonic()
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks, timeout=120)
    return sum(begin <= t < end for t in done), end - begin, monitor.summary(), gpu.summary() or {}


async def run_steady(server, items, names, concurrency: int, warmup: float, seconds: float, seed: int | None = 0,
                     seed_shuffle: int = 0) -> dict:
    """Steady-state throughput: `concurrency` images in flight, each replaced by the next as it finishes (the items
    cycled, shuffled), measured over `seconds` after `warmup`. Unlike a batch's time, no tail of long reasoning
    chains: images/s, server token rates and GPU use at a fixed load. Verdicts are not kept."""
    out = {}
    for name in names:
        variant = JUDGES[name]
        n, elapsed, server_summary, g = await _steady_one(
            server, variant.make(), variant.timeout, shuffled_by_checklist(items, seed_shuffle), concurrency, warmup,
            seconds, seed)
        hot = throttled(g)
        out[name] = {"steady": True, "concurrency": concurrency, "warmup": warmup, "seconds": elapsed, "images": n,
                     "images_per_s": n / elapsed, "server": server_summary, "gpu": g, "throttled": hot}
        print(f"{name}: steady {n / elapsed:.2f} images/s at {concurrency} in flight; "
              f"{server_summary.get('generation_tokens_per_s', 0):.0f} generated + "
              f"{server_summary.get('prompt_tokens_per_s', 0):.0f} prompt tokens/s; running "
              f"{server_summary.get('mean_running', 0):.0f}" + (" (GPU THROTTLED)" if hot else ""), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, help="a profile YAML or a bundled name (27b, flash_next)")
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--judges", nargs="+", choices=list(JUDGES), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-concurrency", type=int, help="cap on requests in flight")
    p.add_argument("--startup-timeout", type=float, default=3600, help="a first start compiles kernels")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data, limit=args.limit)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    profile = load_profile(args.profile, overrides)
    profile["args"] = {k: v for k, v in profile["args"].items() if v is not None}
    with Server(profile, model=args.model, log_dir=out / "logs", startup_timeout=args.startup_timeout) as server:
        timings = {"startup_seconds": server.startup_seconds}
        timings.update(asyncio.run(run_all(server, items, args.judges, out, args.max_concurrency)))
        timings["memory"] = server.memory_report()
    (out / f"timings_{int(time.time())}.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
