"""The four ways to run the judge, each a judge plus the server profile and client settings it was measured with.
All four make one request per image with the prompt, the checklist and the image.

  fast           Qwen3.8-27B (FP8), one H100. Thinking off, the strict prompt, and a statement passes only if the
                 model gives "true" a probability of at least 0.999. The cheapest mode.
  quality        Qwen3.8-27B (FP8), one H100. Medium reasoning effort, the strict prompt. Closer to the frontier
                 judges, most of all on prompts that need reasoning; more lenient than the others.
  flash-fast     Qwen3.8-Flash-Next (NVFP4), two H100s. Thinking off, the plain prompt, cut-off 0.998. The least
                 lenient mode.
  flash-quality  Qwen3.8-Flash-Next (NVFP4), two H100s. Medium reasoning effort, the strict prompt. The closest to
                 the frontier judges.

fast and quality share a server (profiles/27b.yaml): thinking is switched per request. flash-fast and flash-quality
share a model and profile (profiles/flash_next.yaml) but not their server settings: flash-fast starts its server with
112 sequences instead of the profile's 96 (5% shorter batches; for quality the batch's tail of long chains sets the
time and 96 is as fast), so serve the mode you will judge with. Numbers: README.md,
docs/results.md, docs/serving.md.

    results = judge("items.jsonl", "results.jsonl", mode="quality")
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from . import judges
from .client import Client
from .data import Item, load_items
from .runner import OBJECTIVES, run
from .server import Replicas, Server, load_profile


@dataclass(frozen=True)
class Mode:
    judge: str
    profile: str        # a server profile shipped in profiles/
    concurrency: int    # requests in flight from the client (more than the server runs at once; it queues the rest)
    timeout: float      # seconds per request; a reasoning chain can run for minutes in a full batch
    server_args: dict = field(default_factory=dict)   # this mode's changes to the profile's server flags


MODES = {
    "fast": Mode(judge="fast", profile="27b", concurrency=256, timeout=600),
    "quality": Mode(judge="quality", profile="27b", concurrency=128, timeout=3600),
    "flash-fast": Mode(judge="flash-fast", profile="flash_next", concurrency=256, timeout=600,
                       server_args={"max-num-seqs": 112}),
    "flash-quality": Mode(judge="flash-quality", profile="flash_next", concurrency=128, timeout=3600),
}


def make_judge(mode: str):
    return judges.make(MODES[mode].judge)


def mode_profile(mode: str) -> dict:
    """The server profile a mode runs with: its profile file plus the mode's own server flags."""
    return load_profile(MODES[mode].profile, MODES[mode].server_args)


def judge(data: str | Path | list[Item], out: str | Path, mode: str = "fast", base_url: str | list[str] | None = None,
          gpus: list[str] | None = None, served_name: str = "judge", log_dir: str | Path | None = None,
          model: str | None = None, objective: str = "makespan") -> dict[str, dict]:
    """Judge every item (a JSONL file or a list of Items) in `mode` and return {image id: result}. Without
    `base_url` a vLLM server is started with the mode's profile and stopped afterwards; `gpus` runs one server per
    group of GPUs the profile needs (one for fast and quality, two for the flash modes) and spreads the images over
    them. With `base_url`, the server(s) must already serve the mode's model (`fastqwenjudge serve --mode ...`).
    Results are appended to `out` as they finish, so a rerun skips the images already judged. `model` replaces the
    profile's model id (a local checkpoint folder on a machine without internet access). `objective` orders
    the images as the CLI's --objective does (makespan: the longest checklists first)."""
    settings = MODES[mode]
    items = load_items(data) if isinstance(data, (str, Path)) else data
    j = make_judge(mode)

    async def go(url: str | list[str], name: str) -> dict[str, dict]:
        servers = len(url) if isinstance(url, list) else 1
        schedule = OBJECTIVES[objective]
        async with Client(url, name, concurrency=settings.concurrency * servers, timeout=settings.timeout,
                          admission=schedule["admission"]) as client:
            return await run(j, items, client, out, order=schedule["order"])

    if base_url:
        return asyncio.run(go(base_url, served_name))
    profile, logs = mode_profile(mode), log_dir or Path(out).parent / "logs"
    with (Replicas(profile, gpus, model=model, log_dir=logs) if gpus
          else Server(profile, model=model, log_dir=logs)) as server:
        return asyncio.run(go(server.base_url, server.served_name))
