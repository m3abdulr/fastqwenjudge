"""The two ways to run the judge, each a judge plus the server profile and client settings it was measured with.

  fast     the neutral judge on Qwen3-VL-32B-Instruct-FP8: three short requests per image, the observe request
           never sees the requirements. The faster mode, and the less lenient one (a better reward signal).
  quality  one request per image on Qwen3-VL-32B-Thinking-FP8, which reasons before it answers. About 1.2x the
           fast mode's time on a 1,000-image batch, and closer to the frontier judges, most of all on prompts that
           need reasoning; more lenient (it passes more images).

    results = judge("items.jsonl", "results.jsonl", mode="quality")
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from . import judges
from .client import Client
from .data import Item, load_items
from .runner import run
from .server import Replicas, Server, load_profile


@dataclass(frozen=True)
class Mode:
    judge: str
    profile: str        # a server profile shipped in profiles/
    concurrency: int    # requests in flight
    timeout: float      # seconds per request; a reasoning chain can run for minutes in a full batch


MODES = {
    "fast": Mode(judge="neutral", profile="fast", concurrency=256, timeout=600),
    "quality": Mode(judge="think", profile="quality", concurrency=128, timeout=3600),
}


def make_judge(mode: str, thinking_budget: int | None = None):
    """The mode's judge. `thinking_budget` (quality mode) replaces the default 8,192 thinking tokens per image;
    -1 removes the budget, leaving the 16,384-token output limit."""
    j = judges.make(MODES[mode].judge)
    if thinking_budget is not None:
        if not hasattr(j, "thinking_budget"):
            raise ValueError(f"a thinking budget applies to the quality mode, not {mode!r}")
        j.thinking_budget = None if thinking_budget < 0 else thinking_budget
    return j


def judge(data: str | Path | list[Item], out: str | Path, mode: str = "fast", base_url: str | list[str] | None = None,
          gpus: list[str] | None = None, served_name: str = "judge", log_dir: str | Path | None = None,
          thinking_budget: int | None = None, model: str | None = None) -> dict[str, dict]:
    """Judge every item (a JSONL file or a list of Items) in `mode` and return {image id: result}. Without
    `base_url` a vLLM server is started with the mode's profile and stopped afterwards, one per GPU in `gpus` if
    given (requests go to the server that already holds the prompt's prefix); with it, the server(s) must already
    serve the mode's model (`fastqwenjudge serve --mode ...`). Results are appended to `out` as they finish, so a
    rerun skips the images already judged. `model` replaces the profile's model id (a local checkpoint folder on a
    machine without internet access); `thinking_budget`: see `make_judge`."""
    settings = MODES[mode]
    items = load_items(data) if isinstance(data, (str, Path)) else data
    j = make_judge(mode, thinking_budget)

    async def go(url: str | list[str], name: str) -> dict[str, dict]:
        servers = len(url) if isinstance(url, list) else 1
        async with Client(url, name, concurrency=settings.concurrency * servers, timeout=settings.timeout) as client:
            return await run(j, items, client, out)

    if base_url:
        return asyncio.run(go(base_url, served_name))
    profile, logs = load_profile(settings.profile), log_dir or Path(out).parent / "logs"
    with (Replicas(profile, gpus, model=model, log_dir=logs) if gpus
          else Server(profile, model=model, log_dir=logs)) as server:
        return asyncio.run(go(server.base_url, server.served_name))
