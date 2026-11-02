"""Checklists for prompts that come without one. The judges need each prompt as a list of statements; benchmarks
usually ship questions or checklists of their own, and for the rest we wrote them with the prompt in
prompts/checklist.txt on Gemini 3.8 Flash (docs/checklists.md). This runs the same prompt on any OpenAI-compatible
model: by default Qwen3.8-27B on a local server, thinking off, greedy. Checklists from the local model have not
been compared with Gemini's.

    fastqwenjudge checklist --data prompts.jsonl --out items.jsonl
    fastqwenjudge checklist --base-url <url> --served-name <model> --data prompts.jsonl --out items.jsonl

With --base-url, an endpoint that needs a key gets OPENAI_API_KEY from the environment.

Input rows need `id` and `prompt`; every other field (`image`, `group`, ...) is copied. Rows that already have
`requirements` are copied as they are. Output rows are appended as they finish, so a rerun skips finished ids.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from . import prompts
from .client import Client


def validate(answer: dict) -> list[str]:
    requirements = [r.strip() for r in answer.get("requirements", []) if isinstance(r, str) and r.strip()]
    if not requirements:
        raise ValueError("Expected at least one requirement")
    if any(r.endswith("?") for r in requirements):
        raise ValueError("Requirements must be declarative statements, not questions")
    return requirements


async def write_checklists(rows: list[dict], client: Client, out: str | Path, extra_body: dict | None = None,
                           progress_every: int = 50) -> int:
    """Append one row per input row to `out`, with `requirements` added; returns how many rows failed."""
    out = Path(out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    system = prompts.load("checklist.txt")
    failed = finished = 0
    with out.open("a") as sink:
        async def one(row: dict):
            nonlocal failed, finished
            if row.get("requirements"):
                sink.write(json.dumps(row) + "\n")
            else:
                try:
                    reply = await client.chat_json("checklist", system,
                                                   [{"type": "text", "text": "PROMPT:\n" + row["prompt"]}],
                                                   prompts.CHECKLIST, max_tokens=2048, validate=validate,
                                                   extra_body=extra_body)
                    sink.write(json.dumps({**row, "requirements": reply.answer}) + "\n")
                except Exception as error:
                    failed += 1
                    print(f"{row['id']}: {type(error).__name__}: {str(error)[:200]}", flush=True)
            sink.flush()
            finished += 1
            if finished % progress_every == 0 or finished == len(todo):
                print(f"checklists {finished}/{len(todo)} | failed {failed}", flush=True)

        await asyncio.gather(*(one(r) for r in todo))
    return failed
