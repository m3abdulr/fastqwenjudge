"""On-policy training data for a draft model: the judge's own requests and the served model's answers.

    python experiments/speculator/traces.py --profile configs/servers/tuned_fp8.yaml --set kv-cache-dtype=fp8 \
        --data data/items/speculator_train.jsonl --out experiments/speculator/data

Runs the neutral judge over images that no evaluation set uses (split.py) and writes every request it made, with
the answer the server returned, as one conversation per request in the speculators library's format
(<out>/conversations.jsonl): system prompt, user turn (text, and the image as a file path), assistant answer.
Retries are kept, since the judge really sends them. The answers come from the model as it is served (greedy,
constrained to the same JSON schemas), which is exactly what a draft model has to predict.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import yaml

from fastqwenjudge import judges
from fastqwenjudge.client import Client
from fastqwenjudge.data import load_items
from fastqwenjudge.runner import run
from fastqwenjudge.server import Server, load_profile


def digest(url: str) -> str:
    return hashlib.blake2b(url.encode(), digest_size=16).hexdigest()


class Recorder(Client):
    """A Client that appends each answered request to a JSONL file as a conversation."""

    def __init__(self, *args, out: Path, images: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.file = out.open("a")
        self.images = images   # digest of an image's data URL -> its file

    def parts(self, content: str | list[dict]) -> list[dict]:
        # Always a list of parts. If some turns were plain strings, the datasets JSON loader would type the column
        # as generic JSON and parse the answers, which are JSON strings, into objects; speculators then keeps only
        # their keys as the text.
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return [{"type": "image", "path": self.images[digest(p["image_url"]["url"])]} if p["type"] == "image_url"
                else p for p in content]

    async def _create(self, stage: str, **payload) -> dict:
        raw = await super()._create(stage, **payload)
        choice = raw["choices"][0]
        turns = [{**m, "content": self.parts(m["content"])} for m in payload["messages"]]
        turns.append({"role": "assistant", "content": self.parts(choice["message"]["content"])})
        self.file.write(json.dumps({"stage": stage, "finish_reason": choice.get("finish_reason"),
                                    "conversations": turns}) + "\n")
        self.file.flush()
        return raw

    async def close(self):
        self.file.close()
        await super().close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--model")
    p.add_argument("--set", nargs="*", default=[], help="server arg overrides, key=value (YAML values)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=256)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data)
    images = {digest(item.image_url()): str(item.image) for item in items}
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    with Server(load_profile(args.profile, overrides), model=args.model, log_dir=out / "logs") as server:
        async def go():
            # an interrupted run resumes from the results file; requests of images in flight at the time are
            # recorded again when those images are redone
            async with Recorder(server.base_url, server.served_name, concurrency=args.concurrency,
                                out=out / "conversations.jsonl", images=images) as client:
                await run(judges.make("neutral"), items, client, out / "neutral.jsonl")
        asyncio.run(go())


if __name__ == "__main__":
    main()
