"""Judge inputs: one generated image, the prompt it came from, and a checklist of requirements.

Datasets are JSONL files, one image per line:

    {"id": "geneval_0359_original", "prompt": "a photo of a stop sign above a fork",
     "requirements": ["The image clearly shows at least one \"fork\".", ...],
     "image": "images/geneval_0359_original.png", "group": "geneval", "variant": "original"}

`image` is resolved relative to the JSONL file. `group` and `variant` are free-form labels used in reports.
Requirements should be declarative statements, not questions (docs/checklists.md).
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    requirements: tuple[str, ...]
    image: Path
    group: str = ""
    variant: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def checklist_key(self) -> str:
        """Items with the same checklist share this key."""
        return short_hash(list(self.requirements))

    def image_url(self) -> str:
        data = self.image.read_bytes()
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def short_hash(value) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def load_items(path: str | Path, groups: list[str] | None = None, limit: int | None = None) -> list[Item]:
    path = Path(path)
    items = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if groups and row.get("group") not in groups:
                continue
            requirements = tuple(row["requirements"])
            if not requirements:
                raise ValueError(f"{row['id']}: empty checklist")
            known = {"id", "prompt", "requirements", "image", "group", "variant"}
            items.append(Item(id=row["id"], prompt=row["prompt"], requirements=requirements,
                              image=(path.parent / row["image"]).resolve(), group=row.get("group", ""),
                              variant=row.get("variant", ""),
                              extra={k: v for k, v in row.items() if k not in known}))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return items[:limit] if limit else items


def shuffled_by_checklist(items: list[Item], seed: int = 0) -> list[Item]:
    """Shuffle with a fixed seed, keeping items that share a checklist next to each other (as a judge sees them
    when grading several images per prompt)."""
    groups: dict[str, list[Item]] = {}
    for i in items:
        groups.setdefault(i.checklist_key, []).append(i)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    return [i for g in order for i in g]
