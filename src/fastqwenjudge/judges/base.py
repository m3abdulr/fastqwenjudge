from __future__ import annotations

from typing import Protocol

from ..client import Client
from ..data import Item


class Judge(Protocol):
    name: str

    async def judge(self, item: Item, client: Client) -> dict:
        """Return {"items": [{"requirement", "evidence", "satisfied"}], "score", "all_satisfied", ...}."""
        ...


def verdicts(answer: dict, requirements) -> dict:
    """Check a checklist answer: one item per requirement, ids 1..n in order, boolean decisions, evidence."""
    if not requirements:
        raise ValueError("Empty checklist")
    answers = answer.get("items") if isinstance(answer, dict) else None
    if not isinstance(answers, list) or len(answers) != len(requirements):
        raise ValueError("Expected exactly one item per requirement")
    items = []
    for number, (a, requirement) in enumerate(zip(answers, requirements, strict=True), 1):
        if (not isinstance(a, dict) or type(a.get("id")) is not int or a["id"] != number
                or type(a.get("satisfied")) is not bool
                or not isinstance(a.get("evidence"), str) or not a["evidence"].strip()):
            raise ValueError(f"Invalid judgment item {number}")
        items.append({"id": number, "requirement": requirement, "evidence": a["evidence"],
                      "satisfied": a["satisfied"]})
    satisfied = sum(i["satisfied"] for i in items)
    return {"items": items, "score": satisfied / len(items), "all_satisfied": satisfied == len(items)}
