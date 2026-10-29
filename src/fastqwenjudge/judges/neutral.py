"""Neutral grounding: keep the model from seeing what it is supposed to find.

A single-call judge reads "three red apples" and then looks at the image, and it tends to see three red apples.
This judge splits the work so that the request that looks at the image never sees the desired answer:

  compile  (text only, once per checklist)  each requirement -> a neutral observation plan: what kind of check
           it is (count / absence / universal / other), what to inspect, a question that does not assert the
           desired answer, and the expected value kept aside for later. Plans that leak the target count or the
           absence into the question are rejected and the model is asked to fix them. An absence plan for a
           requirement that asks for something to be present becomes a plain observation (`repair_plans`).
  observe  (image + neutral questions only)  evidence, counts, and an `uncertain` flag per question. No prompt,
           no requirements, no expected values.
  decide   (text only)  requirements + observations -> a verdict per requirement.
  guards   (code)  a definite observation that contradicts what the requirement itself states fails it: a count
           the requirement gives, a presence check (at least one), an absence, or "every X" when the requirement
           says all / every / each / both. Guards can only turn a pass into a fail, never the reverse. The first
           version (guards="all", judge neutral_v1) also checked counts and "every X" the compile step inferred
           from unnumbered plurals, and failed every uncertain observation; on new prompt styles most of those
           vetoes were wrong.

Compiled plans depend only on the checklist, so every image of the same prompt (different models, seeds or
checkpoints) shares one compile request.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re

from .. import prompts
from ..client import Client
from ..data import Item
from .base import verdicts

OBSERVATION_FORMATS = {
    "count": "Enumerate distinct visible {subject}; report their count and locations. {question}",
    "universal": "Inspect visible {subject} individually. Report total and matching counts. {question}",
    "other": "{question}",
}
NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
KINDS = ("count", "absence", "universal", "other")
GUARD_RULES = ("all", "stated", "none")
# other ways a requirement states a count of one or two
COUNT_SYNONYMS = {1: ("single",), 2: ("pair", "both")}
QUANTIFIER = re.compile(r"\b(all|every|each|both)\b", re.I)
# words by which a requirement says something is missing, forbidden or excluded
ABSENCE_CUE = re.compile(
    r"\b(no|not|none|nothing|nobody|never|neither|nor|without|absent|absence|missing|free|lacks?|lacking|devoid|"
    r"empty|blank|bare|plain|clear|clean|uncluttered|only|solely|exclusively|alone|except|besides|beyond|other than|"
    r"apart from|avoids?|avoiding|excludes?|excluding|isolated|solitary|sole)\b|n't|\b\w+less\b", re.I)


def check_plans(answer: dict, requirements) -> list[dict]:
    plans = answer.get("plans", [])
    if len(plans) != len(requirements):
        raise ValueError("Wrong number of neutral plans")
    for i, p in enumerate(plans, 1):
        if (type(p.get("id")) is not int or p["id"] != i or p.get("kind") not in KINDS
                or not p.get("subject", "").strip() or not p.get("question", "").strip()):
            raise ValueError("Invalid observation plan")
        if p["kind"] in ("count", "absence"):
            if type(p.get("expected")) is not int or p["expected"] < 0 or p.get("comparator") not in ("eq", "ge", "le"):
                raise ValueError("Invalid count guard")
            if p["kind"] == "absence" and (p["expected"] != 0 or p["comparator"] != "eq"):
                raise ValueError("Absence guard must check zero")
            text = p["subject"] + " " + p["question"]
            tokens = [str(p["expected"])] + ([NUMBER_WORDS[p["expected"]]] if p["expected"] in NUMBER_WORDS else [])
            if any(re.search(r"\b" + re.escape(t) + r"\b", text, re.I) for t in tokens):
                raise ValueError("Target count leaked into neutral question; remove " + "/".join(tokens) +
                                 " from subject/question, while keeping expected unchanged")
            if p["kind"] == "absence" and re.search(r"\b(no|none|absent|absence)\b", text, re.I):
                raise ValueError("Absence target leaked into neutral question")
        elif p.get("expected") is not None or p.get("comparator") != "none":
            raise ValueError("Unexpected numeric guard")
    return plans


def inverted_absence(plan: dict, requirement: str) -> bool:
    """An absence plan for a requirement that names nothing missing or excluded: the compile step read "X is
    visible" as "there is no X", and the absence guard would fail exactly the images that show X."""
    return plan["kind"] == "absence" and not ABSENCE_CUE.search(requirement)


def repair_plans(plans: list[dict], requirements) -> list[dict]:
    """Inverted absence plans become plain observations (kind other), left to the decide step with no guard on that
    requirement; every other plan stays as compiled. Repaired in place rather than retried, because a retry
    recompiles the whole checklist and moves the other plans too. A real absence requirement without a cue word
    ("The background is black.") loses only its guard; a presence requirement with one keeps the old plan."""
    for plan in plans:
        if inverted_absence(plan, requirements[plan["id"] - 1]):
            plan.update(kind="other", comparator="none", expected=None)
    return plans


def questions(plans: list[dict]) -> list[dict]:
    out = []
    for p in plans:
        template = OBSERVATION_FORMATS["count" if p["kind"] in ("count", "absence") else
                                       "universal" if p["kind"] == "universal" else "other"]
        out.append({"id": p["id"], "question": template.format(**p)})
    return out


def check_observations(answer: dict, plans: list[dict]) -> list[dict]:
    observations = answer.get("observations", [])
    if len(observations) != len(plans):
        raise ValueError("Wrong number of observations")
    for i, o in enumerate(observations, 1):
        if (type(o.get("id")) is not int or o["id"] != i or not o.get("evidence", "").strip()
                or type(o.get("uncertain")) is not bool):
            raise ValueError("Invalid visual observation")
        for key in ("count", "total", "matching"):
            if o.get(key) is not None and (type(o[key]) is not int or o[key] < 0):
                raise ValueError("Invalid observed count")
        if o["total"] is not None and o["matching"] is not None and o["matching"] > o["total"]:
            raise ValueError("Matching count exceeds population")
    return observations


def states_count(requirement: str, plan: dict) -> bool:
    """Whether the requirement itself gives the number the plan checks ("exactly three", "at least 2", "a single")."""
    n = plan["expected"]
    words = [str(n), *([NUMBER_WORDS[n]] if n in NUMBER_WORDS else []), *COUNT_SYNONYMS.get(n, ())]
    return any(re.search(r"\b" + re.escape(w) + r"\b", requirement, re.I) for w in words)


def guard(plan: dict, observation: dict, requirement: str = "", rules: str = "all") -> bool:
    """Whether the observation is consistent with the requirement's numeric/universal constraint. rules="all"
    checks every count / absence / universal plan and fails uncertain observations; rules="stated" checks a count
    only if the requirement gives that number (or it is a presence check, at least one), "every X" only if the
    requirement says all / every / each / both, and only definite observations, leaving the rest to decide."""
    stated = rules == "stated"
    if observation["uncertain"] and not stated:
        return False
    if plan["kind"] in ("count", "absence"):
        n, target = observation["count"], plan["expected"]
        if stated and (n is None or plan["kind"] == "count" and not states_count(requirement, plan)
                       and not (plan["comparator"] == "ge" and target <= 1)):
            return True
        if n is None:
            return False
        return {"eq": n == target, "ge": n >= target, "le": n <= target}[plan["comparator"]]
    if plan["kind"] == "universal":
        total, matching = observation["total"], observation["matching"]
        if stated and (not QUANTIFIER.search(requirement) or total is None or matching is None):
            return True
        return total is not None and total > 0 and matching == total
    return True


def requirement_guards(plans: list[dict], observations: list[dict], requirements, rules: str = "all") -> list[bool]:
    """Per requirement, whether the guards of all of its plans pass."""
    passed = [True] * len(requirements)
    for p, o in zip(plans, observations, strict=True):
        passed[p["id"] - 1] &= guard(p, o, requirements[p["id"] - 1], rules)
    return passed


def apply_guards(result: dict, requirements, plans: list[dict], observations: list[dict],
                 rules: str = "all") -> dict:
    """The guards on top of a checked verdict (`verdicts`); the verdicts before them are kept."""
    result = copy.deepcopy(result)
    result["text_verdict_before_guards"] = copy.deepcopy(result["items"])
    passed = requirement_guards(plans, observations, requirements, rules)
    for item, ok in zip(result["items"], passed, strict=True):
        item["satisfied"] = item["satisfied"] and ok
    result.update(score=sum(i["satisfied"] for i in result["items"]) / len(requirements),
                  all_satisfied=all(i["satisfied"] for i in result["items"]), mechanical_guards=passed)
    return result


def before_guards(result: dict) -> dict:
    """A neutral result as the decide step gave it, before the mechanical guards (the guard ablation)."""
    items = result["text_verdict_before_guards"]
    return {"items": items, "score": sum(i["satisfied"] for i in items) / len(items),
            "all_satisfied": all(i["satisfied"] for i in items)}


class NeutralJudge:
    def __init__(self, max_tokens: int = 2048, tokens_per_entry: int | None = None, guards: str = "stated",
                 leaky_observe: bool = False, compile_prompt: str = "neutral_compile.txt", name: str = "neutral",
                 key_name: str | None = None):
        """`max_tokens` caps each request's output. With `tokens_per_entry`, a request that returns n plans,
        observations or verdicts may write up to max(max_tokens, n * tokens_per_entry): with a fixed cap, the
        observe or compile answer for a checklist of 24-50 statements runs out of room and the image gets no
        verdict. `guards` is one of GUARD_RULES ("none" and `leaky_observe=True` exist for ablations: the latter
        shows the observe step the requirements themselves instead of neutral questions)."""
        if guards not in GUARD_RULES:
            raise ValueError(f"guards must be one of {GUARD_RULES}, not {guards!r}")
        self.systems = {"compile": prompts.load(compile_prompt), "observe": prompts.load("neutral_observe.txt"),
                        "decide": prompts.load("neutral_decide.txt")}
        self.max_tokens = max_tokens
        self.tokens_per_entry = tokens_per_entry
        self.guards = guards
        self.leaky_observe = leaky_observe
        self.name = name
        # results are keyed by `key_name` (default: the name), so rows cached by one judge version never count as
        # finished work for another (judges/__init__.py keeps the earlier versions' keys)
        self.key_name = key_name or name
        self._plans: dict[str, asyncio.Task] = {}

    def output_limit(self, entries: int) -> int:
        return max(self.max_tokens, entries * self.tokens_per_entry) if self.tokens_per_entry else self.max_tokens

    async def compile(self, requirements, client: Client) -> list[dict]:
        reply = await client.chat_json(
            "compile", self.systems["compile"], [{"type": "text", "text": "REQUIREMENTS:\n" +
                                                  prompts.numbered(requirements)}],
            prompts.PLANS, max_tokens=self.output_limit(len(requirements)), field_name="plans",
            count=len(requirements),
            validate=lambda a: repair_plans(check_plans(a, requirements), requirements))
        return reply.answer

    def plan(self, item: Item, client: Client) -> asyncio.Task:
        """The compiled plan for this item's checklist, requested once and shared by every item that needs it."""
        key = item.checklist_key
        if key not in self._plans:
            self._plans[key] = asyncio.ensure_future(self.compile(item.requirements, client))
        return self._plans[key]

    async def observe(self, item: Item, plans: list[dict], client: Client) -> list[dict]:
        if self.leaky_observe:
            asked = [{"id": i, "question": r} for i, r in enumerate(item.requirements, 1)]
        else:
            asked = questions(plans)
        content = [{"type": "text", "text": "OBSERVATION QUESTIONS:\n" + json.dumps(asked)},
                   {"type": "image_url", "image_url": {"url": item.image_url()}}]
        reply = await client.chat_json("observe", self.systems["observe"], content, prompts.OBSERVATIONS,
                                       max_tokens=self.output_limit(len(plans)), field_name="observations",
                                       count=len(plans),
                                       validate=lambda a: check_observations(a, plans))
        return reply.answer

    async def decide(self, item: Item, plans: list[dict], observations: list[dict], client: Client) -> dict:
        text = ("REQUIREMENTS:\n" + prompts.numbered(item.requirements) + "\nOBSERVATIONS:\n" +
                json.dumps(observations))
        reply = await client.chat_json("decide", self.systems["decide"], [{"type": "text", "text": text}],
                                       prompts.VERDICTS, max_tokens=self.output_limit(len(item.requirements)),
                                       field_name="items",
                                       count=len(item.requirements), validate=lambda a: verdicts(a, item.requirements))
        if self.guards == "none":
            return reply.answer
        return apply_guards(reply.answer, item.requirements, plans, observations, self.guards)

    async def judge(self, item: Item, client: Client) -> dict:
        plans = await self.plan(item, client)
        observations = await self.observe(item, plans, client)
        result = await self.decide(item, plans, observations, client)
        return {**result, "plans": plans, "observations": observations}
