"""One request per image with the prompt, the checklist and the image: the judge behind every mode.

The model answers with one verdict per requirement, a line of evidence first. Three settings make a mode of it:

- the system prompt: "v1" is short and plain; "strict" adds rules for near misses (every part of a requirement
  counts, identities must be visible, actions shown rather than inferred), the errors a single call makes most;
- thinking: Qwen3.8 checkpoints reason or answer directly by a chat-template switch, with a reasoning effort;
- a cut-off: with thinking off, the probability the model gives "true" at each verdict (from the server's
  log-probabilities) decides instead of the written word. A statement is satisfied only if that probability is at
  least the cut-off, which removes passes the model is not quite sure of (docs/design.md).

With thinking on, a thinking budget caps the reasoning: after that many tokens the server ends it (vLLM's
thinking_token_budget, with the closing line set in the profiles' reasoning-config) and the model answers from what it
has. The answer has a cap of its own: ANSWER_TOKENS (a base plus an allowance per statement) after the reasoning, far
above any answer seen. Normal output stays under both (the reasoning cap cut one of 27,742 development chains); they
are meant for the rare request that loops, in its reasoning or in its answer, which ran toward the 32,768-token limit
and held a batch open (docs/serving.md). A capped answer is retried with a note to keep the evidence short
(client.feedback).
"""
from __future__ import annotations

import math
import re

from .. import prompts
from ..client import Client
from ..data import Item
from .base import verdicts

TOP_LOGPROBS = 5
ANSWER_TOKENS = (2048, 200)   # output tokens after the reasoning: a base, and an allowance per statement
VERDICT_TOKEN = re.compile(r"\s*(true|false)\b")


def user_text(prompt: str, requirements) -> str:
    return "ORIGINAL PROMPT:\n" + prompt + "\n\nREQUIREMENTS:\n" + prompts.numbered(requirements) + "\n\nIMAGE:"


def verdict_probs(tokens: list[dict]) -> list[float]:
    """P(true) at each "satisfied" value of a JSON answer, from its tokens' top alternatives: true / (true + false).
    "false" missing from the alternatives counts as probability 0."""
    text, out = "", []
    for t in tokens:
        m = VERDICT_TOKEN.match(t["token"])
        if m and re.search(r'"satisfied"\s*:\s*$', text):
            mass = {"true": 0.0, "false": 0.0}
            for alt in t.get("top_logprobs") or [{"token": t["token"], "logprob": t["logprob"]}]:
                a = VERDICT_TOKEN.match(alt["token"])
                if a:
                    mass[a.group(1)] += math.exp(alt["logprob"])
            total = mass["true"] + mass["false"]
            out.append(mass["true"] / total if total > 0 else float(m.group(1) == "true"))
        text += t["token"]
    return out


def apply_cutoff(answer: dict, cutoff: float) -> dict:
    """Each statement satisfied iff its p_true reaches the cut-off; the model's own word is kept as `written`. A
    statement without p_true (its verdict token not found in the log-probabilities) keeps the written verdict."""
    items = [{**it, "written": it["satisfied"], "satisfied": it.get("p_true", float(it["satisfied"])) >= cutoff}
             for it in answer["items"]]
    satisfied = sum(it["satisfied"] for it in items)
    return {**answer, "items": items, "score": satisfied / len(items), "all_satisfied": satisfied == len(items)}


class SingleCallJudge:
    """`prompt` picks the system prompt (prompts/single_<prompt>.txt). `thinking` (None: send no switch, for models
    without one) and `effort` set the chat template's enable_thinking and reasoning_effort; `temperature` and
    `sampling` (top_p, top_k, penalties) are sent with every request; a thinking model needs room for its reasoning
    in `max_tokens`, and `thinking_budget` (None: no cap) ends its reasoning after that many tokens. `cutoff` decides
    each verdict by its p_true (see the module docstring); `p_true` records it without deciding by it."""

    schema = prompts.VERDICTS

    def __init__(self, prompt: str = "v1", *, max_tokens: int = 8192, temperature: float | None = None,
                 sampling: dict | None = None, thinking: bool | None = None, effort: str | None = None,
                 thinking_budget: int | None = None, cutoff: float | None = None, p_true: bool = False,
                 name: str | None = None):
        self.name = name or f"single_{prompt}"
        self.system = prompts.load(f"single_{prompt}.txt")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cutoff = cutoff
        self.p_true = p_true or cutoff is not None
        self.thinking, self.thinking_budget = thinking, thinking_budget
        self.extra_body = dict(sampling or {})
        if thinking is not None:
            switch = {"enable_thinking": thinking, **({"reasoning_effort": effort} if effort else {})}
            self.extra_body["chat_template_kwargs"] = switch
        if thinking_budget is not None:
            self.extra_body["thinking_token_budget"] = thinking_budget

    def validate(self, answer: dict, requirements) -> dict:
        return verdicts(answer, requirements)

    def request_max_tokens(self, statements: int) -> int:
        """The output limit for one image: `max_tokens`, cut to the reasoning budget plus the answer's allowance when
        the reasoning has a known bound (thinking off, or a thinking budget). Without one, `max_tokens` stays."""
        if self.thinking is None or (self.thinking and self.thinking_budget is None):
            return self.max_tokens
        reasoning = self.thinking_budget if self.thinking else 0
        return min(self.max_tokens, reasoning + ANSWER_TOKENS[0] + ANSWER_TOKENS[1] * statements)

    async def judge(self, item: Item, client: Client) -> dict:
        content = [{"type": "text", "text": user_text(item.prompt, item.requirements)},
                   {"type": "image_url", "image_url": {"url": item.image_url()}}]
        reply = await client.chat_json(self.name, self.system, content, self.schema,
                                       max_tokens=self.request_max_tokens(len(item.requirements)),
                                       field_name="items", count=len(item.requirements),
                                       validate=lambda a: self.validate(a, item.requirements),
                                       temperature=self.temperature, extra_body=self.extra_body or None,
                                       top_logprobs=TOP_LOGPROBS if self.p_true else None)
        answer = reply.answer
        if self.p_true:
            probs = verdict_probs(reply.logprobs or [])
            if len(probs) == len(answer["items"]):
                for it, p in zip(answer["items"], probs, strict=True):
                    it["p_true"] = p
            if self.cutoff is not None:
                answer = apply_cutoff(answer, self.cutoff)
        return {**answer, "requests": reply.attempts}
