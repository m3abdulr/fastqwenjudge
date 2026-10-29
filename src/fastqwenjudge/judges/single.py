"""One request per image with the prompt, the checklist and the image (the usual VLM-as-judge setup). On a thinking
model (judge "think") the model reasons before it answers, and the answer is the same JSON verdict list.

A thinking budget caps the reasoning: after that many tokens the server ends it (vLLM's thinking_token_budget, with
the closing line set in the quality profiles' reasoning-config) and the model answers from what it has. About 0.6%
of reasoning-heavy prompts send the model into a loop that would otherwise run to the output limit, fail, and be
retried; on everything else the budget is never reached."""
from __future__ import annotations

from .. import prompts
from ..client import Client
from ..data import Item
from .base import verdicts


def user_text(prompt: str, requirements) -> str:
    return "ORIGINAL PROMPT:\n" + prompt + "\n\nREQUIREMENTS:\n" + prompts.numbered(requirements) + "\n\nIMAGE:"


class SingleCallJudge:
    """`version` picks the system prompt: v1 is short and plain, v2 adds explicit rules for counts, negation,
    "every", attribute binding and spatial relations. `temperature` is the judge's own sampling temperature (None:
    the client's); a thinking model needs room for its reasoning in `max_tokens`, and `thinking_budget` (None: no
    budget) ends the reasoning after that many tokens."""

    def __init__(self, version: str = "v1", max_tokens: int = 1024, temperature: float | None = None,
                 thinking_budget: int | None = None, name: str | None = None):
        self.name = name or f"single_{version}"
        self.system = prompts.load(f"single_{version}.txt")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    async def judge(self, item: Item, client: Client) -> dict:
        content = [{"type": "text", "text": user_text(item.prompt, item.requirements)},
                   {"type": "image_url", "image_url": {"url": item.image_url()}}]
        reply = await client.chat_json(self.name, self.system, content, prompts.VERDICTS,
                                       max_tokens=self.max_tokens, field_name="items",
                                       count=len(item.requirements),
                                       validate=lambda a: verdicts(a, item.requirements),
                                       temperature=self.temperature, extra_body=None if self.thinking_budget is None
                                       else {"thinking_token_budget": self.thinking_budget})
        return {**reply.answer, "requests": reply.attempts}
