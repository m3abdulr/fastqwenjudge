"""Baseline: VQAScore (Lin et al., 2024) computed with the judge VLM instead of CLIP-FlanT5.

score = P("Yes" | image, 'Does this figure show "{prompt}"? Please answer yes or no.') from the first token.
It gives one number per image and ignores the checklist, so it is compared with image-level full success.
"""
from __future__ import annotations

import math

from ..client import Client
from ..data import Item

YES = ("Yes", "yes", " yes", " Yes")


class VQAScore:
    name = "vqascore"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    async def judge(self, item: Item, client: Client) -> dict:
        question = f'Does this figure show "{item.prompt}"? Please answer yes or no.'
        _, top = await client.top_logprobs("vqascore", [{"type": "image_url", "image_url": {"url": item.image_url()}},
                                                        {"type": "text", "text": question}])
        p_yes = sum(math.exp(lp) for token, lp in top if token in YES)
        return {"score": p_yes, "all_satisfied": p_yes >= self.threshold, "items": None}
