from .single import SingleCallJudge

# The four modes' judges (modes.py). The sampling settings are the
# Qwen3.8 model cards' for thinking, and greedy for thinking off, which agreed with Sol about as often as the card's
# sampling on the development images and keeps the cut-off reading the probabilities behind the verdict the model
# writes. Each
# judge keeps the result key it was measured under, so cached rows of one judge never count as finished work for
# another.
GREEDY = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
THINKING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
THINKING_BUDGET = 8192   # reasoning tokens before the server ends the reasoning (judges/single.py)

JUDGES = {
    "fast": lambda: SingleCallJudge("strict", thinking=False, temperature=0.0, sampling=GREEDY, cutoff=0.999,
                                    name="fast"),
    "quality": lambda: SingleCallJudge("strict", thinking=True, effort="medium", temperature=1.0, sampling=THINKING,
                                       max_tokens=32768, thinking_budget=THINKING_BUDGET, name="quality"),
    "flash-fast": lambda: SingleCallJudge("v1", thinking=False, temperature=0.0, sampling=GREEDY, cutoff=0.998,
                                          name="flash-fast"),
    "flash-quality": lambda: SingleCallJudge("strict", thinking=True, effort="medium", temperature=1.0,
                                             sampling=THINKING, max_tokens=32768, thinking_budget=THINKING_BUDGET,
                                             name="flash-quality"),
}


def make(name: str):
    if name not in JUDGES:
        raise KeyError(f"unknown judge {name!r}; choose from {', '.join(JUDGES)}")
    return JUDGES[name]()
