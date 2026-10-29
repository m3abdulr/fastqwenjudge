from .neutral import NeutralJudge
from .single import SingleCallJudge
from .vqascore import VQAScore

# The two modes' judges are "neutral" (fast) and "think" (quality); see modes.py. neutral_v1 is the neutral judge
# before its guards checked only what a requirement states (kept to reproduce the earlier results), and the rest are
# ablations and baselines. Each keeps the result key it was run under, so cached rows of one version never count as
# finished work for another. The exception is neutral's output limit per statement: it only changes requests that
# ran out of room under the fixed 2,048 tokens, so neutral kept its key, and the images that failed are retried.
JUDGES = {
    "neutral": lambda: NeutralJudge(tokens_per_entry=128, key_name="neutral_stated"),
    "think": lambda: SingleCallJudge("v1", max_tokens=16384, temperature=0.6, thinking_budget=8192, name="think"),
    "neutral_v1": lambda: NeutralJudge(guards="all", compile_prompt="neutral_compile_v1.txt", name="neutral_v1",
                                       key_name="neutral_presencefix"),
    "neutral_noguards": lambda: NeutralJudge(guards="none", name="neutral_noguards", key_name="neutral_noguards_v2"),
    "neutral_leaky": lambda: NeutralJudge(leaky_observe=True, name="neutral_leaky", key_name="neutral_leaky_v2"),
    "single_v1": lambda: SingleCallJudge("v1"),
    "single_v2": lambda: SingleCallJudge("v2"),
    "vqascore": VQAScore,
}


def make(name: str):
    if name not in JUDGES:
        raise KeyError(f"unknown judge {name!r}; choose from {', '.join(JUDGES)}")
    return JUDGES[name]()
