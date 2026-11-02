# Design notes

## One request per image

Every mode makes one request per image: a system prompt, then the original prompt, the numbered requirements and the
image. The answer is JSON with one item per requirement, in order, each a sentence of visible evidence followed by
the verdict:

```json
{"items": [{"id": 1, "evidence": "Two large silhouetted palm trees dominate the foreground.", "satisfied": true}, ...]}
```

The server's structured output holds the model to that schema, including the number of items; an answer that still
fails the check (wrong count, an item missing) is retried with the error quoted, up to three attempts. Evidence comes
before the verdict so the model commits to what it sees before it decides.

The four modes are two settings of this one judge (`judges/single.py`): the system prompt, and thinking on at medium
effort or off with a cut-off on the verdict's probability.

## Leniency is a near-miss problem

On development images the 27B with its thinking off passed 45% of the statements Sol fails, and failed 1.5% of the
ones Sol passes. The errors are one-sided, and they are near misses: a dog that is not quite the named breed, a man
whose pose suggests running, a two-part requirement with one part true, a landmark that looks generically like a
landmark. A judge asked "is this satisfied?" by an image that is close to it tends to say yes.

Without thinking, nothing said to the model moved this much ([results.md](results.md#how-the-modes-were-chosen)): the
model card's sampling or greedy decoding, a prompt that describes the image first, one that argues against each
statement before deciding. The 27B stayed 14-18 points more lenient than Sol, and the Qwen3-VL-32B judge's rules for
counts and negation made it worse (+26).

## The strict prompt

`prompts/single_strict.txt` names the near misses: every part of a requirement counts, identities need the features
that identify them, actions and effects must be shown rather than suggested, relations must hold for every object
they involve, counts must be exact, and anything the requirement does not mention (style, quality, extra objects)
does not count against it. Close is not enough. With thinking off it changes little on its own; with medium
reasoning it took the 27B's leniency from +14.3 to +9.7 points at no cost in agreement, and it is the prompt of three
of the four modes. `flash-fast` keeps the plain prompt (`single_v1.txt`).

## The verdict's probability

With thinking off the verdict is a single token, `true` or `false`, and the server can return the probabilities of
the alternatives at every position (`top_logprobs`, 5). The judge finds the token that follows `"satisfied":` for each
item and reads P(true) = p(true) / (p(true) + p(false)) there (`verdict_probs`). A statement counts as satisfied
only if P(true) reaches the mode's cut-off: 0.999 for `fast`, 0.998 for `flash-fast`. The model's own word is kept
in the output as `written`; where the two differ, the model said `true` without being quite sure. A statement whose
verdict token is not among the alternatives keeps the written verdict.

The cut-offs look extreme because the model is sure of almost everything: fewer than 2% of the 27B's verdicts fall
between 0.02 and 0.98, and the cut-offs that change anything are above 0.99. On development images the 27B's leniency
went from +16 points at 0.5 to +5.5 at 0.999, for 0.8 points of statement agreement; past 0.9995 agreement falls fast.
With the plain prompt, leniency at 0.999 moved between +8.8 and +11.5 points across eleven runs of the same judge on
the 600 main-task images (different serving settings, verdicts otherwise within rerun noise), so the cut-off is a
coarse setting: it was set on development images and checked on a separate confirmation set.

The fast modes decode greedily. With sampling, the verdict written and the probability read can come from different
draws; with greedy decoding the cut-off always reads the probabilities behind the verdict the model actually wrote.

## Reasoning, and a cap on it

The quality modes turn thinking on at medium effort (the model card's sampling: temperature 1.0, top-p 0.95, top-k
20, a fixed seed per request). Effort trades agreement against time almost in proportion to output tokens: on the
27B's development images medium is within 0.4 points of xhigh on main-task prompts and 2.7 below it on reasoning
prompts, at half the time. Reasoning helps most where the prompt needs it (a scene after an event, a quantity to
compute, a rule to apply): +4 points against Sol on those held-out images for the 27B.

Reasoning is capped at 8,192 tokens: vLLM ends it with a closing line written into the server profile, and the model
answers from what it has. The answer has a bound of its own: the whole output at most the reasoning budget plus 2,048
tokens and 200 per statement, and an answer cut there is asked again with a note to keep each evidence to one short
sentence. Normal output stays under both: the reasoning cap cut one of 27,742 development chains, and of 207,461
answers on record the closest came to a third of its bound. They are for the rare request that loops, in its reasoning
or in its answer, which otherwise runs toward the 32,768-token limit and holds a whole batch open
([serving.md](serving.md#the-batchs-tail)).

## One request, not a pipeline

The Qwen3-VL-32B judge (`qwen3vl-v1`) split the decision in three for its fast mode: plan a neutral question for each
requirement, ask it of the image without the requirement in view, then compare the answer with the requirement in
text. It existed because Qwen3-VL-32B in one request passed 25 points more images in full than Sol did (its
single-call baseline on the held-out set). On the Qwen3.8 models the single call is the better judge: 1.4-2.8 points
more agreement with Sol on the held-out main task, 2.6-3.1x faster, and the pipeline overcorrected, failing 6-8 points
more images than Sol ([results.md](results.md#baselines-on-the-held-out-set)).

## Two models

**Qwen3.8-27B** is a hybrid: 48 Gated DeltaNet layers, which keep a fixed-size recurrent state per request instead of
a growing KV cache, and 16 attention layers. One checkpoint does both jobs: a chat-template switch turns thinking on
or off. Qwen's FP8 checkpoint (31 GB) runs on one H100 with room for about 90 requests; it runs 64, the knee of the
throughput curve ([serving.md](serving.md#how-many-requests-at-once)).

**Qwen3.8-Flash-Next** is a mixture of experts (125B parameters, 6B active per token: 10 of 512 experts) with
DeltaNet layers, sparse attention, and a 51-billion-parameter n-gram embedding table that vLLM keeps in host memory.
Its FP8 checkpoint needed four H100s; NVIDIA's NVFP4 checkpoint (4-bit experts) fits on two, where vLLM runs the
experts with a mixed-precision kernel (H100s have no FP4 tensor cores). On the held-out set the NVFP4 modes agree with
Sol as well as the FP8 baselines did on four GPUs (90.2% against 90.3% with thinking off, 92.3% against 92.3% with it
on), though with different prompts and settings, so the comparison is not exact.

Both carry a multi-token-prediction head that drafts tokens for speculative decoding, and both are served with it
([serving.md](serving.md)).
