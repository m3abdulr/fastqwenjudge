# Results: agreement with frontier judges

## References and metrics

- **GPT-6 Sol** (OpenAI, reasoning effort xhigh, batch API) on every evaluation image, with the plain single-call
  rubric: the prompt, the numbered requirements and the image, "is each requirement satisfied?".
- **Gemini 3.1 Pro** (batch API, same rubric) on part of each evaluation set.
- **A blind audit by Claude Opus 5.5**: random images, graded by looking at them before any local judge had produced
  an output on them, under strict literal rules ([audit.md](audit.md)). Requirements the image cannot decide are
  marked unsure; statement agreement leaves them out.

Statement agreement is computed per image and averaged over images. Full-image agreement asks whether the judge and
the reference agree that every requirement holds. Leniency is the judge's full-image pass rate minus the
reference's, in points: positive means the judge passes images the reference fails. Brackets are 95% bootstrap
intervals over images.

## Evaluation sets and how they were used

| Set | Images | Used for |
|---|---:|---|
| Development | 1,447: 600 main-task, 847 that need reasoning | every choice: prompts, cut-offs, reasoning effort, sampling, serving |
| Confirmation | 1,557 main-task | the final pick among the development leaders |
| Held-out (round 3) | 1,000: 800 main-task, 200 that need reasoning | scored twice, each time with the judges and settings written down beforehand: the baselines (2026-10-29) and the four modes (2026-10-30), then once more for speed only (2026-10-31) |

Main-task prompts are FLUX-pool prompts (FLUX.2-klein and SD3.5 Medium renders), DiffusionDB, OpenGPT relation
prompts and five benchmarks (ConceptMix, DPG-Bench, GenEval 2, T2I-CompBench, T2I-CoReBench); "reasoning" is
R2I-Bench and T2I-CoReBench's reasoning dimensions, reported separately throughout. Sol labels every image; Gemini
labels 350 held-out main-task images and 894 development images; the audit covers about 130 held-out main-task
images and 263 development images. The checklists were written from the prompts once, before any judging, and are
the same for every judge ([checklists.md](checklists.md)).

## The four modes on the held-out set

Statement agreement in %, leniency in points. The judges are as they ship, with the serving settings of the held-out
run (64 requests at once for Flash-Next, input order, no thinking budget). The later serving changes moved verdicts on
development images about as much as rerunning the same settings does (2.3-2.8% of greedy statement verdicts and
4.2-5.4% of sampled ones, against 2.1-2.6% and 3.7-5.6% between identical runs; [serving.md](serving.md)). Those
greedy figures are the verdicts the model writes; with the fast modes' cut-offs applied, identical runs differ on
2.7-3.3% (Flash-Next) and 4.0-5.0% (27B) of statements, and the kept changes on 2.9-3.8% and 4.6-5.4%. The
Qwen3-VL-32B rows are the judge at the tag
[`qwen3vl-v1`](https://github.com/m3abdulr/fastqwenjudge/tree/qwen3vl-v1), its fast mode rerun on these images;
the tag's documentation quotes the pre-registered run of an earlier version (87.7% against Sol), within half a point.

| Judge | GPUs | Sol, main (800) | full image | lenient | Sol, reasoning (200) | full image | lenient | Gemini (350) | lenient | audit (~130) | lenient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fast`: 27B, strict prompt, thinking off, P(true) >= 0.999 | 1 | 89.5 [88, 91] | 82.9 | +6.9 | 82.1 [80, 85] | 95.5 | +1.5 | 89.7 | +7.4 | 86.9 | +19.2 |
| `quality`: 27B, strict prompt, medium reasoning | 1 | 91.0 [90, 92] | 83.4 | +11.1 | 86.1 [84, 88] | 93.5 | +3.5 | 91.7 | +11.1 | 86.1 | +22.3 |
| `flash-fast`: Flash-Next, plain prompt, thinking off, P(true) >= 0.998 | 2 | 90.2 [89, 91] | 84.8 | +2.0 | 86.2 [84, 89] | 97.0 | -1.0 | 90.9 | +0.9 | 86.2 | +13.1 |
| `flash-quality`: Flash-Next, strict prompt, medium reasoning | 2 | 92.3 [91, 93] | 86.5 | +4.8 | 89.0 [87, 91] | 94.5 | -1.5 | 93.2 | +5.1 | 87.7 | +16.9 |
| Qwen3-VL-32B (`qwen3vl-v1`), fast: neutral pipeline | 1 | 87.4 [86, 89] | 80.6 | +2.6 | 74.4 [71, 78] | 89.4 | +4.5 | 87.7 | +5.1 | 85.2 | +7.7 |
| Qwen3-VL-32B-Thinking (`qwen3vl-v1`), quality: single call | 1 | 89.8 [89, 91] | 82.2 | +13.5 | 81.4 [78, 84] | 92.0 | +6.0 | 90.6 | +15.1 | 83.8 | +21.5 |

On the 350 images Gemini labelled, Gemini and Sol agree on 93.4% of statements; on the same images the four modes
agree with Sol on 90.2% (`fast`), 92.1% (`quality`), 89.8% (`flash-fast`) and 93.0% (`flash-quality`). No mode left an
image without a verdict; the previous fast mode left 2.

| Judge | 1,000 images (s) | images/s | $ per 1,000 images | one image alone (s) | request latency p50 / p95 under load (s) | output tokens per image | generated tokens/s (per GPU) | prompt + generated tokens/s (per GPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fast` | 177 | 5.64 | 0.15 | 1.6 | 40.0 / 64.0 | 299 | 1,687 | 10,942 |
| `quality` | 295 | 3.39 | 0.25 | 4.0 | 31.7 / 66.8 | 843 | 2,855 | 8,399 |
| `flash-fast` | 121 | 8.26 | 0.20 | 1.3 | 26.6 / 55.8 | 316 | 2,611 (1,306) | 14,432 (7,216) |
| `flash-quality` | 249 | 4.02 | 0.41 | 3.5 | 25.0 / 61.4 | 872 | 3,506 (1,753) | 10,048 (5,024) |
| Qwen3-VL-32B (`qwen3vl-v1`), fast | 431 | 2.32 | 0.36 | 10.6 | 26.1 / 68.2 | 1,294 | 2,999 | 9,473 |
| Qwen3-VL-32B-Thinking (`qwen3vl-v1`), quality | 497 | 2.01 | 0.41 | 8.7 | 37.5 / 133.9 | 1,208 | 2,433 | 5,276 |

All 1,000 images submitted at once; dollars at $3 per H100-hour; one image alone is the mean over 30 development
images. Tokens per second are over the whole batch: generated tokens include thinking tokens, prompt tokens include
the image's tokens, and per GPU divides Flash-Next's two GPUs' total by two (per-request decode speed and time to
first token: [serving.md](serving.md#tokens-per-second-and-latency-per-request)). The Qwen3.8 rows are a timing-only
run of the package's settings (2026-10-31; planned in advance, verdicts not re-scored); the Qwen3-VL-32B rows are the
tag's own held-out run. GPT-6 Sol costs $5.51 and Gemini 3.1 Pro $5.84 per 1,000 images at batch list prices.

Reading it:

- Against Sol on the main task, `fast` matches Qwen3-VL-32B-Thinking (89.5% against 89.8%) and the other three
  modes are above it; on prompts that need reasoning every mode is above both Qwen3-VL-32B judges (82-89% against
  74-81%).
- The audit is the strictest reference: every judge passes more images in full than it does. Against it the four
  modes sit within 1.6 points of each other, inside the noise of a 130-image sample.
- `quality` is the most lenient mode: +11 points against Sol. `flash-fast` is within 2 points of Sol's pass rate
  on both halves of the set.

## Baselines on the held-out set

Before any tuning, both models were run on the held-out set with their model cards' sampling settings, in two judging
styles: the neutral pipeline that the Qwen3-VL-32B judge used (three requests; the one that sees the image never sees
the requirements) and a single call. The neutral-pipeline rows ran on the code at the tag `qwen3vl-v1`, with the
Qwen3.8 models in place of Qwen3-VL-32B. Flash-Next ran here in its FP8 checkpoint on four H100s.

| Judge (first held-out run) | Sol, main (800) | full image | Sol, reasoning (200) | Gemini (350) | audit (~130) | lenient vs Sol, main | 1,000 images |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B, neutral pipeline, thinking off | 87.7 [86, 89] | 81.0 | 80.7 [78, 84] | 89.0 | 86.3 | -5.8 | 504 s |
| Qwen3.8-27B, single call, thinking off | 89.1 [88, 90] | 78.6 | 82.8 [80, 86] | 89.2 | 83.9 | +18.1 | 196 s |
| Qwen3.8-27B, single call, thinking on | 90.9 [90, 92] | 83.8 | 88.3 [86, 91] | 91.0 | 87.6 | +10.5 | 806 s |
| Flash-Next (FP8, 4 GPUs), neutral pipeline | 87.5 [86, 89] | 80.1 | 81.9 [79, 85] | 89.7 | 86.6 | -7.9 | 393 s |
| Flash-Next (FP8, 4 GPUs), single call, thinking off | 90.3 [89, 91] | 84.0 | 85.1 [83, 88] | 91.3 | 86.0 | +5.0 | 128 s |
| Flash-Next (FP8, 4 GPUs), single call, thinking on | 92.3 [91, 93] | 87.8 | 88.7 [87, 91] | 92.7 | 90.0 | +2.5 | 342 s |

On these models the neutral pipeline is not worth its three requests: the single call agrees more often with Sol
and Gemini (the audit preferred the pipeline to the single call with thinking off, by 2.4 points on the 27B and 0.6 on
Flash-Next, but not to the thinking ones) and is 2.6-3.1x faster, and the pipeline overcorrects, failing 6-8 points
more images in full than Sol does. What remained was the single call's leniency with thinking off (+18 points for the
27B), which is what the development work went after. Thinking on, at the model's default effort, was accurate but slow
(806 s for the 27B); the shipped quality modes use medium effort, and the Flash-Next modes moved to NVIDIA's NVFP4
checkpoint on two GPUs.

## A judge without a checklist

The usual baseline: no checklist, just the prompt and the image, list the problems, then score 0-10. An image
passes at a score of at least T, set per judge on the development images (the integer that maximises full-image
agreement with Sol; ties go to the smaller leniency). There are no statement verdicts, so only full-image numbers
exist.

| Judge (held-out) | T | full image vs Sol, main | lenient | full image, reasoning | AUROC vs Sol's pass/fail, main | reasoning | 1,000 images |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27B, thinking off | 10 | 78.0 | +1.0 | 89.0 | 0.807 | 0.859 | 111 s |
| 27B, medium reasoning | 10 | 79.8 | -11.8 | 90.5 | 0.866 | 0.926 | 218 s |
| Flash-Next, thinking off | 9 | 75.8 | -10.2 | 94.0 | 0.825 | 0.853 | 89 s |
| Flash-Next, medium reasoning | 10 | 72.6 | -25.1 | 92.5 | 0.875 | 0.923 | 233 s |

Against the checklist modes on the same model and thinking setting, the score loses 3.6 to 13.9 points of
full-image agreement on the main task (79.8 against 83.4 for the 27B with reasoning; 72.6 against 86.5 for
Flash-Next with reasoning), and its pass mark is either lenient or far too strict depending on the judge. The
checklist also says which requirement failed, which a score does not.

## How the modes were chosen

All on the 1,447 development images (main-task numbers below; statement agreement with Sol, leniency against Sol,
batch time for all 1,447 on one H100 for the 27B and two for Flash-Next).

**Thinking off.** The single call's leniency did not move with anything said to the model: the model card's sampling
or greedy decoding, the Qwen3-VL-32B judge's rules for counts and negation (v2), a prompt that describes the image
first (look), one that writes the case against each statement before deciding (doubt), and the strict prompt all left
the 27B at +14 to +18 points, and v2 made it worse (+26). What moved it was the model's own confidence: count a
statement as satisfied only when the probability of `true` at the verdict reaches a cut-off.

| 27B, strict prompt, thinking off: cut-off | Sol, main | full image | lenient |
|---|---:|---:|---:|
| none (0.5) | 91.0 | 79.8 | +16.2 |
| 0.99 | 91.2 | 81.0 | +15.0 |
| 0.998 | 91.0 | 82.2 | +10.5 |
| 0.999 (`fast`) | 90.2 | 82.2 | +5.5 |
| 0.9995 | 89.8 | 82.7 | +2.0 |
| 0.9999 | 84.0 | 79.3 | -12.0 |

Fewer than 2% of the 27B's verdicts have a probability between 0.02 and 0.98, which is why the cut-offs that matter
are in the fourth decimal. Greedy decoding agreed with Sol about as often as the card's sampling (90.9% against
91.1% with the look prompt) but was 2.7 points more lenient before any cut-off; it was kept because the cut-off then
reads the probabilities behind the verdict the model actually writes. Flash-Next's thinking-off judge uses the plain
prompt with a cut-off of 0.998.

**Thinking on.** Effort trades agreement and leniency against time, almost linearly in output tokens:

| 27B, thinking on | Sol, main | full image | lenient | output tokens per image | 1,447 images |
|---|---:|---:|---:|---:|---:|
| low effort, plain prompt | 91.6 | 80.3 | +15.7 | 886 | 609 s |
| medium, plain prompt | 91.8 | 82.7 | +14.3 | 889 | 563 s |
| medium, strict prompt (`quality`) | 92.3 | 84.7 | +9.7 | 932 | 617 s |
| default effort (xhigh), plain prompt | 92.4 | 84.0 | +11.7 | 2,132 | 1,281 s |
| xhigh, strict prompt | 92.7 | 83.7 | +7.3 | 2,158 | 1,265 s |

Medium effort with the strict prompt is within 0.4 points of xhigh at half its time. For Flash-Next the same pair
gave 93.4% and +5.2 points (xhigh: 93.2%, +4.7, 1.6x the time).

**The pick** among the development leaders was made on the 1,557-image confirmation set, and the four modes were
then frozen and scored once on the held-out set. The 27B fast judge missed the development selection rule by 0.1
point and was promoted on the confirmation set, the one exception to the rule, written down before the held-out run.

## What was tried and not kept

- The neutral pipeline, on both models (above).
- Prompts for leniency with thinking off: v2's rules, look, doubt (above). With xhigh reasoning the doubt prompt
  was less lenient than the strict one (+5.3 against +7.3 points) for 1.5x the time.
- The model card's sampling with thinking off, in place of greedy decoding.
- xhigh reasoning effort: 0.4 points over medium on main-task development images and 2.7 on reasoning ones, for
  twice the time.
- Flash-Next's FP8 checkpoint on four GPUs, in place of NVIDIA's NVFP4 checkpoint on two: no better agreement for
  twice the GPUs.
- The no-checklist 0-10 judges (above).

## Caveats

- Sol was the main reference while tuning and is also a test reference; Gemini and the audit are the independent
  checks.
- The audit is one model grading by eye, on small samples (about 130 held-out main-task images), so differences of
  a few points against it are within noise.
- The held-out set was scored twice: once for the baselines and once for the final four modes. The second scoring
  is the one the README reports; both were frozen beforehand. A third run, also planned in advance, measured only
  the speed of the package's settings; its verdicts were not re-scored.
- The quality modes sample, and batching changes the arithmetic, so their verdicts move a little between runs (about
  4-6% of statements). The fast modes are greedy, but their probabilities often sit near the cut-off, so 3-5% of
  their final verdicts move (about 2% of the verdicts the model writes).
