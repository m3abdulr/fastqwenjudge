# Results: agreement with frontier judges

## References and metrics

- **GPT-6 Sol** (OpenAI, reasoning effort xhigh, batch API) on every evaluation image, with the plain single-call
  rubric: the prompt, the numbered requirements and the image, "is each requirement satisfied?".
- **Gemini 3.1 Pro** (batch API, same rubric) on part of each evaluation set.
- **A blind audit by Claude Opus 5.5**: random images, graded by looking at them, before any local judge had
  produced an output on them, under strict literal rules (docs/audit.md). Requirements the image cannot decide are
  marked unsure; statement agreement leaves them out, full-image agreement counts them as failed.

Statement agreement is computed per image and averaged over images. Full-image agreement asks whether the judge
and the reference agree that every requirement holds. Leniency is the judge's full-image pass rate minus the
reference's. Intervals are 95% bootstrap intervals over images; differences between two judges are paired on
the same images.

## Evaluation sets and how they were used

The judge was changed in rounds. Each round's rules (what to try, in what order, what counts as better) were
written before its held-out set existed; changes were decided on development sets only, and each held-out set
was used once, after the configuration was frozen.

| Set | Images | Contents | Used as |
|---|---:|---|---|
| Final evaluation | 3,936 | FLUX-pool prompts (FLUX.2-klein and SD3.5 Medium renders) 700, DiffusionDB 250, five benchmarks 300, R2I-Bench 582, GenEval and GenAI-Bench 2,104 | the first test; development set of rounds 2 and 3 |
| Round-2 held-out | 1,172 | FLUX-pool prompts 400, DiffusionDB 250, benchmarks 300, reasoning 222 | round 2's test; second development set of round 3 (numbers only) |
| Round-3 held-out | 1,000 | FLUX-pool prompts 200, OpenGPT relation prompts 150 (a prompt style the judge had never seen), DiffusionDB 200, benchmarks 250, reasoning 200 | round 3's test |

Benchmarks are ConceptMix, DPG-Bench, GenEval 2, T2I-CompBench and T2I-CoReBench; "reasoning" is R2I-Bench and
T2I-CoReBench's reasoning dimensions, reported separately throughout.

## The two modes on the round-3 held-out set

Both modes are as they ship: `fast` is the neutral judge after round 2, `quality` the Thinking model's single call
on Qwen's FP8 checkpoint, with n-gram drafts in the tail and the thinking budget. "First version" is the neutral
judge as it was before round 2. Round 3's pre-registered test of the quality mode ran before the serving work (BF16
weights quantized at load, no budget); its rows in the paired table are marked "round-3 test". The `fast` numbers
come from the pre-registered held-out run at 128 requests in flight, which left 6 main-task images (checklists of
24-50 statements whose answers ran past a then fixed output limit) and 1 reasoning image without a verdict; its rows
are over the images it judged. As it ships (256 in flight, the output limit grown with the checklist;
[serving.md](serving.md#long-checklists)) it judges 799 main-task and 199 reasoning images: 87.4% statement
agreement with Sol on the main task (80.6% full image), 87.7% with Gemini, 85.2% with the audit and 74.4% with Sol
on reasoning prompts; on the 794 main-task images both judged, -0.2 points [-0.7, +0.3] against Sol. The single call
left 8 long main-task checklists without a verdict (its 1,024-token limit) and the first version 1; the quality mode
judged every image.

**Statement agreement with GPT-6 Sol (%)**

| Set | Images | First version | `fast` | Single call (Instruct) | `quality` | Gemini 3.1 Pro |
|---|---:|---:|---:|---:|---:|---:|
| FLUX-pool prompts | 200 | 85.1 | 90.8 | 90.7 | 93.0 | 94.6 |
| OpenGPT relations | 150 | 79.5 | 86.6 | 86.5 | 87.8 | 91.7 |
| DiffusionDB | 200 | 83.4 | 84.4 | 84.1 | 87.4 | - |
| Benchmarks | 250 | 84.9 | 88.5 | 86.0 | 90.2 | - |
| **Main task** | 800 | 83.6 | 87.7 | 86.8 | 89.8 | 93.4 (350) |
| Reasoning | 200 | 75.4 | 74.5 | 74.7 | 81.4 | - |

**Full-image agreement with GPT-6 Sol (%)**

| Set | First version | `fast` | Single call | `quality` | Gemini 3.1 Pro |
|---|---:|---:|---:|---:|---:|
| FLUX-pool prompts | 69.0 | 82.5 | 72.5 | 82.5 | 87.5 |
| OpenGPT relations | 76.7 | 82.7 | 74.0 | 82.7 | 87.3 |
| DiffusionDB | 72.0 | 73.0 | 61.0 | 74.5 | - |
| Benchmarks | 81.9 | 85.2 | 78.9 | 88.0 | - |
| **Main task** | 75.2 | 81.0 | 71.8 | 82.2 | 87.4 (350) |
| Reasoning | 93.5 | 89.4 | 80.5 | 92.0 | - |

**Statement agreement with the Claude Opus 5.5 audit (%; sure statements)**

| Set | Audited images | First version | `fast` | Single call | `quality` | GPT-6 Sol | Gemini 3.1 Pro |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX-pool prompts | 40 | 84.9 | 90.1 | 87.8 | 90.9 | 93.3 | 95.0 |
| OpenGPT relations | 30 | 76.7 | 82.6 | 81.9 | 81.7 | 89.0 | 88.5 |
| DiffusionDB | 30 | 78.9 | 81.1 | 74.7 | 76.5 | 84.6 | - |
| Benchmarks | 30 | 76.2 | 83.4 | 79.6 | 84.1 | 90.1 | - |
| **Main task** | 130 | 79.6 | 84.8 | 81.5 | 83.8 | 89.6 | 92.2 (70) |
| Reasoning | 30 | 69.6 | 75.6 | 66.0 | 71.6 | 88.3 | - |

**Leniency against Sol, main task (points):** first version -2.3, `fast` +3.4, single call +25.1, `quality`
+13.5, Gemini -1.7.

**Paired differences, main task, against `fast` (95% intervals)**

| | vs Sol: statements | vs Sol: full image | vs audit: statements | vs audit: full image |
|---|---:|---:|---:|---:|
| First version | -4.1 [-5.2, -2.9] | -5.9 [-8.8, -3.1] | -5.3 [-8.4, -2.5] | -0.8 [-7.8, +5.4] |
| Single call (Instruct) | -0.9 [-2.1, +0.3] | -9.1 [-12.5, -5.8] | -3.2 [-5.8, -0.7] | -16.3 [-24.8, -7.8] |
| `quality` | +2.0 [+0.7, +3.2] | +1.1 [-2.1, +4.3] | -1.0 [-3.9, +1.7] | -3.1 [-10.9, +4.7] |
| `quality`, reasoning prompts | +7.0 [+3.8, +10.3] | +2.5 [-2.0, +7.0] | -4.0 [-10.6, +1.8] | -16.7 [-30.0, -3.3] |
| `quality`, round-3 test | +2.1 [+0.8, +3.2] | +0.9 [-2.4, +4.0] | +0.5 [-2.4, +3.5] | -3.1 [-11.6, +5.4] |
| `quality`, round-3 test, reasoning prompts | +6.4 [+3.3, +9.6] | +2.0 [-1.5, +6.0] | -0.1 [-7.7, +7.7] | -10.0 [-20.1, +0.0] |

The quality mode was a pre-registered hypothesis: offer it if, on this held-out set, its statement agreement with
Sol beats the fast mode's (interval above zero) without losing ground against the audit. In the round-3 test both
held. As it ships, the gain against Sol is the same, but against the audit the quality mode is 1.0 point below the
fast mode on the main task (interval -3.9 to +1.7), and on the 30 audited reasoning images the audit prefers the
fast mode, by 16.7 points on full images. The audit covers 130 and 30 of these images, Sol all 800 and 200; across
all 1,047 labelled reasoning images the audit sides with the quality mode (next section).

**How close to the frontier judges (main task).** Gemini's labels cover 350 of the 800 images (the FLUX-pool
prompts and OpenGPT relations). On those, Gemini agrees with Sol on 93.4% of statements, the fast mode on 89.0%
and the quality mode on 90.8%. About three statements in four pass, which flatters raw agreement; corrected for
chance (Cohen's kappa over statements, same 350 images) the gap is wider: 0.63 and 0.69 against Gemini's 0.81. On
the statements where Sol and Gemini give the same verdict, the modes agree with them on 91.0% and 93.4%. Against
the blind audit the gap is larger: 84.8% and 83.8% of sure statements against Sol's 89.6% and Gemini's 92.2% (70
images), kappa 0.57 and 0.58 against 0.79 and 0.82. Most of the local judges' disagreements are passes the
reference fails: 8.5% and 8.7% of statements against Sol, where Gemini has 2.7%.

## Prompts that need reasoning: every labelled image

R2I-Bench and T2I-CoReBench's reasoning dimensions ask for images that are only right if the model worked something
out first (what a scene looks like after an event, a quantity that has to be computed, a rule that has to be
applied). The held-out table above has 200 of them. Here are all 1,047 reasoning images with Sol labels: 625 from
the final evaluation, 222 from round 2's held-out set and the 200 from round 3's, judged by both modes as they ship
(the quality mode with its thinking budget). The fast mode's rules were tuned on the first 847 (traces read on the
final evaluation's, checked on round 2's), so those rows favour it; the 200 held-out images are the clean test.

**Statement agreement (%)**

| Reference | Slice | Images | `fast` | `quality` | Paired difference | Gemini 3.1 Pro |
|---|---|---:|---:|---:|---:|---:|
| GPT-6 Sol | all reasoning images | 1,047 | 72.3 | 81.7 | +9.3 [+8.0, +10.7] | 86.5 (582) |
| GPT-6 Sol | R2I-Bench | 828 | 71.4 | 82.1 | +10.7 [+9.2, +12.3] | 86.5 (582) |
| GPT-6 Sol | T2I-CoReBench reasoning | 219 | 76.0 | 80.2 | +4.2 [+1.4, +7.1] | - |
| GPT-6 Sol | round-3 held-out only | 200 | 74.5 | 81.4 | +7.0 [+3.8, +10.3] | - |
| Gemini 3.1 Pro | R2I-Bench | 582 | 68.3 | 79.9 | +11.6 [+9.6, +13.5] | (Sol: 86.5) |
| Audit, sure statements | all reasoning images | 153 | 73.6 | 78.8 | +5.2 [+1.7, +8.8] | (Sol: 87.3) |
| Audit, sure statements | T2I-CoReBench reasoning | 39 | 75.2 | 71.9 | -3.3 [-9.9, +3.6] | (Sol: 92.2) |
| Audit, sure statements | round-3 held-out only | 30 | 75.6 | 71.6 | -4.0 [-10.6, +1.8] | (Sol: 88.3) |

**Full images.** Almost every reasoning image fails at least one requirement (Sol passes 3% in full, the fast mode
12%, the quality mode 10%), so full-image agreement is mostly agreement on "fails" and says less than the statement
numbers: against Sol 88.5% for `fast` and 92.4% for `quality` (+3.8 [+1.9, +5.8]); against the audit 90.2% and
88.2% (-2.0 [-7.8, +3.3]). Leniency against Sol is +8.4 points for `fast` and +6.5 for `quality`: on these prompts
the quality mode is the stricter of the two, the reverse of the main task (+3.4 and +13.5). On the 200 held-out
images alone it is not: +6.0 against +4.5.

**Speed on the 847 development images** (one H100, 128 requests in flight): `fast` 361 s (2.34 images/s, $0.36 per
1,000; three requests and 1,343 output tokens per image), `quality` 530 s (1.60 images/s, $0.52 per 1,000; one
request and 1,719 output tokens per image). Reasoning prompts cost the quality mode more than main-task prompts do
because its chains are about twice as long (docs/serving.md).

Reading:

- Against Sol and Gemini the quality mode's lead on reasoning prompts is large: 7-12 points of statement
  agreement, +7.0 on the 200 held-out images. The fast mode was tuned on most of these images and still
  trails.
- Nearly all of it comes from R2I-Bench. On T2I-CoReBench's reasoning dimensions the lead against Sol is 4 points,
  and the audit does not see one.
- Against the audit the lead is smaller (+5.2 over 153 images), and on the 30 audited held-out images the audit
  sides with the fast mode: -4.0 points of statement agreement (-10.6 to +1.8), and the quality mode passes six of
  them in full that the audit fails, the fast mode one. Thirty images are few, but it is the one place where the
  numbers point the other way.
- Neither mode is at frontier level here: Gemini agrees with Sol on 86.5% of reasoning statements, against 81.7%
  for the quality mode.

## Round 2: what made the fast judge better

The first version's code guards vetoed a pass whenever a definite observation contradicted a count or "every X"
that the compile step had inferred, and failed every observation flagged uncertain. On the final evaluation's new
prompt styles most of those vetoes were wrong. Round 2 kept two changes: guards that check only what a requirement
itself states (a number, "all/every/each/both", an absence, a presence), and compile rules to plan counts only from
stated numbers. On the round-2 held-out set (950 main-task images):

| | First version | Round 2 | Paired difference |
|---|---:|---:|---:|
| Statements vs Sol | 85.4 | 88.1 | +2.8 [+1.8, +3.7] |
| Full image vs Sol | 77.4 | 80.7 | +3.4 [+0.6, +6.0] |
| Statements vs audit (150 random) | 83.7 | 87.5 | +3.7 [+1.1, +6.4] |
| Statements vs Gemini (FLUX prompts, 400) | 87.1 | 91.9 | |

On the round-3 held-out set the same comparison is +4.1 against Sol and +5.3 against the audit (table above).

## What was tried and not kept

- **Round 2:** a probability threshold on the decide step's "true" (the probabilities sit at 0 or 1: only 2.4% of
  statements fall in between), open questions only in the observe step (-0.3 statements), splitting compound
  requirements in the compile step (the model almost never split), and splitting each requirement into separate
  claims (-1.3 statements on the main task, +32% time; it helped reasoning prompts on round 2's held-out set
  (+4.5 full-image agreement) but not on round 3's (+0.7 statements, not significant)).
- **Round 3** (from a hand trace of 100 development-set failures, docs/design.md): observer rules against describing
  what is not there, compile rules against naming the claimed object, an echo of each observation's subject, decide
  rules for alternatives and shades of a colour, and a ground-and-verify step that crops what the observer reports
  and has the crops described blind. Each moved main-task statement agreement with Sol on development data by -0.5
  to +0.4 points while re-deciding hundreds of borderline statements, and on the held-out main task each stayed
  within -0.4 to +0.4. On held-out reasoning prompts the observer rules gained +1.7 [-0.5, +3.9], short of the bar
  pre-registered for them. A majority vote over several of them gained at most 0.8 points at three to five times the
  cost.
- **The neutral pipeline on the Thinking model:** 12-15 statement points behind both modes; its decide step
  fails what it cannot prove.

## Caveats

- Sol was the main reference while tuning and is also a test reference; Gemini and the audit are the independent
  checks.
- The audit is one model grading by eye. In round 3 it was done by six fresh Claude sessions that knew only the
  rules, not the judge's failure modes. Audit samples are small (130 main-task and 30 reasoning images in round 3),
  so differences of a few points against the audit are within noise.
- The quality mode samples; its verdicts move a little between runs.
