# Manual audit protocol

A reference besides the API judge labels: Claude Opus 5.5 grades a sample of images by looking at them directly,
inside an interactive session, without any judge's verdicts. This is a manual audit, not an
automated API judge, and it is a single auditor; its value is that it is independent of every judge compared
here and of the API labels.

## Samples

The first audit (`experiments/audit/sample.py`) built its queue before any auditing:

- **random**: 75 images from the API-labelled set (GenEval, GenAI-Bench) and 75 from the out-of-domain set
  (DPG-Bench, T2I-CoReBench, GenEval2, T2I-CompBench, ConceptMix), fixed seed. Agreement numbers are reported
  on this sample.
- **disagreement**: 50 images per set where the neutral and single-call judges disagree on at least one
  requirement. Used only to ask which judge is right when they differ.

The two samples are merged and shuffled; the queue contains id, prompt, requirements and the image path, and
nothing about which sample an image came from or what any judge said. The sample membership file is read
only after the audit is finished.

## Rules

The same rubric as the single-call system prompt (`single_v1.txt`): strict and literal; a requirement is
satisfied only if it is clearly true in the image; exact counts must match; missing, unrecognizable or
ambiguous evidence means not satisfied; nothing is penalized that the requirement does not mention; left and
right from the viewer's side. Each verdict gets a one-line evidence note. Requirements that cannot be decided
from the image are marked `?`, stored as not satisfied and listed under `unsure`; results are reported with
and without them.

Verdicts are recorded once and not revised after judge outputs are seen.

## Later audits

Each evaluation round added a fresh random sample, drawn with a fixed seed and graded under the same rules before
any local judge had produced an output on those images:

- **Final evaluation:** 200 random images from the new sets, plus 69 images where two judges disagreed, chosen by a
  rule fixed before the local judges ran (used only to ask which judge was right).
- **Round-2 held-out set:** 150 random main-task images and 40 reasoning images.
- **Round-3 held-out set:** 160 random images (130 main task, 30 reasoning), graded by six fresh Claude sessions in
  parallel, each given only these rules and its slice of the queue. Unlike the earlier audits, the auditors knew
  nothing about the judge's known failure modes.

## For the Qwen3.8 release

No new audit was run. The held-out tables in [results.md](results.md) use the round-3 audit, which was graded before
any Qwen3.8 judge existed and so is blind to them as well; development numbers use the 263 audited development images
(111 main-task) of the earlier rounds. `experiments/audit/sample.py` builds a queue the same way for a new set.
