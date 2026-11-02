# Checklists

The judge grades an image against a checklist: the prompt split into short statements, each true or false of the
image alone ("There are exactly three apples.", "The bowl is blue.", "No people are present."). This page is how the
evaluation checklists were made and how to make your own.

## What makes a good checklist

- **Statements, not questions.** "The sofa is red.", not "Is the sofa red?". The judges decide whether each
  statement holds; a question such as "Does the image contain X?" leaves the judge to guess which answer is the
  pass.
- **Atomic.** One requirement per sentence: split conjunctions, but keep an attribute with the object it describes
  ("The red cup is on the left of the plate.").
- **Exact where the prompt is exact.** "exactly three" when the prompt says three; "one or more" only when it is not
  specific. Negations become what must be absent.
- **Nothing the prompt does not ask for.** No image-quality, style or realism demands unless the prompt makes them;
  the judges are told not to penalize anything a requirement does not mention.

## Where the evaluation checklists came from

Every set in [results.md](results.md) has its checklists fixed before any judge ran, the same for every judge and
every reference (GPT-6 Sol, Gemini 3.1 Pro and the audit grade the same statements):

| Prompts | Checklist |
|---|---|
| Benchmarks that ship question lists or properties (GenEval 2, DPG-Bench, T2I-CompBench, T2I-CoReBench, R2I-Bench) | the benchmark's own questions, rewritten as statements by Gemini 3.8 Flash: one statement per question, in order, true exactly when the expected answer holds ("How many backpacks are in the image? (expected answer: one)" becomes "There is exactly one backpack in the image."), conditions, negations and quantifiers kept, nothing added |
| ConceptMix | the benchmark sampler's own property statements |
| Prompts without one (FLUX-pool prompts, DiffusionDB, OpenGPT relation prompts) | written from the prompt by Gemini 3.8 Flash with a checklist prompt: an earlier version for the first sets (1-8 requirements, only what the prompt states), the version in `prompts/checklist.txt` for the later ones |

The shipped prompt (`src/fastqwenjudge/prompts/checklist.txt`) differs from the earlier one in three ways: up to 20
requirements instead of 8; visible text as its own kind of requirement; and, for prompts whose content must be
inferred (a cause, a consequence, a referenced entity, the result of a rule), the visible outcome the prompt clearly
implies, from a reference description of the intended image where one exists, and nothing when several outcomes are
plausible.

## Making your own

```bash
fastqwenjudge checklist --data prompts.jsonl --out items.jsonl
```

Input rows need `id` and `prompt`; every other field (`image`, `group`, ...) is copied to the output, and rows that
already have `requirements` are passed through, so a file with some checklists can be completed in place. The
command starts the `fast` modes' Qwen3.8-27B server and runs the shipped checklist prompt with thinking off and
greedy decoding; an answer with a question in it, or with no requirement, is retried. To use another model, point it
at any OpenAI-compatible endpoint (the key, if it needs one, is read from `OPENAI_API_KEY`):

```bash
fastqwenjudge checklist --base-url https://api.example.com/v1 --served-name <model> --data prompts.jsonl --out items.jsonl
```

Checklists written by the local 27B have not been compared with Gemini's. For a benchmark that ships its own
questions or rules, convert those rather than writing new ones, and keep the benchmark's meaning: its counts, its
conditions, and what its own scorer counts as a pass. A checklist decides what "correct" means for every judge that
uses it, so it is worth reading a sample before judging a large set.
