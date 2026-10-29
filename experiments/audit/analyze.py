"""Compare every judge with the manual audit, after the audit is finished.

    python experiments/audit/analyze.py --queue experiments/audit/results/queue.jsonl \
        --audit experiments/audit/results/claude.jsonl \
        --results experiments/quality/results/sol_subset experiments/quality/results/ood \
        --references data/references/sol.jsonl --out experiments/audit/results/report

Three questions:
  1. agreement with the audit on the random sample (the only sample that estimates accuracy);
  2. on requirements where the neutral and single-call judges disagree, which one the audit sides with;
  3. on requirements where a mechanical guard overturned the decide step, whether the guard was right.
Requirements the auditor marked unsure are left out of 2 and 3, and reported both ways in 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fastqwenjudge import metrics
from fastqwenjudge.judges.neutral import before_guards

API_LABELLED = ("gev_", "gab_")   # GenEval and GenAI-Bench, the sets with API reference labels


def fmt(ci):
    return f"{100 * ci[0]:.1f} [{100 * ci[1]:.1f}, {100 * ci[2]:.1f}]"


def agreement_without_unsure(judge: dict, audit: dict, ids: list[str]) -> tuple[float, float, float]:
    per_image = []
    for i in ids:
        unsure = set(audit[i].get("unsure", []))
        same = [a["satisfied"] == b["satisfied"] for n, (a, b) in
                enumerate(zip(judge[i]["items"], audit[i]["items"], strict=True), 1) if n not in unsure]
        if same:
            per_image.append(np.mean(same))
    return metrics.bootstrap(np.array(per_image))


def sides(a: dict, b: dict, audit: dict, ids: list[str]) -> dict:
    """Over requirements where judges a and b disagree (auditor sure), how often the audit sides with each."""
    count = {"a": 0, "b": 0, "n": 0}
    for i in ids:
        if i not in a or i not in b:
            continue
        unsure = set(audit[i].get("unsure", []))
        for n, (x, y, z) in enumerate(zip(a[i]["items"], b[i]["items"], audit[i]["items"], strict=True), 1):
            if x["satisfied"] != y["satisfied"] and n not in unsure:
                count["n"] += 1
                count["a" if x["satisfied"] == z["satisfied"] else "b"] += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--audit", required=True)
    p.add_argument("--results", nargs="+", required=True, help="judge results dirs; files are <judge>.jsonl")
    p.add_argument("--references", nargs="*", default=[])
    p.add_argument("--out")
    args = p.parse_args()
    audit = metrics.load(args.audit)
    origin = json.loads(args.queue.with_suffix(".origin.json").read_text())
    judges: dict[str, dict] = {}
    for d in args.results:
        for path in sorted(Path(d).glob("*.jsonl")):
            judges.setdefault(path.stem, {}).update(metrics.load(path))
    judges["neutral_noguards"] = {k: before_guards(r) for k, r in judges["neutral"].items()}
    for r in args.references:
        judges[Path(r).stem] = metrics.load(r)

    random_ids = [i for i, o in origin.items() if "random" in o and i in audit]
    subsets = {"all": random_ids,
               "GenEval + GenAI-Bench": [i for i in random_ids if i.startswith(API_LABELLED)],
               "out-of-domain": [i for i in random_ids if not i.startswith(API_LABELLED)]}
    lines, report = ["## Agreement with the audit, random sample\n"], {"random": {}}
    for name, ids in subsets.items():
        lines += [f"\n### {name} ({len(ids)} images)\n",
                  "| Judge | Images | Requirement agreement | without unsure | kappa | Full-success agreement "
                  "| Leniency |", "|---|---:|---:|---:|---:|---:|---:|"]
        for judge, res in judges.items():
            common = [i for i in ids if i in res]
            if not common:
                continue
            r = metrics.compare(res, audit, common)
            report["random"].setdefault(name, {})[judge] = r
            fs, req = r["full_success"], r.get("requirements")
            if req:
                req_txt = (f"{fmt(req['agreement'])} | {fmt(agreement_without_unsure(res, audit, common))} | "
                           f"{req['kappa']:.3f}")
            else:
                req_txt = f"AUROC {fs['auroc']:.3f} | - | -"
            lines.append(f"| {judge} | {r['images']} | {req_txt} | {fmt(fs['agreement'])} | "
                         f"{100 * fs['leniency'][0]:+.1f} |")

    all_ids = list(audit)
    disagreement_ids = [i for i, o in origin.items() if "disagreement" in o and i in audit]
    lines += ["\n## Who the audit sides with\n", "| Comparison | Images | Requirements in dispute | First right "
              "| Second right |", "|---|---:|---:|---:|---:|"]
    for label, a, b, ids in [("neutral vs single_v1 (disagreement sample)", "neutral", "single_v1", disagreement_ids),
                             ("neutral vs single_v1 (all audited)", "neutral", "single_v1", all_ids),
                             ("neutral vs single_v2 (all audited)", "neutral", "single_v2", all_ids),
                             ("guards on vs off (all audited)", "neutral", "neutral_noguards", all_ids),
                             ("neutral vs neutral_leaky (all audited)", "neutral", "neutral_leaky", all_ids)]:
        if a not in judges or b not in judges:
            continue
        c = sides(judges[a], judges[b], audit, ids)
        report.setdefault("sides", {})[label] = c
        if c["n"]:
            lines.append(f"| {label} | {len(ids)} | {c['n']} | {c['a']} ({100 * c['a'] / c['n']:.0f}%) | "
                         f"{c['b']} ({100 * c['b'] / c['n']:.0f}%) |")
    unsure = sum(len(r.get("unsure", [])) for r in audit.values())
    total = sum(len(r["items"]) for r in audit.values())
    lines.append(f"\nAudit: {len(audit)} images, {total} requirements, {unsure} marked unsure.")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
