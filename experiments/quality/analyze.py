"""Agreement tables: every judge against every reference, overall and per benchmark.

    python experiments/quality/analyze.py --data data/items/sol_subset.jsonl \
        --results experiments/quality/results/sol_subset --references data/references/sol.jsonl audit/claude.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastqwenjudge import metrics
from fastqwenjudge.data import load_items
from fastqwenjudge.judges.neutral import before_guards


def fmt(ci):
    return f"{100 * ci[0]:.1f} [{100 * ci[1]:.1f}, {100 * ci[2]:.1f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--references", nargs="+", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    items = load_items(args.data)
    groups = sorted({i.group for i in items})
    judged = {path.stem: metrics.load(path) for path in sorted(Path(args.results).glob("*.jsonl"))}
    if "neutral" in judged:
        judged["neutral_noguards"] = {k: before_guards(r) for k, r in judged["neutral"].items()}
    refs = {Path(r).stem: metrics.load(r) for r in args.references}
    report, lines = {}, []
    for ref_name, ref in refs.items():
        lines += [f"\n## vs {ref_name}\n", "| Judge | Images | Requirement agreement | kappa | Full-success agreement "
                  "| Leniency (full success) |", "|---|---:|---:|---:|---:|---:|"]
        candidates = {**judged, **{k: v for k, v in refs.items() if k != ref_name}}
        for name, res in candidates.items():
            ids = [i.id for i in items if i.id in res and i.id in ref]
            if not ids:
                continue
            r = metrics.compare(res, ref, ids)
            r["groups"] = {g: metrics.compare(res, ref, [i.id for i in items if i.group == g and i.id in res
                                                         and i.id in ref]) for g in groups
                           if any(i.group == g and i.id in res and i.id in ref for i in items)}
            report.setdefault(ref_name, {})[name] = r
            req = r.get("requirements")
            req_txt = f"{fmt(req['agreement'])} | {req['kappa']:.3f}" if req else \
                f"AUROC {r['full_success'].get('auroc', float('nan')):.3f} | -"
            fs = r["full_success"]
            lines.append(f"| {name} | {r['images']} | {req_txt} | {fmt(fs['agreement'])} | "
                         f"{100 * fs['leniency'][0]:+.1f} |")
    pairs = {}
    for i in items:
        pairs.setdefault(i.id.rsplit("__", 1)[0], {})[i.variant] = i.id
    pairs = [(v["original"], v["rewrite"]) for v in pairs.values() if {"original", "rewrite"} <= set(v)]
    lines += ["\n## Original -> rewrite gain in full success\n", "| Judge | Gain |", "|---|---:|"]
    for name, res in {**judged, **refs}.items():
        lines.append(f"| {name} | {fmt(metrics.gains(res, pairs))} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
