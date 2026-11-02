"""Judge-vs-reference agreement. Any two result files can be compared, including two reference judges."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict[str, dict]:
    """{image id: result} from a results JSONL (runner output or an exported reference)."""
    out = {}
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "result" in row:
                    out[row["id"]] = row["result"]
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = np.mean(a == b)
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def bootstrap(per_image: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, resampling images."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_image), size=(draws, len(per_image)))
    means = per_image[idx].mean(axis=1)
    return float(per_image.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(judge: dict[str, dict], reference: dict[str, dict], ids: list[str] | None = None) -> dict:
    """Requirement-level and image-level agreement of `judge` with `reference` on their common images."""
    ids = [i for i in (ids or judge) if i in judge and i in reference]
    if not ids:
        raise ValueError("no common images")
    full_j = np.array([judge[i]["all_satisfied"] for i in ids], dtype=bool)
    full_r = np.array([reference[i]["all_satisfied"] for i in ids], dtype=bool)
    out = {"images": len(ids),
           "full_success": {"judge": float(full_j.mean()), "reference": float(full_r.mean()),
                            "agreement": bootstrap((full_j == full_r).astype(float)),
                            "kappa": kappa(full_j, full_r),
                            "leniency": bootstrap(full_j.astype(float) - full_r.astype(float))}}
    if all(judge[i].get("items") for i in ids):
        pairs = [(a["satisfied"], b["satisfied"]) for i in ids
                 for a, b in zip(judge[i]["items"], reference[i]["items"], strict=True)]
        j, r = np.array(pairs, dtype=bool).T
        per_image = np.array([np.mean([a["satisfied"] == b["satisfied"] for a, b in
                                       zip(judge[i]["items"], reference[i]["items"], strict=True)]) for i in ids])
        out["requirements"] = {"n": len(pairs), "judge": float(j.mean()), "reference": float(r.mean()),
                               "agreement": bootstrap(per_image), "kappa": kappa(j, r),
                               "false_pass": float(np.mean(j & ~r)), "false_fail": float(np.mean(~j & r))}
    else:   # judges that only score images: ranking quality against the reference's full success
        scores = np.array([judge[i]["score"] for i in ids], dtype=float)
        out["full_success"]["auroc"] = auroc(scores, full_r)
    return out


def gains(results: dict[str, dict], pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Mean full-success change from the first to the second image of each pair (e.g. original -> rewrite)."""
    usable = [(a, b) for a, b in pairs if a in results and b in results]
    diff = np.array([float(results[b]["all_satisfied"]) - float(results[a]["all_satisfied"]) for a, b in usable])
    return bootstrap(diff)
