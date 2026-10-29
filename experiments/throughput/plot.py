"""Figures from the concurrency sweeps (experiments/throughput/results/sweep_*.json).

    python experiments/throughput/plot.py --out docs/figures

latency_throughput.png  median image latency against throughput, one line per setup, one point per concurrency
per_gpu.png             throughput per GPU against concurrency
cost.png                cost per 1,000 images against median latency, with the API reference judge's cost
gpu.png                 requests running, KV cache in use and power draw against concurrency
efficiency.png          MFU, memory bandwidth utilization and tokens per engine step against concurrency
token_latency.png       time to first token, inter-token latency and queueing time against concurrency
stages.png              where the time goes for one image at a time (concurrency 1)
makespan_<name>.png     one batch submitted at once (makespan.py): images finished and requests running over time,
                        per scheduling policy

MFU and MBU need the checkpoints' safetensors headers (efficiency.py); pass --checkpoint name=path for a model
that is not in the Hugging Face cache under its name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from efficiency import efficiency, load_model  # noqa: E402

HERE = Path(__file__).parent
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed per setup, so a setup keeps its color and marker in every figure
SETUPS = {
    "stock": ("vLLM defaults*", "#2a78d6", "o"),
    "tuned": ("tuned", "#eb6834", "s"),
    "tuned_fp8": ("tuned, FP8 weights", "#1baf7a", "D"),
    "tp2": ("tuned, tensor parallel (2 GPUs)", "#eda100", "^"),
    "dp2": ("tuned, 2 replicas (2 GPUs)", "#e87ba4", "v"),
    "tuned_single": ("single-call judge, tuned", "#008300", "p"),
    "tuned_fp8_fp8kv": ("tuned, FP8 weights + FP8 KV cache", "#6250d6", "h"),
    # speculative decoding: the base setup's color and marker, hollow marker, line style by method (SPECULATION)
    "tuned_ngram": ("tuned + n-gram speculation", "#eb6834", "s"),
    "tuned_fp8_fp8kv_ngram": ("FP8 weights + FP8 KV cache + n-gram speculation", "#6250d6", "h"),
    "tuned_fp8_fp8kv_ngram_dynamic": ("FP8 + FP8 KV + n-gram, only below 97 running", "#6250d6", "h"),
    "tuned_fp8_fp8kv_eagle3_k3": ("FP8 + FP8 KV + EAGLE-3 head (ours)", "#6250d6", "h"),
}
SPECULATION = {"_ngram": (0, (4, 2)), "_ngram_dynamic": (0, (1, 1.5)), "_eagle3_k3": (0, (6, 2, 1, 2))}
STAGES = [("compile", "#86b6ef"), ("observe", "#2a78d6"), ("decide", "#184f95")]
# makespan runs, by policy; the cold first batch and the repeat of the first policy are drawn dashed and dotted
POLICIES = {"fifo": ("input order", "#2a78d6"), "lpt": ("longest first", "#eb6834"),
            "lpt_rank": ("longest first, started images first", "#1baf7a")}
# GPT-6 Sol at the published per-token prices, applied to the token usage it recorded on the 2,106 API-labelled
# images (references/sol.jsonl); those checklists are shorter than the sweep workload's
API_USD_PER_1000 = {"GPT-6 Sol API, standard": 6.63, "GPT-6 Sol API, batch": 3.32}


def steady(sweep: dict) -> list[dict]:
    """Points that reached steady state: images in flight (Little's law) within 25% of the concurrency."""
    pts = [q for q in sweep["points"] if q["images"]]
    for q in pts:
        if q["in_flight"] < 0.75 * q["concurrency"]:
            print(f"{sweep['name']}: c={q['concurrency']} left out, only {q['in_flight']:.0f} in flight")
    return [q for q in pts if q["in_flight"] >= 0.75 * q["concurrency"]]


def replicas(sweep: dict) -> int:
    return int(sweep["profile"]["args"].get("data-parallel-size", 1))


def gpus(sweep: dict) -> int:
    return int(sweep["profile"]["args"].get("tensor-parallel-size", 1)) * replicas(sweep)


def style(ax, xlabel: str, ylabel: str, grid: str = "both"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def line(ax, name: str, xs, ys):
    label, color, marker = SETUPS[name]
    style = next((v for k, v in SPECULATION.items() if name.endswith(k)), None)
    speculative = style is not None
    ax.plot(xs, ys, color=color, linewidth=2, linestyle=style or "-",
            solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round", zorder=2)
    ax.plot(xs, ys, linestyle="none", marker=marker, markersize=7, markerfacecolor=SURFACE if speculative else color,
            markeredgecolor=color if speculative else SURFACE, markeredgewidth=1.5, label=label, zorder=3)


def panel_figure(path: Path, title: str, rows: dict[str, list[dict]], specs: list[tuple]):
    """A row of panels against images in flight; specs are (y label, key, scale, log y)."""
    fig, axes = plt.subplots(1, len(specs), figsize=(11, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ylabel, key, scale, log) in zip(axes, specs, strict=True):
        for name, points in rows.items():
            points = [r for r in points if r.get(key) is not None]
            if points:
                line(ax, name, [r["concurrency"] for r in points], [scale * r[key] for r in points])
        ax.set_xscale("log", base=2)
        if log:
            ax.set_yscale("log")
        style(ax, "images in flight", ylabel)
    axes[0].set_title(title, loc="left", color=TEXT, fontsize=11, pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def figure(width=7.5, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def finish(fig, ax, path: Path, title: str, note: str | None = None, **legend):
    """Title, legend (outside the plot on the right unless placed by the caller) and footnote, then save."""
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT,
              **(legend or {"loc": "center left", "bbox_to_anchor": (1.01, 0.5)}))
    if note:
        fig.text(0.01, 0.01, note, color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def makespan_figure(path: Path, out: Path):
    """Images finished and requests running against time since the batch was submitted, one line per run."""
    batch = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    seen = set()
    for run in batch["runs"]:
        timeline = json.loads(path.with_name(f"{path.stem}.{run['run']}.timeline.json").read_text())
        label, color = POLICIES[run["policy"]]
        dash = "-"
        if run["run"] == "cold":
            label, dash = f"{label}, cold server", (0, (4, 2))
        elif run["policy"] in seen:
            label, dash = f"{label}, repeated", (0, (1, 2))
        else:
            seen.add(run["policy"])
        finish = sorted(i["finish"] for i in timeline["images"])
        axes[0].plot([0, *finish], range(len(finish) + 1), drawstyle="steps-post", color=color, linewidth=2,
                     linestyle=dash, label=f"{label}: {run['makespan_s']:.0f} s")
        samples = timeline["server"]
        axes[1].plot([q["t"] for q in samples], [q["running"] or 0 for q in samples], color=color, linewidth=1.5,
                     linestyle=dash)
    style(axes[0], "seconds since the batch was submitted", "images finished")
    style(axes[1], "seconds since the batch was submitted", "requests running in the server")
    first = batch["runs"][0]
    axes[0].set_title(f"One batch of {first['images']:,} images, {first['concurrency']} requests in flight "
                      f"({batch['name']})", loc="left", color=TEXT, fontsize=11, pad=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"{path.stem}.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / f'{path.stem}.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--gpu-hour-usd", type=float, default=3.00,
                   help="price of one H100-hour; on-demand H100 SXM listings ranged $2.69-4.29 in October 2026")
    p.add_argument("--checkpoint", nargs="*", default=[], help="model=path, see efficiency.py")
    args = p.parse_args()
    paths = dict(kv.split("=", 1) for kv in args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweeps = {}
    for path in sorted(Path(args.results).glob("sweep_*.json")):
        s = json.loads(path.read_text())
        if s["name"] in SETUPS and s["points"]:
            sweeps[s["name"]] = s
    sweeps = {k: sweeps[k] for k in SETUPS if k in sweeps}
    points = {name: steady(s) for name, s in sweeps.items()}
    models = {m: load_model(paths.get(m, m)) for m in {s["profile"]["model"] for s in sweeps.values()}}
    rows = {name: [efficiency(s, q, models[s["profile"]["model"]], args.gpu_hour_usd) for q in points[name]]
            for name, s in sweeps.items()}
    note = ("* vLLM defaults plus the three flags needed to start and stay up on one H100: max-model-len 6144, "
            "no video inputs, multimodal cache off.")

    fig, ax = figure()
    for name, pts in points.items():
        line(ax, name, [q["images_per_s"] for q in pts], [q["latency"]["p50"] for q in pts])
    ax.set_yscale("log")
    style(ax, "throughput (images/s)", "median latency per image (s, log scale)")
    finish(fig, ax, out / "latency_throughput.png",
           "Latency vs throughput, neutral judge unless noted (points: 1 to 256 images in flight)", note)

    fig, ax = figure()
    for name, s in sweeps.items():
        pts = points[name]
        line(ax, name, [q["concurrency"] for q in pts], [q["images_per_s"] / gpus(s) for q in pts])
    ax.set_xscale("log", base=2)
    style(ax, "images in flight", "images/s per GPU")
    finish(fig, ax, out / "per_gpu.png", "Throughput per GPU", note)

    fig, ax = figure()
    for name, s in sweeps.items():
        pts = points[name]
        usd = [1000 * gpus(s) * args.gpu_hour_usd / (3600 * q["images_per_s"]) for q in pts]
        line(ax, name, [q["latency"]["p50"] for q in pts], usd)
    for label, usd in API_USD_PER_1000.items():
        ax.axhline(usd, color=MUTED, linewidth=1)
        ax.annotate(label, (1, usd), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "median latency per image (s, log scale)", "USD per 1,000 images (log scale)")
    finish(fig, ax, out / "cost.png", f"Cost per 1,000 images at ${args.gpu_hour_usd:.2f} per GPU-hour",
           note + " API cost measured on shorter checklists.")

    # nvidia-smi's utilization.gpu only says a kernel was running; it reads above 90% at every concurrency, so
    # the panels show what limits the batch instead
    panel_figure(out / "gpu.png", "Server and GPU state during the measured windows", rows,
                 [("requests running per model replica", "running", 1, False),
                  ("KV cache in use (%)", "kv_cache_used", 100, False),
                  ("power per GPU (W)", "power_w_per_gpu", 1, False)])
    panel_figure(out / "efficiency.png", "Share of the GPU in use (H100 SXM dense peaks)", rows,
                 [("MFU (% of peak tensor FLOPS)", "mfu", 100, False),
                  ("MBU (% of peak HBM bandwidth)", "mbu", 100, False),
                  ("tokens computed per engine step", "tokens_per_step", 1, True)])
    panel_figure(out / "token_latency.png", "Request latency inside the server", rows,
                 [("time to first token, median (s)", "ttft_p50", 1, True),
                  ("inter-token latency, mean (ms)", "itl_mean_ms", 1, False),
                  ("time queued, mean (s)", "queue_mean", 1, True)])

    single = [(name, next(q for q in s["points"] if q["concurrency"] == 1)) for name, s in sweeps.items()
              if any(q["concurrency"] == 1 and q["images"] for q in s["points"])]
    fig, ax = figure(height=0.45 * len(single) + 1.4)
    seen = set()
    for y, (_, q) in enumerate(single):
        left = 0.0
        for stage, color in STAGES + [("single_v1", "#008300")]:
            if stage in q["stages"]:
                width = q["stages"][stage]["latency"]["p50"]
                ax.barh(y, width, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2,
                        label=None if stage in seen else stage)
                seen.add(stage)
                left += width
        ax.annotate(f"{left:.1f} s", (left, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT)
    ax.set_yticks(range(len(single)), [SETUPS[n][0] for n, _ in single])
    ax.invert_yaxis()
    style(ax, "median request latency, one image at a time (s)", "", grid="x")
    finish(fig, ax, out / "stages.png", "Single-image latency by request")

    for path in sorted(Path(args.results).glob("makespan_*.json")):
        if "." not in path.stem:   # not a per-run timeline
            makespan_figure(path, out)


if __name__ == "__main__":
    main()
