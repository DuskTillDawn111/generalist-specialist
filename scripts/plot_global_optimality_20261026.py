"""Render the global-optimality comparison in the paper-style two-panel layout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/paper-figures-20261025-final/data/02-static-utility-data.csv"
ORDER = ("C1", "C2", "C3", "C4", "C5")
LABELS = {
    "C1": "Selected",
    "C2": "+OCR replica",
    "C3": "Image split",
    "C4": "Pure-S",
    "C5": "Pure-G*",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    with SOURCE.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["candidate"] in ORDER]
    assert len(rows) == 15
    data = {}
    for candidate in ORDER:
        subset = [r for r in rows if r["candidate"] == candidate]
        values = {int(r["seed"]): float(r["U"]) for r in subset}
        assert sorted(values) == [43, 44, 45]
        data[candidate] = values
    return data


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .6,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": .5,
        "ytick.major.width": .5,
        "xtick.minor.width": .4,
        "ytick.minor.width": .4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.handlelength": 1.3,
        "legend.handletextpad": .4,
        "legend.borderaxespad": .3,
        "legend.labelspacing": .25,
        "grid.color": "#b0b0b0",
        "grid.linestyle": ":",
        "grid.linewidth": .4,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })


def panel(ax, tag, title):
    ax.set_title(f"({tag}) {title}", pad=6)


def render(data, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    configure_style()
    seeds = (43, 44, 45)
    markers = ("o", "^", "s")
    colors = ("#5d5d5d", "#9a9a9a", "#c0c0c0")
    blue = "#1f77b4"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.60))
    fig.subplots_adjust(left=.095, right=.985, bottom=.19, top=.85, wspace=.31)

    # C5 is many orders worse and is therefore reported in panel (b),
    # while panel (a) keeps the near-optimal deployment region readable.
    left = ("C1", "C2", "C3", "C4")
    x = np.arange(len(left))
    for seed, marker, color in zip(seeds, markers, colors):
        vals = [data[c][seed] for c in left]
        axes[0].plot(x, vals, color=color, lw=.65, marker=marker, ms=3.4,
                     markerfacecolor="white", markeredgewidth=.7,
                     label=f"Seed {seed}")
    means = [st.fmean(data[c][seed] for seed in seeds) for c in left]
    axes[0].scatter(x, means, marker="D", s=17, facecolor=blue, edgecolor=blue,
                    zorder=4, label="Mean")
    axes[0].set_xticks(x, [LABELS[c] for c in left], fontsize=7)
    axes[0].set_ylabel("Measured utility, $U$")
    axes[0].set_ylim(-.345, -.298)
    axes[0].set_yticks([-.34, -.33, -.32, -.31, -.30])
    panel(axes[0], "a", "Utility of competitive deployments")
    axes[0].grid(axis="y", alpha=.6)

    right = ("C2", "C3", "C4", "C5")
    xr = np.arange(len(right))
    for i, candidate in enumerate(right):
        gaps = [data["C1"][seed] - data[candidate][seed] for seed in seeds]
        assert all(gap > 0 for gap in gaps)
        axes[1].plot([i, i], [min(gaps), max(gaps)], color="#b0b0b0", lw=.7)
        for k, (value, marker, color) in enumerate(zip(gaps, markers, colors)):
            axes[1].plot(i + (k - 1) * .12, value, marker=marker, ms=3.4,
                         markerfacecolor="white", markeredgecolor=color,
                         markeredgewidth=.7)
        mean = st.fmean(gaps)
        axes[1].plot(i, mean, marker="D", ms=3.6, color=blue)
        axes[1].annotate(f"{mean:.3f}", (i, max(gaps)), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylim(.002, 16)
    axes[1].set_ylabel(r"Selected-plan advantage, $U_{\rm C1}-U_{\rm control}$")
    axes[1].set_xticks(xr, [LABELS[c] for c in right], fontsize=7)
    axes[1].set_xlim(-.45, 3.45)
    axes[1].yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    panel(axes[1], "b", "Paired utility differences")
    axes[1].grid(axis="y", which="major", alpha=.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.51, 1.02), ncol=4, columnspacing=1.5)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"global-optimality-utility.{ext}", dpi=600,
                    bbox_inches="tight", pad_inches=.035)
    plt.close(fig)

    means = {candidate: st.fmean(data[candidate][seed] for seed in seeds)
             for candidate in ORDER}
    gaps = {candidate: {str(seed): data["C1"][seed] - data[candidate][seed]
                        for seed in seeds} for candidate in right}
    payload = {
        "source_csv": str(SOURCE.resolve()),
        "source_csv_sha256": digest(SOURCE),
        "candidates": list(ORDER),
        "labels": LABELS,
        "utility": {c: {str(seed): data[c][seed] for seed in seeds} for c in ORDER},
        "mean_utility": means,
        "paired_advantage": gaps,
    }
    (output / "plot-data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "README.zh.md").write_text(
        "# 全局最优部署组合科研图\n\n"
        "左图展示 C1--C4 的三个 seed 总效用与均值；右图展示 C1 相对 C2--C5 的逐 seed 与均值优势。\n\n"
        "C5 的效用远低于其他候选，因此放在右图的对数优势轴中，避免压缩左图近优方案的差异。\n"
        "证据范围是冻结模型池、离散副本域和声明路由域内的有限搜索，不等同于连续物理 GPU 全局最优。\n"
    )
    manifest = {
        "source_csv": str(SOURCE.resolve()),
        "source_csv_sha256": digest(SOURCE),
        "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "script_sha256": digest(Path(__file__).resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "source_sha256": manifest["source_csv_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_data(), args.output.resolve())
