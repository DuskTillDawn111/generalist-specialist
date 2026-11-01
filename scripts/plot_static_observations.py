"""Export descriptive GPU figures from strictly imported observation reports."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy.evidence import digest
from gsdeploy.io import save


def plot(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = json.loads(report_path.read_text())
    observations = report["observations"]
    if not observations or any(row["strict_native_import"] != "PASS" for row in observations):
        raise ValueError("Only verified native observation reports may be plotted")
    output.mkdir(parents=True, exist_ok=False)
    names = {"shared_generalist_2": "Generalist G2", "specialists_shared_board": "Four specialists"}
    labels = [names.get(row["labels"][0], row["labels"][0].replace("mixed_generalist_2_", "G2 + ")
                        .replace("_specialist_share_", " share=")) for row in observations]
    terms = [("Q", "Quality loss"), ("T", "Throughput loss"), ("L", "Mean latency"),
             ("P", "P95 latency"), ("M", "Memory"), ("E", "Board energy")]
    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * .7)))
    offsets = np.zeros(len(observations))
    for (term, title), color in zip(terms, plt.cm.tab10.colors):
        values = np.array([-row["measured_utility"][term + "_contribution"] for row in observations])
        ax.barh(labels, values, left=offsets, label=title, color=color)
        offsets += values
    for index, (row, value) in enumerate(zip(observations, offsets)):
        ax.annotate(f"U={row['measured_utility']['U']:.4f}", (value, index), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Weighted cost = -U (smaller is better)")
    ax.set_title("Measured six-term utility at the fixed workload")
    ax.set_xlim(0, max(offsets) * 1.2)
    ax.invert_yaxis()
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.31))
    fig.text(.01, .01, "Descriptive calibration conditions; no independent confirmation or full-pool optimum inferred.", fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, 1))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / ("measured-utility-costs." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(observations), figsize=(max(7, len(observations) * 3), 4), squeeze=False)
    for ax, row, label in zip(axes[0], observations, labels):
        tasks = list(row["per_task"])
        service = [row["per_task"][t]["service"]["mean_ms"] / 1000 for t in tasks]
        wait = [row["per_task"][t]["waiting_plus_transport"]["mean_ms"] / 1000 for t in tasks]
        ax.bar(tasks, service, label="Worker service", color="#4477AA")
        ax.bar(tasks, wait, bottom=service, label="Wait + transport", color="#EE9944")
        ax.set_title(label, fontsize=9, wrap=True)
        ax.set_ylabel("Mean end-to-end latency (s)")
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean latency decomposition on the measured cohort")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / ("measured-latency-decomposition." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    save(output / "manifest.json", {"report": {"path": str(report_path), "sha256": digest(report_path)},
        "files": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "scope": "Descriptive native calibration evidence only", "gpu_started": False})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.report, args.output))
