"""Render standalone Phase 1 CPU figures; no synthetic native gains."""

import argparse
import csv
import json
from pathlib import Path


def render(source, output, inner=None, dimensions=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = json.loads((source/"summary.json").read_text())
    rows = summary["records"]
    workloads = sorted({r["workload"] for r in rows})
    algorithms = sorted({r["algorithm"] for r in rows})
    colors = dict(zip(algorithms, plt.get_cmap("tab10").colors))
    for x, xlabel in (("elapsed_seconds","Optimization wall time (s)"),
                      ("full_prediction_calls","Full prediction bundles"),
                      ("outer_evaluations","Outer evaluations (DE only)")):
        fig, axes = plt.subplots(1,len(workloads), figsize=(5*len(workloads),4),squeeze=False)
        for ax, workload in zip(axes[0],workloads):
            for row in rows:
                if row["workload"] != workload:
                    continue
                raw = json.loads((source/row["raw_file"]).read_text())
                points = [p for p in raw.get("convergence",[]) if p.get(x) is not None and p.get("best_feasible_utility") is not None]
                if points:
                    ax.step([p[x] for p in points],[p["best_feasible_utility"] for p in points],
                            where="post",color=colors[row["algorithm"]],alpha=.5)
                elif x != "outer_evaluations" and row.get(x) is not None and row["utility"] is not None:
                    ax.scatter(row[x],row["utility"],color=colors[row["algorithm"]],marker="x")
            ax.set(title=workload,xlabel=xlabel,ylabel="Best feasible utility (higher is better)")
            ax.grid(alpha=.2)
        for algorithm in algorithms:
            axes[0][0].plot([],[],color=colors[algorithm],label=algorithm)
        axes[0][0].legend(fontsize=8)
        fig.suptitle("CPU synthetic fixture; baseline time shown as terminal points")
        fig.tight_layout()
        fig.savefig(output/f"utility-vs-{x}.png",dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1,len(workloads),figsize=(5*len(workloads),4),squeeze=False)
    for ax,w in zip(axes[0],workloads):
        values = [[r["utility"] for r in rows if r["workload"]==w and r["algorithm"]==a and r["utility"] is not None] for a in algorithms]
        ax.boxplot([v or [float("nan")] for v in values],tick_labels=algorithms)
        ax.tick_params(axis="x",labelrotation=35)
        ax.set(title=w,ylabel="Best feasible utility")
    fig.suptitle("Optimizer seed distribution; deterministic baselines repeat the same problem")
    fig.tight_layout()
    fig.savefig(output/"seed-distribution.png",dpi=160)
    plt.close(fig)
    # Chosen routing by workload, using the first seed rather than test ranking.
    first = min(r["seed"] for r in rows)
    selected = [r for r in rows if r["algorithm"]=="de_enumeration" and r["seed"]==first]
    tasks = sorted({t for r in selected if r["routing"] for t in r["routing"]})
    fig,axes = plt.subplots(1,max(1,len(tasks)),figsize=(6*max(1,len(tasks)),4),squeeze=False)
    for ax,task in zip(axes[0],tasks):
        models = sorted({m for r in selected if r["routing"] for m in r["routing"][task]})
        bottom = np.zeros(len(selected))
        for m in models:
            values = [r["routing"][task].get(m,0)/sum(r["routing"][task].values()) if r["routing"] else 0 for r in selected]
            ax.bar([r["workload"] for r in selected],values,bottom=bottom,label=m)
            bottom += values
        ax.set(title=task,ylabel="Route share")
        ax.legend()
    fig.suptitle("Frozen training winners; CPU fixture only")
    fig.tight_layout()
    fig.savefig(output/"routing-by-load.png",dpi=160)
    plt.close(fig)
    if inner:
        inner = Path(inner)
        with (inner/"inner-cost.csv").open() as stream:
            cost = list(csv.DictReader(stream))
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for name,marker in (("enumeration","o"),("milp","x")):
            selected = [r for r in cost if r["inner"]==name]
            axes[0].plot([int(r["route"]) for r in selected],[float(r["seconds"]) for r in selected],marker=marker,label=name)
            axes[1].plot([int(r["route"]) for r in selected],[int(r["predictions"]) for r in selected],marker=marker,label=name)
        for ax,y in zip(axes,("Seconds","Prediction bundles")):
            ax.set(xlabel="Paired fixed route",ylabel=y)
            ax.legend()
        fig.suptitle("Inner MILP versus enumeration; alternating run order")
        fig.tight_layout()
        fig.savefig(output/"inner-cost.png",dpi=160)
        plt.close(fig)
    if dimensions:
        dimension_rows = json.loads((Path(dimensions)/"summary.json").read_text())["records"]
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for algorithm in sorted({r["algorithm"] for r in dimension_rows}):
            selected=sorted((r for r in dimension_rows if r["algorithm"]==algorithm),key=lambda r:r["workload"])
            xs=[int(r["workload"][3]) for r in selected]
            axes[0].plot(xs,[r["elapsed_seconds"] for r in selected],marker="o",label=algorithm)
            axes[1].plot(xs,[r["full_prediction_calls"] for r in selected],marker="o",label=algorithm)
        for ax,y in zip(axes,("Optimization seconds","Full prediction bundles")):
            ax.set(xlabel="Continuous dimension",ylabel=y,xticks=[1,2,3])
            ax.legend()
        fig.suptitle("CPU dimension diagnostics; task sets differ, not an asymptotic scaling claim")
        fig.tight_layout()
        fig.savefig(output/"dimension-cost.png",dpi=160)
        plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--inner")
    parser.add_argument("--dimensions")
    args=parser.parse_args()
    render(args.source,args.output,args.inner,args.dimensions)
