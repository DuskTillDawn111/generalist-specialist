"""Export the motivation figure and results 2/3/4 from frozen native evidence.

Run from any directory. No solver or GPU measurement is run. Existing study
packages are read only; all generated files go into a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics as st
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_static_confirmation import verify_report
from plot_static_search_efficiency import audit

ROOT = Path(__file__).resolve().parents[1]
STATIC = "artifacts/static-study-gpu-20261022/"
SPILL = "artifacts/static-spillover-study-20261022/"
SLOTS = ("generalist_1", "generalist_2", "generalist_3")
TASKS = ("text", "image", "ocr", "asr")
TASK_NAMES = ("Text QA", "Image", "OCR", "ASR")
SHORT = {"generalist_1": "Baichuan-Omni", "generalist_2": "MiniCPM-o", "generalist_3": "Qwen2.5-Omni",
         "text": "ChatQA", "image": "SigLIP2", "ocr": "PP-OCRv6", "asr": "Granite Speech"}
FULL = {"generalist_1": "baichuan-inc/Baichuan-Omni-1d5", "generalist_2": "openbmb/MiniCPM-o-4_5-GPTQ",
        "generalist_3": "Qwen/Qwen2.5-Omni-7B (BNB4)", "text": "nvidia/Llama3-ChatQA-1.5-8B (INT8+LoRA)",
        "image": "google/siglip2-giant-opt-patch16-384", "ocr": "PP-OCRv6-medium-det+rec",
        "asr": "ibm-granite/granite-speech-4.1-2b"}
BLUE, ORANGE, RED, GREEN = "#2575ad", "#ee8b22", "#c34a45", "#288775"
MODEL_COLORS = [BLUE, "#f5bb75", ORANGE, "#b96a13"]
TERM_COLORS = ["#638fbb", "#c3cbd4", "#e1a24d", "#c26753", "#8f80ac", "#6fa497"]
TERMS = ("Q", "T", "L", "P", "M", "E")
TERM_LABELS = ("Q: quality loss", "T: window loss", "L: mean delay", "P: P95 delay", "M: memory", "E: energy")
RECEIPTS = {}
CHECKS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def track(path, expected=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if str(p) in RECEIPTS:
        digest = RECEIPTS[str(p)]["sha256"]
    else:
        digest = sha(p)
        RECEIPTS[str(p)] = {"path": str(p), "sha256": digest, "bytes": p.stat().st_size}
    if expected is not None:
        assert digest == expected, f"Receipt mismatch: {p}"
    return p


def read(path):
    return json.loads(track(path).read_text())


def rows(path):
    with track(path).open(newline="") as f:
        return list(csv.DictReader(f))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, data):
    if not data:
        return
    fields = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in data:
            w.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def cost_audit(row, terms=TERMS, prefix="", suffix=""):
    close(float(row["U"]), -st.fmean(float(row[prefix + t + suffix]) for t in terms))


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#74808c", "axes.linewidth": .7,
        "xtick.color": "#34404c", "ytick.color": "#34404c", "text.color": "#203040",
        "axes.labelcolor": "#203040", "grid.color": "#dce2e8", "grid.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white"})


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", pad=14)
    ax.set_axisbelow(True)


def title(fig, main, sub):
    fig.suptitle(main, x=.055, y=.975, ha="left", fontsize=19, weight="bold")
    subtitle_y = min(.927, .975 - .43 / fig.get_figheight())
    fig.text(.055, subtitle_y, sub, ha="left", fontsize=10, color="#5c6876")


def finish(fig, out, name, pdf, footer):
    fig.text(.055, .028, footer, fontsize=8.5, color="#5c6876", va="bottom", linespacing=1.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=240, bbox_inches="tight", pad_inches=.18)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def motivation(out, pdf):
    measured = read("artifacts/retained-replica-profiles-20261021/combined-summary.json")
    config = read(STATIC + "search-v2-v5-K8/system-config.json")
    q = config["catalog"]["quality"]
    profiles = {(r["slot"], r["task"]): r for r in measured["rows"]
                if r["replicas"] == 1 and r["status"] == "COMPLETE"}
    assert len(profiles) == 16
    data = []
    for task in TASKS:
        ids = set(q["input_order"][task])
        for slot in (task,) + SLOTS:
            r, quality = profiles[slot, task], q["paths"][slot][task]
            assert set(quality["samples"]) == ids and len(ids) == r["unique_inputs"]
            track(r["profile"])
            value = quality["value"]
            data.append(dict(task=task, slot=slot, model=FULL[slot], model_short=SHORT[slot],
                model_class="specialist" if slot == task else "generalist", quality_metric=quality["metric"],
                native_quality=value, higher_better_quality=1-value if task in ("ocr", "asr") else value,
                mean_service_ms=r["mean_service_ms"], p95_service_ms=r["p95_service_ms"],
                sampled_owned_process_peak_GiB=r["simultaneous_process_peak_sum_GiB"],
                unique_inputs=r["unique_inputs"], profiling_requests=r["requests"],
                profile=r["profile"]))
    by = {(r["slot"], r["task"]): r for r in data}
    fig = plt.figure(figsize=(19, 10.4))
    title(fig, "Motivation | Task-wise quality, latency and GPU memory",
          "Frozen seven-model pool  /  single retained replica  /  batch = 1  /  measured task paths")
    gs = fig.add_gridspec(2, 12, left=.075, right=.985, top=.82, bottom=.16, hspace=.62, wspace=1.65,
                          height_ratios=[1.05, 1])
    metrics = ["QA EM", "Accuracy", "CER", "WER"]
    for i, task in enumerate(TASKS):
        ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
        rr = [by[s, task] for s in (task,) + SLOTS]
        ax.barh(np.arange(4), [r["mean_service_ms"] for r in rr], color=MODEL_COLORS,
                height=.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(80, max(r["mean_service_ms"] for r in rr) * 150)
        ax.set_ylim(3.7, -.65)
        ax.set_yticks(range(4), [r["model_short"] for r in rr], fontsize=8)
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.set_xlabel("Mean service (ms, log scale)", fontsize=9)
        ax.grid(axis="x", which="major", alpha=.8)
        panel(ax, "abcd"[i], f"{TASK_NAMES[i]}  |  {metrics[i]} {'↓' if i>1 else '↑'}")
        for j, r in enumerate(rr):
            metric_label = ("EM", "Acc", "CER", "WER")[i]
            ax.annotate(f"{r['mean_service_ms']:,.1f} ms\n{metric_label} {r['native_quality']:.3f} | {r['sampled_owned_process_peak_GiB']:.2f} GiB",
                        (r["mean_service_ms"], j), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7.5)
    axes = [fig.add_subplot(gs[1, i*4:(i+1)*4]) for i in range(3)]
    x, width = np.arange(4), .18
    for idx, (key, axis_title, ylabel) in enumerate([
            ("higher_better_quality", "Quality comparison", "EM / accuracy / 1 - CER / 1 - WER"),
            ("mean_service_ms", "Latency comparison", "Mean service (ms, log scale)"),
            ("sampled_owned_process_peak_GiB", "Memory comparison", "Sampled process peak (GiB)")]):
        ax = axes[idx]
        for j, (slot, label, color) in enumerate(zip((None,) + SLOTS,
                ("Task specialist", "Baichuan-Omni", "MiniCPM-o", "Qwen2.5-Omni"), MODEL_COLORS)):
            vals = [by[slot or t, t][key] for t in TASKS]
            b = ax.bar(x + (j-1.5)*width, vals, width, label=label, color=color, zorder=3)
            ax.bar_label(b, labels=[f"{v:.3f}" if idx==0 else f"{v:,.0f}" if idx==1 else f"{v:.1f}" for v in vals],
                         padding=3, fontsize=6.8, rotation=90 if idx==0 else 0)
        panel(ax, "efg"[idx], axis_title)
        ax.set_xticks(x, TASK_NAMES)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y")
        if idx == 0:
            ax.set_ylim(0, 1.17)
            ax.set_yticks([0,.25,.5,.75,1])
        elif idx == 1:
            ax.set_yscale("log")
            ax.set_ylim(70, 95000)
        else:
            ax.set_ylim(0, 27)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54,.901),
               ncol=4, frameon=False, fontsize=10)
    finish(fig, out, "01-motivation", pdf,
        "Quality: same unique inputs per task (QA/OCR/ASR: 32; image: 64); latency: repeated profiling requests over three rounds.\n"
        "Memory: sampled simultaneous owned-process peak, including workspace. OCR uses project-reviewed references, not externally certified gold.")
    write_csv(out / "01-motivation-data.csv", data)
    CHECKS["motivation"] = dict(paths=16, unique_input_ids_match=True, memory="sampled simultaneous owned-process peak", quality_scope=q["scope"])
    return data


def static_result(out, pdf):
    report_path = track(STATIC + "confirmation-report-v1/report.json")
    report, packet = verify_report(report_path)
    track(report["packet"]["path"], report["packet"]["sha256"])
    for r in report["runs"]:
        track(r["path"], r["sha256"])
    role_order = ("free_search_winner", "strongest_alternative_deployment", "strongest_alternative_route",
                  "best_specialist_only", "best_generalist_only")
    names = ["Selected", "Extra OCR replica", "Image split", "Pure specialist", "Pure-G candidate*"]
    ticks = ["C1\nSelected", "C2\nExtra OCR", "C3\nImage split", "C4\nPure-S", "C5\nPure-G*"]
    candidates = [next(c for c in packet["candidates"] if role in c["labels"]) for role in role_order]
    groups = [[r for r in report["runs"] if r["plan_key"] == c["plan_key"]] for c in candidates]
    means = [st.fmean(r["utility"]["U"] for r in group) for group in groups]
    fig = plt.figure(figsize=(15.5, 10))
    title(fig, "Result 2 | Static deployment selection and GPU confirmation",
          "Four tasks at 0.074836 requests/s each  /  Poisson  /  1,800 s  /  independent seeds 43, 44, 45")
    gs = fig.add_gridspec(2, 2, left=.075, right=.965, top=.845, bottom=.19, wspace=.26, hspace=.59)
    seedcolors = ["#527ba8", "#df9259", "#6a9b8c"]
    for j, selection in enumerate((range(5), range(4))):
        ax = fig.add_subplot(gs[0, j])
        for i in selection:
            vals = [next(r["utility"]["U"] for r in groups[i] if r["seed"]==seed) for seed in (43,44,45)]
            ax.plot([i,i], [min(vals),max(vals)], color="#b5bec8", lw=2, zorder=1)
            for k, (v, color) in enumerate(zip(vals, seedcolors)):
                ax.scatter(i+(k-1)*.12, v, s=35, color=color, label=f"Seed {43+k}" if i==0 else None)
            ax.scatter(i, means[i], marker="D", color="#172c42", s=43, zorder=5,
                       label="Mean" if i==0 else None)
            ax.annotate(f"{means[i]:.4f}", (i, means[i]), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(list(selection), [ticks[i] for i in selection], fontsize=9)
        ax.set_ylabel("Measured total utility U (higher is better)")
        ax.grid(axis="y")
        ax.set_xlim(-.45, max(selection)+.45)
        if j == 0:
            ax.set_ylim(-6.7, .4)
            ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower left")
        else:
            ax.set_ylim(-.350, -.287)
        panel(ax, "ab"[j], "All five measured candidates" if j==0 else "Four competitive candidates, expanded scale")
    ax = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(4)
    for t, color, label in zip(TERMS, TERM_COLORS, TERM_LABELS):
        v = [st.fmean(r["weighted_costs"][t] for r in g) for g in groups[:4]]
        ax.bar(range(4), v, bottom=bottom, color=color, width=.56, label=label, zorder=3)
        bottom += v
    ax.set_xticks(range(4), ticks[:4], fontsize=9)
    ax.set_ylabel("Mean weighted loss = -U (lower is better)")
    ax.set_ylim(0,.41)
    ax.grid(axis="y")
    panel(ax, "c", "Six-term cost decomposition")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(-.02,1.01))
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    panel(ax, "d", "Selected deployment and task routing")
    data = [["Text QA", "MiniCPM-o x1", "100%"], ["ASR", "MiniCPM-o x1 (shared)", "100%"],
            ["Image", "SigLIP2 x1", "100%"], ["OCR", "PP-OCRv6 x1", "100%"]]
    table = ax.table(cellText=data, colLabels=["Task", "Destination", "Traffic share"], cellLoc="left",
                     colLoc="left", colWidths=[.22,.55,.23], bbox=[0,.26,1,.72])
    table.auto_set_font_size(False);table.set_fontsize(10)
    format_table(table)
    ax.text(0,.13, "3 resident replicas; no task splitting in selected plan", transform=ax.transAxes, fontsize=10, color=BLUE)
    ax.text(0,.015, "Predicted U = -0.309303; measured mean U = -0.312130", transform=ax.transAxes, fontsize=9)
    finish(fig,out,"02-static-deployment",pdf,
        "U = -(Q + T + L + P + M + E) / 6; no absolute quality gate. C1 also equals the best no-split candidate.\n"
        "*C5 is a frozen pure-generalist budget candidate, not a certified best pure-G plan. C1 leads every measured seed; n = 3, exact paired p = 0.25.\n"
        "Prediction certificate covers the frozen finite domain; GPU results compare these five plans and do not prove physical global optimality.")
    export=[]
    plans=[]
    for i,(c,g) in enumerate(zip(candidates,groups)):
        for r in g:
            export.append(dict(candidate=f"C{i+1}",name=names[i],seed=r["seed"],U=r["utility"]["U"],
                **{t+"_norm":r["utility"][t+"_norm"] for t in TERMS},
                **{t+"_weighted_cost":r["weighted_costs"][t] for t in TERMS},raw_path=r["path"],raw_sha256=r["sha256"]))
        plans.append(dict(candidate=f"C{i+1}",name=names[i],mean_U=means[i],plan_key=c["plan_key"],**c["plan"]))
    write_csv(out/"02-static-utility-data.csv",export)
    dump(out/"02-static-plans.json",plans)
    write_csv(out/"02-static-plans.csv",plans)
    # Companion table: every physical comparator's deployment and routing.
    tabledata=[
        ["C1 Selected", "G2 x1 + Image x1 + OCR x1", "QA/ASR -> G2; Image -> S; OCR -> S", f"{means[0]:.6f}"],
        ["C2 Extra OCR", "G2 x1 + Image x1 + OCR x2", "Same task shares as C1", f"{means[1]:.6f}"],
        ["C3 Image split", "Same resident deployment as C1", "Image: G2 1/6, S 5/6; others as C1", f"{means[2]:.6f}"],
        ["C4 Pure-S", "ChatQA/Image/OCR/ASR: each x1", "Each task -> its own specialist", f"{means[3]:.6f}"],
        ["C5 Pure-G*", "G2 x1 + Qwen2.5-Omni x1", "QA -> Qwen; Image/ASR -> G2;\nOCR: G2 5/6, Qwen 1/6", f"{means[4]:.6f}"]]
    fig,ax=plt.subplots(figsize=(16,5.2));ax.axis("off")
    title(fig,"Result 2 | Concrete comparator designs", "G2 = MiniCPM-o-4_5-GPTQ; S = corresponding task specialist; xN = resident replica count")
    tab=ax.table(cellText=tabledata,colLabels=["Candidate","Resident deployment","Routing (fraction of task arrivals)","Mean measured U"],
                 cellLoc="left",colLoc="left",colWidths=[.14,.28,.43,.15],bbox=[0,.12,1,.72])
    tab.auto_set_font_size(False);tab.set_fontsize(10);format_table(tab)
    fig.subplots_adjust(left=.055,right=.97,top=.84,bottom=.16)
    finish(fig,out,"02-static-comparator-table",pdf,
        "All five candidates: identical arrival traces within seed; seeds 43/44/45; six-term equal weights.\n*Pure-G budget candidate: subdomain optimality was not established.")
    CHECKS["static"] = dict(status=report["status"],completed_runs=15,receipt_and_scoring_audit="PASS",candidates=5)
    return export,plans


def format_table(table):
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor("#e2e7ec");cell.set_linewidth(.7);cell.PAD=.055
        if i==0:
            cell.set_facecolor("#eaf0f6");cell.set_text_props(weight="bold", color="#233d55")
        elif i%2:
            cell.set_facecolor("#f6f8fb")


def efficiency(out,pdf):
    report, checked = audit(track(STATIC+"five-model-v2-v5-K4-final/report.json"),
                            track(STATIC+"surrogate-v2/system-config.json"))
    for r in checked["input_receipts"]:
        track(r["path"],r["sha256"])
    dump(out/"03-dfs-audit.json",checked)
    algorithms=("milp","dfs","plain_dfs");labels=("MILP (ours)","Bounded DFS","Plain DFS")
    colors=(BLUE,GREEN,ORANGE)
    fig=plt.figure(figsize=(14.5,9.2))
    title(fig,"Result 3 | Same optimum, fewer evaluations, faster CPU search",
          "Five-model domain: MiniCPM-o + four specialists; each absent or x1; routing denominators 1..4; five fresh-process repeats")
    gs=fig.add_gridspec(2,2,left=.08,right=.965,top=.835,bottom=.16,wspace=.3,hspace=.61)
    exports=[]
    for j,(field,label) in enumerate((("search_wall_seconds","Search wall time (s, log scale)"),
                                       ("parent_process_wall_seconds","Whole-process wall time (s, log scale)"))):
        ax=fig.add_subplot(gs[0,j])
        for i,(algo,c) in enumerate(zip(algorithms,colors)):
            v=[r[field] for r in report["records"] if r["algorithm"]==algo]
            median=st.median(v)
            ax.bar(i,median,color=c,width=.48,alpha=.26,zorder=2)
            ax.scatter(i+np.linspace(-.10,.10,5),v,color=c,s=27,zorder=3)
            ax.plot([i-.25,i+.25],[median,median],color=c,lw=2.3)
            ax.annotate(f"{median:.3f} s",(i,max(v)),xytext=(0,10),textcoords="offset points",ha="center",weight="bold",fontsize=10)
        ax.set_yscale("log");ax.set_ylim(.1 if j==0 else .3,100)
        ax.set_xticks(range(3),labels);ax.set_ylabel(label);ax.grid(axis="y",which="major")
        panel(ax,"ab"[j],"Search time" if j==0 else "Including process startup and imports")
        summaries={r["algorithm"]:r for r in report["summaries"]}
        k="median_search_wall_seconds" if j==0 else "median_parent_wall_seconds"
        ax.text(.03,.91,f"MILP vs plain DFS: {summaries['plain_dfs'][k]/summaries['milp'][k]:.2f}x\n"
                f"MILP vs bounded DFS: {summaries['dfs'][k]/summaries['milp'][k]:.2f}x",
                transform=ax.transAxes,va="top",fontsize=10,bbox=dict(facecolor="white",edgecolor="none",alpha=.85))
    ax=fig.add_subplot(gs[1,0])
    counts=[next(r for r in report["summaries"] if r["algorithm"]==a)["median_full_predictions"] for a in algorithms]
    bars=ax.bar(range(3),counts,color=colors,width=.48,zorder=3)
    ax.set_yscale("log");ax.set_ylim(.6,6000);ax.bar_label(bars,labels=[f"{int(v):,}" for v in counts],padding=7,fontsize=11,weight="bold")
    ax.set_xticks(range(3),labels);ax.set_ylabel("Complete predictor evaluations (log scale)");ax.grid(axis="y")
    panel(ax,"c","Evaluation count (identical in all five repeats)")
    ax=fig.add_subplot(gs[1,1]);ax.axis("off")
    panel(ax,"d","Agreement with exhaustive plain DFS")
    tab=ax.table(cellText=[["Same objective value","5 / 5"],["Same deployment + routing","5 / 5"],["All three certificates complete","5 / 5"]],
                 colLabels=["Per-repeat check","Passed"],cellLoc="left",colLoc="left",colWidths=[.77,.23],bbox=[0,.32,1,.66])
    tab.auto_set_font_size(False);tab.set_fontsize(10);format_table(tab)
    score=report["records"][0]["result"]["winner"]["score"]
    ax.text(0,.215,f"Identical optimum U = {score:.9f}",transform=ax.transAxes,weight="bold",color=BLUE)
    ax.text(0,.035,"Winner: G2 x1 + Image x1 + OCR x1\nQA/ASR -> G2; Image/OCR -> specialists",transform=ax.transAxes,fontsize=10,linespacing=1.6)
    finish(fig,out,"03-dfs-efficiency",pdf,
        "Dots = fresh-process repeats; bars / horizontal strokes = medians. Serial interleaved runs with the same trace, predictor and CPU affinity.\n"
        "Finite calibrated prediction domain: 4,097 route/deployment identities before exclusions; plain DFS makes 1,696 full evaluations.\n"
        "This is CPU search efficiency and finite-domain agreement; it is not inference speedup or a seven-model physical global-optimum proof.")
    for r in report["records"]:
        exports.append(dict(repeat=r["repeat"],algorithm=r["algorithm"],search_wall_seconds=r["search_wall_seconds"],
            whole_process_wall_seconds=r["parent_process_wall_seconds"],full_prediction_calls=r["result"]["metrics"]["full_prediction_calls"],
            optimum_U=r["result"]["winner"]["score"],certified=r["result"]["optimality_certified"],
            deployment=r["result"]["winner"]["deployment"],routing=r["result"]["winner"]["routing"]))
    write_csv(out/"03-dfs-efficiency-data.csv",exports)
    write_csv(out/"03-dfs-efficiency-summary.csv",report["summaries"])
    CHECKS["efficiency"] = dict(status=checked["status"],identical_optimum=checked["identical_optimum"],repeats=5,certificates=15)
    return exports


def spillover_data():
    data=[]
    def add(stage,seed,on,off,source,terms=6,contrib=None):
        data.append(dict(stage=stage,seed=seed,U_spill=on,U_control=off,delta_U=None if on is None or off is None else on-off,
                         objective_terms=terms,source=source,contribution_delta=contrib or {},
                         status="UNKNOWN" if on is None or off is None else "MEASURED"))
    p=SPILL+"text185-measured-comparison-v7/report.json";d=read(p)
    for seed in (107,108):
        on=next(r for r in d["observations"] if r["label"]==f"text185_g2share_1of32_seed{seed}")
        off=next(r for r in d["observations"] if r["label"]==f"text185_g2share_0of1_seed{seed}")
        assert on["cohort_sha256"]==off["cohort_sha256"]
        for r in (on,off):
            track(r["raw_receipt"]["path"],r["raw_receipt"]["sha256"])
        add("fixed_development",seed,on["utility"]["U"],off["utility"]["U"],p)
    p=SPILL+"search-top-two-prequential-development-v1/execution-v1/whole-pair-uncertainty/report.json";d=read(p)
    track(d["source"]["path"],d["source"]["sha256"])
    native=read(d["source"]["path"])
    for r in native["observations"]:
        track(r["raw_receipt"]["path"],r["raw_receipt"]["sha256"])
    for r in d["pairs"]:
        close(r["delta_U"],r["spillover_U"]-r["no_spillover_U"])
        add("fixed_validation",r["seed"],r["spillover_U"],r["no_spillover_U"],p,contrib=r["delta_original_contributions"])
    p="artifacts/queue-aware-study-20261023/final-report-v1/report.json";d=read(p)
    # Upstream names the winning no-spill arm 'selected'; reconstruct sign explicitly.
    for phase in ("independent-validation","final-confirmation"):
        obs=d["stages"][phase]["observations"]
        for seed in d["stages"][phase]["seeds"]:
            on=next(r for r in obs if r["seed"]==seed and r["candidate"]["strategy"]=="queue_latency")
            off=next(r for r in obs if r["seed"]==seed and r["candidate"]["strategy"]=="queue_specialist_only")
            for r in (on,off):
                track(r["raw"])
                cost_audit(r["terms"]|{"U":r["U"]})
            add("queue_aware",seed,on["U"],off["U"],p,contrib={t:on["contributions"][t]-off["contributions"][t] for t in TERMS})
    p="artifacts/queue-refined-study-20261023/final-report-v1/native-runs.csv";d=rows(p)
    for seed in range(205,210):
        on=next(r for r in d if int(r["seed"])==seed and r["role"]=="calibrated_spill")
        off=next((r for r in d if int(r["seed"])==seed and r["role"]=="risk_matched_no_spill"),None)
        for r in (on,off):
            if r:
                track(r["raw_path"],r["raw_sha256"])
                if r["U"]:cost_audit(r,suffix="_norm")
        add("refined",seed,float(on["U"]) if on["U"] else None,float(off["U"]) if off else None,p,
            contrib={t:float(on["contribution_"+t])-float(off["contribution_"+t]) for t in TERMS} if off else {})
    p="artifacts/queue-coupled-study-20261023/final-report-v1/native-results.csv";d=rows(p)
    for r in d:cost_audit(r)
    for seed in (303,304):
        on=next(r for r in d if int(r["seed"])==seed and r["role"]=="coupled_spill")
        off=next(r for r in d if int(r["seed"])==seed and r["role"]=="matched_no_spill")
        add("coupled",seed,float(on["U"]),float(off["U"]),p,
            contrib={t:-(float(on[t])-float(off[t]))/6 for t in TERMS})
    coupled=rows("artifacts/queue-coupled-study-20261023/native-endpoint-diagnostics-v1/deployment-routing-metrics.csv")
    p="artifacts/queue-five-term-study-20261023/report-final-v1/runs.csv";d=rows(p)
    for r in d:
        rr=json.loads(r["raw"]);track(rr["path"],rr["sha256"])
        cost_audit(r,terms=tuple("QTLME"),suffix="_norm")
    for seed in (401,402):
        on=next(r for r in d if int(r["seed"])==seed and r["role"]=="coupled_spill")
        off=next(r for r in d if int(r["seed"])==seed and r["role"]=="matched_no_spill")
        assert on["cohort_signature"]==off["cohort_signature"]
        add("no_p95",seed,float(on["U"]),float(off["U"]),p,terms=5,
            contrib={t:-(float(on[t+"_norm"])-float(off[t+"_norm"]))/5 for t in "QTLME"})
    for r in data:
        if r["contribution_delta"] and r["delta_U"] is not None:
            close(sum(r["contribution_delta"].values()),r["delta_U"])
    assert len(data)==20 and sum(r["status"]=="MEASURED" for r in data)==19
    CHECKS["spillover"] = dict(measured_pairs=19,unknown_pairs=1,delta_sign="spill minus own matched no-spill control",pooled_across_stages=False)
    return data,d,coupled


def spillover(out,pdf):
    data,five,coupled=spillover_data()
    stages=("fixed_development","fixed_validation","queue_aware","refined","coupled","no_p95")
    titles=("Fixed 1/32: initial development","Fixed 1/32: new seeds","Queue-aware: independent pairs",
            "Refined: validation + confirmation","GPU-coupled: development","P95 removed: development")
    fig,axes=plt.subplots(2,3,figsize=(16.7,10.2))
    title(fig,"Result 4 | Autonomous spillover exists; stable utility gain is unproven",
          "Text 1.85 requests/s; image / OCR / ASR 0.01 each  /  600 s  /  ChatQA x2 + MiniCPM-o x1 + OCR x1")
    fig.subplots_adjust(left=.075,right=.98,top=.815,bottom=.17,hspace=.63,wspace=.35)
    for i,(ax,stage,heading) in enumerate(zip(axes.flat,stages,titles)):
        rr=[r for r in data if r["stage"]==stage]
        known=[r for r in rr if r["delta_U"] is not None]
        for x,r in enumerate(rr):
            if r["delta_U"] is None:
                ax.text(x,.65,"UNKNOWN\nE / U missing;\ncontrol not run",transform=ax.get_xaxis_transform(),ha="center",va="center",
                        fontsize=8,color="#7b8390",bbox=dict(facecolor="#f2f4f7",edgecolor="#d4dbe4",boxstyle="round,pad=.35"))
                continue
            v=r["delta_U"];c=GREEN if v>0 else RED
            ax.bar(x,v,width=.5,color=c,alpha=.25,zorder=2)
            ax.plot([x,x],[0,v],color=c,lw=2,zorder=3)
            ax.scatter(x,v,color=c,s=40,zorder=4)
            ax.annotate(f"{v:+.5f}" if abs(v)<.1 else f"{v:+.4f}",(x,v),xytext=(0,9 if v>0 else -15),
                        textcoords="offset points",ha="center",fontsize=9,color=c,weight="bold")
        vals=[r["delta_U"] for r in known]
        ax.axhline(0,color="#657180",lw=1)
        ax.set_xticks(range(len(rr)),[str(r["seed"]) for r in rr]);ax.set_xlabel("Arrival seed")
        ax.set_xlim(-.65,len(rr)-.35)
        ax.set_ylabel("Delta U = spill - control")
        ax.grid(axis="y",alpha=.8)
        if stage=="refined":
            ax.set_yscale("symlog",linthresh=.02,linscale=1)
            ax.set_ylim(-100,.025)
            ax.set_yticks([-10,-1,-.1,-.02,0,.02])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
            text="1 / 4 positive; seed 209 UNKNOWN\nSymlog axis; severe seed 208 retained"
        else:
            lo=min(vals+[0]);hi=max(vals+[0]);span=max(hi-lo,.005)
            ax.set_ylim(lo-.25*span,hi+.43*span)
            text=f"{sum(v>0 for v in vals)} / {len(vals)} positive; mean {st.fmean(vals):+.5f}"
        ax.text(0,1.035,text,transform=ax.transAxes,fontsize=8.5,ha="left",va="bottom",color="#5c6876")
        panel(ax,"abcdef"[i],heading)
        # Raise heading above the per-facet annotation.
        ax.set_title(f"({'abcdef'[i]}) {heading}",loc="left",pad=42,fontsize=11)
        objective="5 terms; P95 weight = 0" if stage=="no_p95" else "6 terms; P95 included"
        ax.text(.98,.97,objective,transform=ax.transAxes,ha="right",va="top",fontsize=8,color="#5c6876")
    finish(fig,out,"04-spillover-results",pdf,
        "Each point compares allowing spillover with its own same-deployment / same-policy no-spill control. Green > 0, red < 0; no cross-stage pooled mean.\n"
        "Axes have different scales. Panel (d) uses symlog to retain the severe negative seed; UNKNOWN is not zero. Panel (f) is the historical five-term objective.\n"
        "These are completed historical measurements; latest prospective objective includes P95 again. No new GPU runs or retrospective reweighting.")
    write_csv(out/"04-spillover-pairs.csv",data)
    return data,five,coupled


def mechanism(out,pdf,data,five,coupled):
    request=rows("artifacts/queue-five-term-study-20261023/gpu-contention-forensics-v1/spill-requests.csv")
    metrics=[]
    for seed in (401,402):
        rr=[r for r in request if int(r["seed"])==seed]
        metrics.append(dict(seed=seed,n=len(rr),
            queue_saved_ms=st.fmean(float(r["off_worker_queue_ms"])-float(r["on_worker_queue_ms"]) for r in rr),
            service_added_ms=st.fmean(float(r["on_service_ms"])-float(r["off_matched_service_ms"]) for r in rr),
            e2e_added_ms=st.fmean(float(r["e2e_change_ms"]) for r in rr)))
    route=[]
    for r in coupled:
        if r["role"]=="coupled_spill":
            route.append(dict(seed=int(r["seed"]),S0=int(r["text_to_s0"]),S1=int(r["text_to_s1"]),G2=int(r["text_to_g2"]),terms=6))
    for r in five:
        if r["role"]=="coupled_spill":
            route.append(dict(seed=int(r["seed"]),S0=int(r["text_0_requests"]),S1=int(r["text_1_requests"]),G2=int(r["actual_text_spills"]),terms=5))
    route.sort(key=lambda r:r["seed"])
    fig=plt.figure(figsize=(16,6.8))
    title(fig,"Result 4 | Where spillover helps, and where it loses",
          "Two specialist replicas remain active; the generalist receives a small, autonomously selected share of text requests")
    gs=fig.add_gridspec(1,3,left=.065,right=.98,top=.79,bottom=.23,wspace=.35)
    ax=fig.add_subplot(gs[0,0]);x=np.arange(4);bottom=np.zeros(4)
    for key,c,label in zip(("S0","S1","G2"),(BLUE,"#8dbad7",ORANGE),("ChatQA #0","ChatQA #1","MiniCPM-o")):
        counts=np.array([r[key] for r in route]);total=np.array([r["S0"]+r["S1"]+r["G2"] for r in route])
        vals=counts/total*100
        ax.bar(x,vals,bottom=bottom,width=.57,color=c,label=label,zorder=3)
        if key!="G2":
            for i,v in enumerate(vals):ax.text(i,bottom[i]+v/2,str(counts[i]),ha="center",va="center",fontsize=9,color="white" if key=="S0" else "#254562")
        else:
            for i,r in enumerate(route):ax.text(i,102,f"G2: {r['G2']}\n({vals[i]:.2f}%)",ha="center",va="bottom",fontsize=8.5,color="#a65c13")
        bottom+=vals
    panel(ax,"a","Observed text routing")
    ax.set_xticks(x,[f"{r['seed']}\n{r['terms']}-term" for r in route]);ax.set_xlabel("Seed / frozen objective")
    ax.set_ylabel("Share of text requests (%)");ax.set_ylim(0,123);ax.set_yticks([0,25,50,75,100]);ax.grid(axis="y")
    ax.legend(frameon=False,ncol=1,fontsize=8,loc="lower left")
    ax=fig.add_subplot(gs[0,1]);x=np.arange(2);w=.24
    for j,(k,c,label) in enumerate((("queue_saved_ms",GREEN,"Queue time saved"),("service_added_ms",RED,"Service time added"),("e2e_added_ms",ORANGE,"E2E time added"))):
        vals=[r[k] for r in metrics];b=ax.bar(x+(j-1)*w,vals,w,color=c,label=label,zorder=3)
        ax.bar_label(b,labels=[f"{v:.0f}" for v in vals],padding=4,fontsize=9)
    panel(ax,"b","Matched diverted requests: mean changes")
    ax.set_xticks(x,[f"Seed {r['seed']}\nn = {r['n']} diverted" for r in metrics]);ax.set_ylabel("Milliseconds per diverted request")
    ax.set_ylim(0,1560);ax.grid(axis="y");ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax=fig.add_subplot(gs[0,2]);x=np.arange(2);pos=np.zeros(2);neg=np.zeros(2)
    rr=[r for r in data if r["stage"]=="no_p95"]
    for t,c in zip(TERMS,TERM_COLORS):
        if t=="P":continue
        vals=np.array([r["contribution_delta"][t] for r in rr]);bottom=np.where(vals>=0,pos,neg)
        ax.bar(x,vals,bottom=bottom,width=.46,color=c,label=t,zorder=3)
        pos+=np.maximum(vals,0);neg+=np.minimum(vals,0)
    ax.scatter(x,[r["delta_U"] for r in rr],marker="D",color="#233950",s=35,zorder=4,label="Total delta U")
    for i,r in enumerate(rr):ax.annotate(f"{r['delta_U']:+.5f}",(i,r["delta_U"]),xytext=(0,-19),textcoords="offset points",ha="center",fontsize=9)
    panel(ax,"c","Historical five-term utility decomposition")
    ax.axhline(0,color="#657180",lw=1);ax.set_xticks(x,["Seed 401","Seed 402"]);ax.set_xlim(-.65,1.65)
    ax.set_ylim(-.015,.004);ax.set_ylabel("Contribution to delta U (positive = gain)");ax.grid(axis="y")
    ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper left")
    finish(fig,out,"04-spillover-mechanism",pdf,
        "Panel (b): matched input / request identities across paired runs; all diverted requests retained. Queue savings are outweighed by added service time.\n"
        "Worker-queue, handler-service and E2E clocks have different boundaries; these bars are separate diagnostics, not an exact additive timing identity.\n"
        "Handler service includes more than GPU kernels. These observations do not isolate SM, L2 or DRAM interference causally.")
    write_csv(out/"04-spillover-request-mechanism.csv",metrics)
    write_csv(out/"04-spillover-routing.csv",route)
    CHECKS["mechanism"]=dict(matched_diverted_requests=len(request),seeds=[401,402],kernel_causality_claimed=False)
    return metrics,route


def write_readme(out):
    text="""# 动机实验与成果2、3、4图表（2026-10-25）

这些图由冻结原始报告/逐请求记录生成，没有新跑GPU、重拟合或改权重。PNG用于预览，PDF/SVG为矢量稿。英文图便于论文排版，本页提供中文口径。

|图|内容|核心证据与口径|
|---|---|---|
|[01 动机图](01-motivation.png)|参考图的上四下三布局；七模型、16条任务路径|专用蓝、通用橙；上图质量为原生QA EM/分类accuracy/OCR CER/ASR WER；下图OCR/ASR转换为1−错误率。全体三个通用模型保留。|
|[02 静态部署](02-static-deployment.png)|五方案独立确认U、竞争者放大图、六项损失、选中路由|七模型冻结配置池；四任务各0.0748360843 rps；H1800；seed43/44/45；六项等权，含P95。|
|[02 具体对照设计表](02-static-comparator-table.png)|每个被测方案的部署、路由和平均U|C1自由搜索与最优不分流为同一物理方案；C5为冻结纯通预算候选，不能称最优纯通。|
|[03 DFS效率](03-dfs-efficiency.png)|搜索/整进程耗时、完整预测次数、一致性|五模型G2+四专用、0/1副本、K4的CPU有限预测域；5轮串行新进程；三种算法全部同目标、同部署、同路由。|
|[04 分流结果](04-spillover-results.png)|6个阶段，19个完整配对与1个UNKNOWN|ΔU统一为允许分流减自身同驻留同策略禁分流对照；不跨阶段拼均值；205–208的严重负结果与209缺测保留。|
|[04 分流机制](04-spillover-mechanism.png)|多副本路由、排队与服务代价、五项效用分解|303/304六项与401/402历史五项分开标注。排队确实减少，但服务时长增加更大；不是kernel竞争因果证据。|

## 模型与动机图口径

完整模型ID见 `01-motivation-data.csv`。七模型为Baichuan-Omni-1d5、MiniCPM-o-4_5-GPTQ、Qwen2.5-Omni-7B、ChatQA、SigLIP2、PP-OCRv6、Granite Speech。

动机图使用当前模型池真实测量，沿用参考图结构，不沿用参考图旧模型或旧数值。QA/OCR/ASR各32个相同独立输入、图像64个；质量按唯一输入聚合，服务延迟按三轮闭环画像中的重复请求聚合，二者不是同一请求频率。显存为采样的所属GPU进程同时峰值之和（GiB），含运行时工作区，既不是权重大小，也不是整卡峰值。OCR使用项目复核参考，非外部认证gold。质量只证明该观察子集，不宣称总体精度结论；当前专用也不保证比每个通用都更快/更省显存。

## 评价与结论范围

六项效用为 `U=−(Q+T+L+P+M+E)/6`：质量损失、测量窗内完成损失、平均端到端延迟、P95端到端延迟、显存、能耗。没有绝对质量准入。401/402为历史五项实验 `U=−(Q+T+L+M+E)/5`，P95权重为0；不追溯重算为当前六项目标。Q/T/L/P/M/E为各阶段冻结归一化量，具体定义见各原始报告，不跨阶段直接比较U绝对值。

静态C1实测领先本批全部竞争者，但每方案仅3个种子，精确双侧配对符号翻转p=0.25；不能据此写显著或物理全局最优。预测有限域最优证书与GPU有限方案实测分开理解。纯通候选的子域最优未被证明。

高负载以ChatQA×2、G2×1、OCR×1为主部署：文本1.85 rps，其余各0.01，Poisson，H600、warmup30、drain上限300；图像/ASR给G2，OCR给OCR。动态分流的控制臂保留相同驻留和专用队列策略，只关闭文本到G2。固定比例轮则比较原有SW分发策略的0与1/32。所有结果均保留原稳定性判据。

算法自主分流已实现，稳定正总效用尚未证实。图4(d)使用symlog轴完整显示seed208的ΔU≈−25.1465；209缺合法能耗导致E/U UNKNOWN，且控制臂未运行，不能画为0或纳入配对均值。六阶段分别标轴，不能按柱高跨图比较。

机制图中的被分流请求按身份匹配。队列、服务、E2E计时边界不同，不强行相加。handler变慢并不等于全部由GPU kernel竞争引起；尚不能区分SM/L2/DRAM或CPU准备/同步/回收贡献。

## 文件与复现

- `all-figures.pdf`：全部6页图表矢量合集。
- 每图有PNG/PDF/SVG；CSV保留完整数值；JSON保留具体部署/路由和统一绘图数据。
- `manifest.json`：每个读取源文件的绝对路径、SHA256和输出哈希。
- `validation.json`：收据、效用算术、同输入集、DFS一致性检查结果；`03-dfs-audit.json`保留严格CPU审计结果。
- `figures-and-data.zip`：本目录图表与配套数据（不打包原始大体积GPU请求日志）。

在仓库根目录运行（输出目录必须新建）：

```bash
python scripts/plot_research_results_20261025.py --output artifacts/research-figures-20261025-v2
```

绘图脚本只读取原始实验包。manifest记录本次源文件哈希，生成完成前复查来源未改变。严格静态确认校验和CPU算法审计复用既有审计器；其余配对读取既有审计报告并校验可用的原始运行收据/算术，不声称重新验证全部GPU物理测量。
"""
    (out/"README.zh.md").write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/research-figures-20261025-v1")
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    style()
    track(__file__)
    track(ROOT/"scripts/plot_static_confirmation.py")
    track(ROOT/"scripts/plot_static_search_efficiency.py")
    with PdfPages(out/"all-figures.pdf") as pdf:
        mot=motivation(out,pdf)
        util,plans=static_result(out,pdf)
        eff=efficiency(out,pdf)
        pairs,five,coupled=spillover(out,pdf)
        mech,route=mechanism(out,pdf,pairs,five,coupled)
    dump(out/"plot-data.json",dict(motivation=mot,static_utility=util,static_plans=plans,efficiency=eff,
                                  spillover_pairs=pairs,spillover_mechanism=mech,spillover_routes=route))
    write_readme(out)
    for r in RECEIPTS.values():assert sha(r["path"])==r["sha256"],f"Source changed while exporting: {r['path']}"
    dump(out/"validation.json",dict(status="PASS",gpu_experiments_run=False,checks=CHECKS))
    dump(out/"manifest.json",dict(schema="gsdeploy.research_figures.v1",date="2026-10-25",
         input_receipts=list(RECEIPTS.values()),environment=dict(python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__),
         outputs=[dict(file=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()]))
    with zipfile.ZipFile(out/"figures-and-data.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.name!="figures-and-data.zip":z.write(p,p.name)
    print(json.dumps(dict(status="COMPLETE",output=str(out),validation=CHECKS),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
