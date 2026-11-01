"""Paper figures in the layout of the user's reference, using verified exports.

No GPU run, rescoring, fitting, or solver execution. Captions live outside the
figures. Run with a new --output directory; the evidence exports are read only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/research-figures-20261025-v2"
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
GRAY = "#777777"
TASKS = ("text", "image", "ocr", "asr")
TASK_NAMES = ("Text", "Image", "OCR", "ASR")
GENERALISTS = ("generalist_1", "generalist_2", "generalist_3")
TERMS = ("Q", "T", "L", "P", "M", "E")
TERM_COLORS = ("#5b8fb9", "#d7d7d7", "#e7ad58", "#bf725e", "#9c8cad", "#7fa390")
NAMES = {"text": "ChatQA-8B", "image": "SigLIP2", "ocr": "PP-OCRv6", "asr": "Granite-2B",
         "generalist_1": "Baichuan-Omni", "generalist_2": "MiniCPM-o-4.5", "generalist_3": "Qwen2.5-Omni"}
LAYOUT_CHECKS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelweight": "normal",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.minor.width": .4, "ytick.minor.width": .4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7,
        "legend.handlelength": 1.3, "legend.handletextpad": .4,
        "legend.borderaxespad": .3, "legend.labelspacing": .25,
        "grid.color": "#b0b0b0", "grid.linestyle": ":", "grid.linewidth": .4,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600,
    })


def panel(ax, tag, title):
    ax.set_title(f"({tag}) {title}", pad=6)


def export(fig, out, name, pdf):
    # One geometry for PNG and vector exports; no report banners or footers.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    moved = 0
    for ax in fig.axes:
        placed = []
        for text in ax.texts:
            if not text.get_visible() or not text.get_text():
                continue
            for _ in range(12):
                bounds = text.get_window_extent(renderer)
                if not any(bounds.expanded(1.04, 1.06).overlaps(other) for other in placed):
                    break
                if not isinstance(text, Annotation) or text.anncoords != "offset points":
                    raise ValueError(f"Unresolved label overlap: {name}: {text.get_text()}")
                dx, dy = text.xyann
                text.xyann = (dx, dy+2)
                moved += 1
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
            else:
                raise ValueError(f"Unresolved label overlap: {name}: {text.get_text()}")
            placed.append(text.get_window_extent(renderer))
    LAYOUT_CHECKS.append(dict(figure=name, data_label_collisions=0, label_offset_adjustments=moved))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=.035)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def motivation(data, out, pdf):
    by = {(r["slot"], r["task"]): r for r in data["motivation"]}
    fig = plt.figure(figsize=(10.6, 6.35))
    fig.suptitle("Task-wise Quality, Latency and Memory Comparison of Specialist and Generalist Models",
                 y=.998, fontsize=11)
    fig.legend(handles=[Patch(facecolor=BLUE, label="Specialist"), Patch(facecolor=ORANGE, label="Generalist")],
               loc="upper center", bbox_to_anchor=(.53, .966), ncol=2, fontsize=7.5, columnspacing=1.4)
    top_x = (.085, .330, .575, .820)
    metric = ("EM", "Acc", "CER", "WER")
    for i, task in enumerate(TASKS):
        ax = fig.add_axes([top_x[i], .535, .163, .355])
        rr = [by[s, task] for s in (task,) + GENERALISTS]
        v = [r["mean_service_ms"] for r in rr]
        ax.barh(range(4), v, height=.62, color=[BLUE] + [ORANGE]*3, edgecolor="none")
        ax.set_xscale("log")
        lo, hi = 80, max(v)*12
        ax.set_xlim(lo, hi)
        ax.set_ylim(3.55, -.55)
        ax.set_yticks(range(4), [NAMES[s] for s in (task,) + GENERALISTS], fontsize=7.2)
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Latency (ms, log scale)", labelpad=3)
        ax.grid(axis="x", which="major", alpha=.5)
        panel(ax, "abcd"[i], f"{TASK_NAMES[i]} — {metric[i]}")
        annotations = []
        for y, r in enumerate(rr):
            mem = round(r["sampled_owned_process_peak_GiB"]*1024)
            annotation = ax.annotate(f"{r['mean_service_ms']:,.1f} ms\n{metric[i]} {r['native_quality']:.3f}\n{mem:,} MiB",
                        (r["mean_service_ms"], y), xytext=(2, 0), textcoords="offset points",
                        fontsize=6.7, va="center", ha="left", linespacing=1.12)
            annotations.append(annotation)
        # Reserve only the measured text width, without an excessive empty axis.
        for _ in range(20):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if all(a.get_window_extent(renderer).x1 < ax.get_window_extent(renderer).x1-2 for a in annotations):
                break
            hi *= 1.2
            ax.set_xlim(lo, hi)
        else:
            raise ValueError("Top-row annotations overflow")
        ticks = [10**k for k in range(2, 8) if lo < 10**k < hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks[::2] if len(ticks)>4 else ticks))
    lower = [fig.add_axes([x, .085, .235, .315]) for x in (.105, .425, .745)]
    x, width = np.arange(4), .38
    for i, (key, heading, ylabel) in enumerate((
        ("higher_better_quality", "Quality comparison", "Quality score"),
        ("mean_service_ms", "Latency comparison", "Latency (ms, log scale)"),
        ("sampled_owned_process_peak_GiB", "Memory comparison", "Peak process VRAM (MiB)"),
    )):
        ax = lower[i]
        for j, (slot, color, label) in enumerate(((None, BLUE, "Specialist"), ("generalist_3", ORANGE, "Qwen2.5-Omni-7B"))):
            values = [by[slot or t, t][key] for t in TASKS]
            if i == 2:
                values = [v*1024 for v in values]
            bars = ax.bar(x+(j-.5)*width, values, width, color=color, label=label, edgecolor="none")
            labels = [f"{v:.3f}" if i==0 else f"{v:.0f}" for v in values]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=6.3)
        ax.set_xticks(x, TASK_NAMES)
        ax.set_xlim(-.56, 3.56)
        ax.set_ylabel(ylabel, labelpad=4)
        panel(ax, "efg"[i], heading)
        ax.grid(axis="y", which="major", alpha=.55)
        if i == 0:
            ax.set_ylim(0, 1.10)
            ax.set_yticks([0, .2, .4, .6, .8, 1.0])
            ax.legend(loc="upper left", fontsize=7)
        elif i == 1:
            ax.set_yscale("log"); ax.set_ylim(60, 75000)
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(0, 15500)
            ax.set_yticks([0, 4000, 8000, 12000])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    export(fig, out, "01-motivation", pdf)


def static_utility(data, out, pdf):
    by = {(r["candidate"], r["seed"]): r for r in data["static_utility"]}
    seeds, markers, colors = (43, 44, 45), ("o", "^", "s"), ("#5d5d5d", "#9a9a9a", "#c0c0c0")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.60))
    fig.subplots_adjust(left=.095, right=.985, bottom=.19, top=.85, wspace=.31)
    names = ["Selected", "+OCR replica", "Image split", "Pure-S"]
    x = np.arange(4)
    for seed, marker, color in zip(seeds, markers, colors):
        vals = [by[f"C{i+1}", seed]["U"] for i in range(4)]
        axes[0].plot(x, vals, color=color, lw=.65, marker=marker, ms=3.4, markerfacecolor="white",
                     markeredgewidth=.7, label=f"Seed {seed}")
    means = [st.fmean(by[f"C{i+1}", s]["U"] for s in seeds) for i in range(4)]
    axes[0].scatter(x, means, marker="D", s=17, facecolor=BLUE, edgecolor=BLUE, zorder=4, label="Mean")
    axes[0].set_xticks(x, names, fontsize=7)
    axes[0].set_ylabel("Measured utility, $U$")
    axes[0].set_ylim(-.345, -.298)
    axes[0].set_yticks([-.34, -.33, -.32, -.31, -.30])
    panel(axes[0], "a", "Utility of competitive deployments")
    axes[0].grid(axis="y", alpha=.6)
    means = []
    for i, candidate in enumerate(("C2", "C3", "C4", "C5")):
        gaps = [by["C1", s]["U"]-by[candidate, s]["U"] for s in seeds]
        assert all(g>0 for g in gaps)
        axes[1].plot([i,i], [min(gaps), max(gaps)], color="#b0b0b0", lw=.7)
        for k,(v, marker,color) in enumerate(zip(gaps,markers,colors)):
            axes[1].plot(i+(k-1)*.12,v,marker=marker,ms=3.4,markerfacecolor="white",markeredgecolor=color,markeredgewidth=.7)
        mean = st.fmean(gaps); means.append(mean)
        axes[1].plot(i,mean,marker="D",ms=3.6,color=BLUE)
        axes[1].annotate(f"{mean:.3f}",(i,max(gaps)),xytext=(0,5),textcoords="offset points",ha="center",fontsize=7)
    axes[1].set_yscale("log"); axes[1].set_ylim(.002, 16)
    axes[1].set_ylabel(r"Selected-plan advantage, $U_{\rm C1}-U_{\rm control}$")
    axes[1].set_xticks(x, ["+OCR replica", "Image split", "Pure-S", "Pure-G*"], fontsize=7)
    axes[1].set_xlim(-.45, 3.45)
    panel(axes[1], "b", "Paired utility differences")
    axes[1].grid(axis="y", which="major", alpha=.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.51, 1.02),
               ncol=4, columnspacing=1.5)
    export(fig, out, "02-static-utility", pdf)


def static_components(data, out, pdf):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    fig.subplots_adjust(left=.15, right=.98, bottom=.19, top=.94)
    x, bottom = np.arange(4), np.zeros(4)
    for t,c in zip(TERMS,TERM_COLORS):
        values = np.array([st.fmean(r[t+"_weighted_cost"] for r in data["static_utility"] if r["candidate"]==f"C{i+1}") for i in range(4)])
        ax.bar(x,values,bottom=bottom,color=c,width=.58,label=t,edgecolor="white",linewidth=.25)
        bottom += values
    ax.set_xticks(x,["Selected","+OCR replica","Image split","Pure-S"],fontsize=7)
    ax.set_ylim(0,.41);ax.set_yticks([0,.1,.2,.3,.4]);ax.set_ylabel(r"Mean weighted cost ($-U$)")
    ax.legend(ncol=3,loc="upper left",columnspacing=1.2,fontsize=7)
    ax.grid(axis="y",alpha=.55)
    export(fig,out,"02-static-cost-components",pdf)


def efficiency(data,out,pdf):
    records=data["efficiency"]
    algos=("milp","dfs","plain_dfs"); names=["MILP","Bounded DFS","Plain DFS"]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.55))
    fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.87,wspace=.32)
    x,w=np.arange(3),.29
    for j,(key,color,label) in enumerate((("search_wall_seconds",BLUE,"Search"),("whole_process_wall_seconds",ORANGE,"Whole process"))):
        values=[st.median(r[key] for r in records if r["algorithm"]==a) for a in algos]
        bars=axes[0].bar(x+(j-.5)*w,values,w,color=color,edgecolor="none",label=label)
        axes[0].bar_label(bars,labels=[f"{v:.3f}" if v<1 else f"{v:.2f}" for v in values],padding=3+5*j,fontsize=6.3)
        for i,a in enumerate(algos):
            vv=[r[key] for r in records if r["algorithm"]==a]
            axes[0].scatter(i+(j-.5)*w+np.linspace(-.05,.05,5),vv,s=4,color="black",zorder=4,linewidths=0)
    axes[0].set_yscale("log");axes[0].set_ylim(.1,150)
    axes[0].set_xticks(x,names);axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].legend(loc="upper left",ncol=2,fontsize=7)
    panel(axes[0],"a","Search and whole-process time")
    axes[0].grid(axis="y",which="major",alpha=.55)
    values=[st.median(r["full_prediction_calls"] for r in records if r["algorithm"]==a) for a in algos]
    bars=axes[1].bar(x,values,width=.48,color=[BLUE,"#a9b8c3","#d3d3d3"],edgecolor="black",linewidth=.35)
    axes[1].bar_label(bars,labels=[f"{int(v):,}" for v in values],padding=4,fontsize=8)
    axes[1].set_yscale("log");axes[1].set_ylim(.6,6000)
    axes[1].set_xticks(x,names);axes[1].set_ylabel("Complete predictor evaluations")
    panel(axes[1],"b","Evaluation count")
    axes[1].grid(axis="y",which="major",alpha=.55)
    export(fig,out,"03-search-efficiency",pdf)


def spillover(data,out,pdf):
    stages=("fixed_development","fixed_validation","queue_aware","refined","coupled","no_p95")
    titles=("Fixed 1/32: development","Fixed 1/32: validation","Queue-aware: independent",
            "Refined routing: independent","Coupled routing: development","Without P95: development")
    limits=[(-.10,.30),(-.18,.035),(-.14,.025),(-70,.025),(-.05,.01),(-.014,.0025)]
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.05))
    fig.subplots_adjust(left=.09,right=.985,top=.92,bottom=.115,hspace=.67,wspace=.39)
    for i,(ax,stage,name,limit) in enumerate(zip(axes.flat,stages,titles,limits)):
        rr=[r for r in data["spillover_pairs"] if r["stage"]==stage]
        for x,r in enumerate(rr):
            if r["delta_U"] is None:
                ax.text(x,.22,"NA",ha="center",transform=ax.get_xaxis_transform(),fontsize=7,color=GRAY)
                continue
            value=r["delta_U"]
            color=BLUE if value>0 else "#343434"
            ax.plot([x,x],[0,value],color=color,lw=.8)
            ax.plot(x,value,marker="o",ms=3.2,color=color)
            if stage=="refined" and r["seed"]==208:
                ax.annotate("−25.15",(x,value),xytext=(-5,-2),textcoords="offset points",ha="right",fontsize=6.8)
        ax.axhline(0,color=GRAY,lw=.6,ls="--",zorder=0)
        ax.set_xlim(-.55,len(rr)-.45);ax.set_ylim(*limit)
        ax.set_xticks(range(len(rr)),[str(r["seed"]) for r in rr],fontsize=7)
        if i%3==0:ax.set_ylabel(r"Utility gain, $\Delta U$")
        ax.set_xlabel("Seed",labelpad=2)
        ax.grid(axis="y",alpha=.5)
        panel(ax,"abcdef"[i],name)
        if stage=="refined":
            ax.set_yscale("symlog",linthresh=.02,linscale=.75)
            ax.set_yticks([-10,-1,-.1,0,.02])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{y:g}"))
        elif stage=="no_p95":
            ax.set_yticks([-.012,-.008,-.004,0])
        elif stage=="coupled":
            ax.set_yticks([-.04,-.02,0])
    export(fig,out,"04-spillover-utility",pdf)


def mechanism(data,out,pdf):
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.45))
    fig.subplots_adjust(left=.075,right=.985,top=.90,bottom=.19,wspace=.48)
    rr=data["spillover_routes"]
    x,bottom=np.arange(4),np.zeros(4)
    for key,c,label in (("S0",BLUE,"S0"),("S1","#9ec6df","S1"),("G2",ORANGE,"G")):
        vals=np.array([r[key]/(r["S0"]+r["S1"]+r["G2"])*100 for r in rr])
        axes[0].bar(x,vals,bottom=bottom,width=.62,color=c,label=label,edgecolor="none")
        if key=="G2":
            for i,v in enumerate(vals):axes[0].text(i,102,f"{v:.1f}%",ha="center",va="bottom",fontsize=6.6)
        bottom+=vals
    axes[0].set_ylim(0,130);axes[0].set_yticks([0,25,50,75,100]);axes[0].set_ylabel("Text requests (%)")
    axes[0].set_xticks(x,[str(r["seed"]) for r in rr]);axes[0].set_xlabel("Seed")
    panel(axes[0],"a","Text routing")
    axes[0].legend(loc="upper center",ncol=3,fontsize=6.2,columnspacing=1.0)
    mr=data["spillover_mechanism"];x=np.arange(2);w=.24
    colors=("#7baf9f","#bd7261",ORANGE)
    labels=("Queue saved","Service added","E2E added")
    for j,(key,c,label) in enumerate(zip(("queue_saved_ms","service_added_ms","e2e_added_ms"),colors,labels)):
        vals=[r[key] for r in mr]
        bars=axes[1].bar(x+(j-1)*w,vals,w,color=c,label=label)
        axes[1].bar_label(bars,labels=[f"{v:.0f}" for v in vals],fontsize=6.5,padding=2)
    axes[1].set_xticks(x,[str(r["seed"]) for r in mr]);axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Mean change (ms)")
    axes[1].set_ylim(0,1850);axes[1].set_yticks([0,500,1000,1500]);axes[1].grid(axis="y",alpha=.5)
    panel(axes[1],"b","Per-request delay changes")
    axes[1].legend(loc="upper left",ncol=1,fontsize=6.4,labelspacing=.0)
    pairs=[r for r in data["spillover_pairs"] if r["stage"]=="no_p95"]
    pos,neg=np.zeros(2),np.zeros(2)
    for t,c in zip(TERMS,TERM_COLORS):
        if t not in ("Q","L","E"):continue
        vals=np.array([r["contribution_delta"][t] for r in pairs]);bottom=np.where(vals>=0,pos,neg)
        axes[2].bar(x,vals,bottom=bottom,color=c,width=.48,label=t,edgecolor="none")
        pos+=np.maximum(vals,0);neg+=np.minimum(vals,0)
    axes[2].plot(x,[r["delta_U"] for r in pairs],"D",ms=3.1,color="black",label="Total")
    axes[2].axhline(0,color=GRAY,lw=.6)
    axes[2].set_ylim(-.016,.001);axes[2].set_xlim(-.6,1.6)
    axes[2].set_yticks([-.012,-.008,-.004,0]);axes[2].set_ylabel(r"Contribution to $\Delta U$")
    axes[2].set_xticks(x,[str(r["seed"]) for r in pairs]);axes[2].set_xlabel("Seed")
    panel(axes[2],"c","Utility decomposition")
    axes[2].legend(loc="lower left",ncol=2,fontsize=6.5)
    axes[2].grid(axis="y",alpha=.5)
    export(fig,out,"04-spillover-mechanism",pdf)


def write_captions(out,data):
    captions="""# 科研插图与图注

本版按参考图重排，使用 Python/Matplotlib：白底、衬线字体、细轴线、紧凑子图、直接标数；图内不放汇报式结论、大段说明或方案表。01为七子图总图；02/03/04为独立论文插图。PNG为600 dpi，PDF/SVG为矢量。数据完全继承已校验导出，不重新实验、改分数或筛除负结果。

## 图1：动机实验

![动机图](01-motivation.png)

**图注。** 通用与专用模型的任务级质量、服务延迟和GPU显存对比。上排(a–d)分别给出文本QA、图像分类、OCR及ASR在单驻留副本、batch=1下的平均服务延迟，并标注原生质量指标与显存。下排(e–g)按用户参考图选用Qwen2.5-Omni-7B作为固定通用基线，对比质量、服务延迟及显存；其余两种通用模型完整保留在上排。QA为EM、图像为accuracy；下排OCR为1−CER，ASR为1−WER，均越高越好；上排OCR/ASR标注仍为原生CER/WER，越低越好。显存为采样的所属进程同时峰值之和，以MiB表示，包含运行时工作区。QA/OCR/ASR各32个唯一输入，图像64个；质量按唯一输入聚合，延迟按三轮闭环画像中的重复请求聚合。OCR参考经项目复核，非外部认证gold。完整模型ID和精确数值见data/plot-data.json。

## 图2a：静态部署效用比较

![静态效用](02-static-utility.png)

**图注。** 四任务各0.0748360843 req/s、Poisson到达、H=1800 s、独立种子43/44/45下的GPU确认结果。(a)四个竞争方案的实测总效用，细线连接相同种子，蓝菱形为三种子均值。(b)选中方案相对四个控制的配对效用优势，采用对数纵轴；浅灰线为实测最小至最大值，不是置信区间，点形与(a)相同。均值标数。Pure-G*为冻结的纯通预算候选，其子域最优未证明，不能称最优纯通。U=−(Q+T+L+P+M+E)/6，含P95，无绝对质量门槛。C1也等于最优不分流候选。C1在本批每个种子领先每个已测对照，但n=3，精确双侧配对符号翻转p=0.25。预测有限域最优与物理全局最优不可混同。

|方案|驻留部署|路由|平均实测U|
|---|---|---|---:|
|C1 Selected|MiniCPM×1、图像专用×1、OCR专用×1|QA/ASR→MiniCPM；图像/OCR→对应专用|−0.312130|
|C2 +OCR replica|C1基础上OCR改为2副本|任务级份额同C1|−0.319643|
|C3 Image split|驻留同C1|图像1/6给MiniCPM、5/6给专用；其余同C1|−0.332513|
|C4 Pure-S|四专用各1副本|各任务给对应专用|−0.337135|
|C5 Pure-G*|MiniCPM×1、Qwen×1|QA→Qwen，图像/ASR→MiniCPM；OCR在MiniCPM/Qwen间5/6、1/6|−5.469076|

## 图2b：静态效用构成

![效用构成](02-static-cost-components.png)

**图注。** 四个竞争方案的三种子平均加权损失，六项堆叠之和等于−U。Q/T/L/P/M/E分别为质量损失、窗内完成损失、平均端到端延迟、P95端到端延迟、显存与能耗，每项权重1/6。图2a(b)仍完整包含纯通控制，不将纯通大数值压入本图相同线性坐标范围。

## 图3：搜索效率

![搜索效率](03-search-efficiency.png)

**图注。** MILP与带界DFS、普通DFS在五模型（MiniCPM＋四专用）、每模型0/1副本、路由分母1..4的冻结预测域内比较。(a)搜索及整进程墙钟时间；柱高为5次新进程运行的中位数，黑点为每次原始运行，柱上标中位数秒值。(b)完整预测次数，5轮均为1、5、1696。对数纵轴。各算法/各轮使用相同目标、轨迹、CPU affinity，串行交错顺序运行；15次全有有限域证书且最优效用、部署、路由一致，U=−0.309302874。MILP搜索中位耗时0.252850 s，带界DFS 0.371656 s，普通DFS 23.662888 s，分别为1.47×与93.58×搜索加速；整进程为1.15×与30.56×。这是CPU搜索效率，不是GPU推理加速或七模型物理全域最优证明。

## 图4a：分流总效用

![分流效用](04-spillover-utility.png)

**图注。** 各阶段分流相对自身控制的配对效用变化，ΔU=允许分流U−禁止分流U，正值为收益。主部署为ChatQA×2、MiniCPM×1、OCR×1；文本1.85 req/s，图像/OCR/ASR各0.01，H600 s、warmup30 s、drain上限300 s。动态策略保持相同驻留和专用队列策略，只关闭通用文本分流；固定1/32轮比较原有SWRR策略下0与1/32分流。各阶段独立冻结评价与策略，不跨阶段合并均值。面板纵轴尺度不同；(d)使用symlog轴（线性阈值0.02）完整保留seed208的ΔU=−25.146501；209因缺合法能耗E/U未知且控制未运行，标NA，不是0。(a–e)为含P95的六项效用，(f)为历史去P95的五项效用，两者不作绝对U横比。全部19个完整配对与1个未知配对保留，没有挑选正收益样本。算法已自主分流，但稳定正总效用尚未证实。

## 图4b：分流的收益和代价

![分流机制](04-spillover-mechanism.png)

**图注。** (a)自主策略的实际文本路由份额：两份ChatQA专用副本均承担任务，通用MiniCPM接收少量分流，柱上标通用份额。303/304为六项，401/402为历史五项实验。(b)逐被分流请求匹配输入/请求身份，给出worker排队时间节省、自身handler服务时间增加和端到端时间增加的均值；401包含31个、402包含19个分流请求，未筛除慢请求。三种时间边界不同，不强行作精确加法恒等式。(c)401/402五项效用的贡献变化；Q/L/E为非零变化项，T/M均为0，菱形为总ΔU。排队有收益，服务代价更大；handler包含准备、同步、回收等，本图不构成SM/L2/DRAM kernel干扰的因果分解。P95已在后续目标恢复，历史五项记录不重评分。

## 数据与复现

- `paper-figures.pdf`：6页矢量图合集。
- `data/`：精确数据、具体方案、原校验报告和源导出manifest。
- `source/plot_paper_figures_20261025.py`：本次绘图脚本副本；仓库脚本为正式入口。
- `manifest.json`：本次输入与所有输出的SHA256。

在原仓库运行，`--output`必须是尚不存在的新目录：

```bash
python scripts/plot_paper_figures_20261025.py --output artifacts/paper-figures-20261025-rerun
```

本次核对原导出manifest内所引用的绘图数据与校验文件哈希，再核对各图使用的效用算术、配对方向和最优一致性；没有声称重新测量GPU或重新进行kernel归因。
"""
    (out/"captions.zh.md").write_text(captions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/paper-figures-20261025-v1")
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    manifest=json.loads((source/"manifest.json").read_text())
    receipts={r["file"]:r for r in manifest["outputs"]}
    required=("plot-data.json","validation.json","03-dfs-audit.json")
    for name in required:assert sha(source/name)==receipts[name]["sha256"],name
    data=json.loads((source/"plot-data.json").read_text())
    assert json.loads((source/"validation.json").read_text())["status"]=="PASS"
    assert len(data["motivation"])==16 and len(data["static_utility"])==15
    assert len(data["efficiency"])==15 and all(r["certified"] for r in data["efficiency"])
    assert len({json.dumps([r["deployment"],r["routing"],r["optimum_U"]],sort_keys=True) for r in data["efficiency"]})==1
    for r in data["static_utility"]:
        assert abs(r["U"]+sum(r[t+"_weighted_cost"] for t in TERMS))<1e-12
    pairs=data["spillover_pairs"]
    assert len(pairs)==20 and sum(r["delta_U"] is None for r in pairs)==1
    for r in pairs:
        if r["delta_U"] is not None:assert abs(r["delta_U"]-(r["U_spill"]-r["U_control"]))<1e-12
    out.mkdir(parents=True,exist_ok=False)
    setup()
    with PdfPages(out/"paper-figures.pdf") as pdf:
        motivation(data,out,pdf)
        static_utility(data,out,pdf)
        static_components(data,out,pdf)
        efficiency(data,out,pdf)
        spillover(data,out,pdf)
        mechanism(data,out,pdf)
    write_captions(out,data)
    (out/"layout-checks.json").write_text(json.dumps(dict(status="PASS",figures=LAYOUT_CHECKS),indent=2)+"\n")
    (out/"data").mkdir();(out/"source").mkdir()
    for p in source.iterdir():
        if p.suffix in (".csv",".json"):
            if p.name in receipts:assert sha(p)==receipts[p.name]["sha256"]
            shutil.copyfile(p,out/"data"/p.name)
    shutil.copyfile(__file__,out/"source"/Path(__file__).name)
    outputs=[dict(file=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size)
             for p in sorted(out.rglob("*")) if p.is_file()]
    record=dict(status="PASS",schema="gsdeploy.paper_figures.v1",source=str(source),
                source_manifest_sha256=sha(source/"manifest.json"),script_sha256=sha(__file__),
                checks=dict(source_data_receipts=True,static_utility_arithmetic=True,search_optima_identical=True,
                            spillover_delta_arithmetic=True,measured_pairs=19,unknown_pairs=1),
                software=dict(matplotlib=matplotlib.__version__,numpy=np.__version__),outputs=outputs)
    (out/"manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    with zipfile.ZipFile(out/"paper-figures-and-data.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix!=".zip":z.write(p,p.relative_to(out))
    print(json.dumps(dict(output=str(out),status="PASS",figures=6,checks=record["checks"]),indent=2))


if __name__=="__main__":
    main()
