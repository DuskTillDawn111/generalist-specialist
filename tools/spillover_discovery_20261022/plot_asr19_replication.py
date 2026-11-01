#!/usr/bin/env python3
"""Render only final ASR1.9 raw observations bound by the v3 audit receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_json(path, expected):
    actual = receipt(path)
    if actual["sha256"] != expected:
        raise ValueError(f"Receipt mismatch: {path}")
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.audit / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outputs = {r["path"]: r["sha256"] for r in manifest["outputs"]}
    inputs = {r["path"]: r["sha256"] for r in manifest["inputs"]}
    audit_path = (args.audit / "audit.json").resolve()
    audit = checked_json(audit_path, outputs[str(audit_path)])
    if audit["status"] != "SEVEN_CONDITIONS_EIGHT_ATTEMPTS_AUDITED":
        raise ValueError("Expected complete v3 audit")
    conditions = audit["conditions"]
    if len(conditions) != 7 or len({c["id"] for c in conditions}) != 7:
        raise ValueError("Expected seven distinct selected conditions")
    raws = {}
    raw_receipts = []
    for c in conditions:
        source = c["source"]
        if inputs.get(source["path"]) != source["sha256"]:
            raise ValueError(f"Raw not bound by audit manifest: {c['id']}")
        raws[c["id"]] = checked_json(source["path"], source["sha256"])
        raw_receipts.append({"path": source["path"], "sha256": source["sha256"]})
        if c["validity"] != (c["U"] is not None):
            raise ValueError("Utility validity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), sharex=True, sharey=True)
    curve_rows = []
    notes = {
        51: "Both eligible: recoverable congestion\nDelta U (split - no split) = +0.000129919518",
        52: "No split: failed, 229 incomplete, U = null\nSplit: completed but unstable, U = null",
        53: "Both eligible: recoverable congestion\nSplit inferior: Delta U = -14.2812245889",
    }
    for ax, seed in zip(axes, (51, 52, 53)):
        for split, color, marker in ((False, "#2367a3", "o"), (True, "#d65d0e", "s")):
            cid = f"seed{seed}_n2_{'split' if split else 'nosplit'}"
            evidence = raws[cid]["evaluation"]["stability_evidence"]
            bins = evidence["backlog_timeline"]
            if len(bins) != 20 or [b["offset_seconds"] for b in bins] != list(range(30, 601, 30)):
                raise ValueError(f"Unexpected bins: {cid}")
            x = [b["offset_seconds"] for b in bins]
            y = [b["mean_backlog_requests"] for b in bins]
            ax.plot(x, y, color=color, marker=marker, markersize=3.2,
                    linewidth=1.8, label="Split 12.5% to G2" if split else "No split")
            curve_rows.extend(dict(condition_id=cid, seed=seed, split=split, **b) for b in bins)
        ax.set_title(f"Seed {seed}", weight="bold")
        ax.set_xlabel("Measurement time (s; 30 s bin end)")
        ax.set_xlim(20, 610)
        ax.set_xticks((120, 240, 360, 480, 600))
        ax.grid(alpha=0.2)
        ax.text(0, -0.24, notes[seed], transform=ax.transAxes, fontsize=9,
                va="top", linespacing=1.6)
    axes[0].set_ylim(0, max(r["mean_backlog_requests"] for r in curve_rows) * 1.08)
    axes[0].set_ylabel("Mean submitted backlog (requests)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("ASR 1.9 requests/s: two-replica routing comparison", fontsize=15, weight="bold")
    fig.text(0.5, 0.025,
             "B(t) = actual host send-start submissions - successful completions. Final 600 s raw only; drain excluded from curves.\n"
             "Development evidence; seed 51 informed selection. Invalid U stays null. No full-pool optimality claim.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.33, wspace=0.12)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"submitted-backlog-seeds51-53.{suffix}", dpi=180)
    plt.close(fig)

    table_rows = []
    for c in conditions:
        table_rows.append(dict(condition_id=c["id"], seed=c["seed"], replicas=c["replicas"],
                               specialist_share=0.875 if c["split"] else 1.0,
                               validity=c["validity"], U=c["U"],
                               stability=c["stability"]["stability_state_v2"],
                               raw_status=raws[c["id"]]["status"],
                               incomplete_requests=raws[c["id"]]["evaluation"]["stability_evidence"]["incomplete_requests"]))
    for name, rows in (("submitted-backlog-20bins.csv", curve_rows), ("seven-condition-utilities.csv", table_rows)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: "null" if v is None else v for k, v in row.items()})
    fig, ax = plt.subplots(figsize=(15.4, 4.2))
    ax.axis("off")
    headers = ["Condition", "ASR replicas", "Specialist share", "Eligible", "U (unrounded)", "Stability"]
    cells = [[r["condition_id"], r["replicas"], r["specialist_share"], r["validity"],
              "null" if r["U"] is None else repr(r["U"]), r["stability"]] for r in table_rows]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center",
                     colWidths=[0.19, 0.10, 0.12, 0.07, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8ced4")
        if row == 0:
            cell.set_facecolor("#e7edf4")
            cell.set_text_props(weight="bold")
        elif not table_rows[row - 1]["validity"]:
            cell.set_facecolor("#fff0e6")
    ax.set_title("ASR 1.9: all seven selected conditions", fontsize=15, weight="bold", pad=16)
    fig.text(0.5, 0.045, "Invalid utility is null; no imputation. Seed 53 no-split uses the predeclared user-pause retry (attempt 2).\n"
             "The interrupted attempt 1 remains in the v3 audit; table contains selected conditions only.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.16)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seven-condition-utilities.{suffix}", dpi=180)
    plt.close(fig)

    split53 = raws["seed53_n2_split"]
    evidence = split53["evaluation"]["stability_evidence"]
    write_json(args.output / "seed53-split-stability.json", dict(
        audit_condition_id="seed53_n2_split", source=next(c["source"] for c in conditions if c["id"] == "seed53_n2_split"),
        stability=split53["evaluation"]["stability"],
        evidence={k: evidence[k] for k in ("tail_backlog_slope", "tail_backlog_growth", "tail_increasing_fraction",
                  "completion_ratio", "incomplete_requests", "end_backlog_requests", "final_backlog_after_drain",
                  "drain_duration", "drain_timeout", "backlog_definition", "tail_slope_definition")},
        offered_end_backlog=evidence["offered_backlog_diagnostic"]["end_backlog_requests"],
        submitted_tail_bins=evidence["backlog_timeline"][-5:],
        interpretation="Eligible recoverable_congested under unchanged rules; substantially inferior utility, not evidence of sustained-load stability."))
    lines = ["# ASR1.9 开发重复：可视化与有效性", "",
             "所有曲线与数值均来自 v3 审计 SHA 绑定的最终原始文件，未访问运行中数据。曲线为 600 秒窗口内 20 个 30 秒 bin 的 submitted backlog 均值，不含 drain。", "",
             "![提交积压](submitted-backlog-seeds51-53.png)", "",
             "| 条件 | ASR副本 | 专用比例 | 有效 | U（完整精度） | 稳定性 |", "|---|---:|---:|---|---:|---|"]
    for r in table_rows:
        lines.append(f"| {r['condition_id']} | {r['replicas']} | {r['specialist_share']} | {r['validity']} | {repr(r['U']) if r['U'] is not None else 'null'} | {r['stability']} |")
    lines += ["", "seed52 不分流失败且有 229 条未完成；分流虽全部完成，但窗口末段持续增长，按原规则为 unstable，两个 U 均保持 null。", "",
              "seed53 分流按原规则为 recoverable_congested：末五个 bin 的 submitted backlog 均值为 170.276、172.795、160.511、142.317、128.046；末段斜率为 -0.383129 请求/秒，增长 -42.2301，上升比例 0.25。窗口末 submitted backlog 118、offered backlog 163，197.2286 秒 drain 后清零，完成率 1.0。分类有效不代表高效用，也不构成无限持续到达下稳定的证明。其较差 U 原样保留。", "",
              "有效配对仅 seed51、53，ΔU 分别为 +0.0001299195183767088、-14.281224588910186，均值 -7.1405473346959045；95% t 区间 [-97.87145448352634,83.59035981413452]，双侧符号翻转 p=1。seed51 已用于开发选择，不能宣称分流可靠增益或最终确认。", "",
              "n3 仅 seed51 结构对照，U=-0.6901186746384068，相对 n2 不分流 ΔU=-0.04051554499352339；不能外推全池最优或算法效率。用户暂停的旧 attempt 保留在 v3 审计中。"]
    (args.output / "report.zh.md").write_text("\n".join(lines) + "\n")
    generated = [receipt(p) for p in sorted(args.output.iterdir()) if p.is_file()]
    write_json(args.output / "manifest.json", dict(schema="gsdeploy.asr19_replication_figures.v1",
        inputs=[receipt(manifest_path), receipt(audit_path), *raw_receipts], code=[receipt(__file__)],
        outputs=generated, selected_condition_count=7, plotted_condition_count=6,
        frozen_inputs_unchanged=True, gpu_started=False))
    print(json.dumps({"status": "FIGURES_COMPLETE", "output": str(args.output.resolve()), "gpu_started": False}))


if __name__ == "__main__":
    main()
