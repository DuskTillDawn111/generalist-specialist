"""Render the verified static-search candidate utility table as a paper table figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/paper-figures-20261025-final/data/02-static-utility-data.csv"
SEARCH = ROOT / "artifacts/static-study-gpu-20261022/search-v2-v5-K8/search-results.json"
REPORT = ROOT / "artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md"

ORDER = ("C1", "C2", "C3", "C4", "C5")
LABELS = {
    "C1": "算法选中方案",
    "C2": "最强其他部署",
    "C3": "同部署最强其他路由",
    "C4": "纯专控制",
    "C5": "纯通预算候选",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    rows = list(csv.DictReader(SOURCE.open(newline="")))
    rows = [r for r in rows if r["candidate"] in ORDER]
    assert len(rows) == 15
    by_candidate = {c: [r for r in rows if r["candidate"] == c] for c in ORDER}
    assert all(sorted(r["seed"] for r in rs) == ["43", "44", "45"] for rs in by_candidate.values())
    data = {}
    for candidate in ORDER:
        seed_values = {int(r["seed"]): float(r["U"]) for r in by_candidate[candidate]}
        data[candidate] = {**seed_values, "mean": sum(seed_values.values()) / 3.0}
    return data


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render(data, output: Path):
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    cjk_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    assert Path(cjk).is_file() and Path(cjk_bold).is_file()
    regular_prop = font_manager.FontProperties(fname=cjk)
    bold_prop = font_manager.FontProperties(fname=cjk_bold)
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(12.8, 4.9), dpi=240)
    fig.patch.set_facecolor("#171717")
    ax.set_facecolor("#171717")
    ax.axis("off")

    columns = ["方案", "seed43", "seed44", "seed45", "均值"]
    body = [[LABELS[c], fmt(data[c][43]), fmt(data[c][44]), fmt(data[c][45]), fmt(data[c]["mean"])] for c in ORDER]
    table = Table(ax, bbox=[0.025, 0.16, 0.95, 0.77])
    widths = [0.33, 0.1675, 0.1675, 0.1675, 0.1675]
    header_h, row_h = 0.14, 0.172
    edge = "#3d3d3d"
    header_color = "#202020"
    cell_color = "#171717"
    selected_color = "#1d2c3c"
    white = "#f2f2f2"
    muted = "#bdbdbd"

    for j, (label, width) in enumerate(zip(columns, widths)):
        cell = table.add_cell(0, j, width, header_h, text=label, loc="center",
                              facecolor=header_color, edgecolor=edge)
        cell.set_linewidth(0.7)
        cell.get_text().set_color(white)
        cell.get_text().set_fontsize(13)
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontproperties(bold_prop)
    for i, row in enumerate(body, start=1):
        for j, (value, width) in enumerate(zip(row, widths)):
            face = selected_color if i == 1 else cell_color
            align = "left" if j == 0 else "center"
            cell = table.add_cell(i, j, width, row_h, text=value, loc=align,
                                  facecolor=face, edgecolor=edge)
            cell.set_linewidth(0.65)
            cell.PAD = 0.02
            cell.get_text().set_color(white if j == 0 or i == 1 else muted)
            cell.get_text().set_fontsize(12.2)
            cell.get_text().set_fontproperties(bold_prop if i == 1 and j == 0 else regular_prop)
            if j == 0:
                cell.get_text().set_fontweight("bold" if i == 1 else "normal")
    ax.add_table(table)
    ax.text(0.025, 0.965, "静态任务负载下的部署组合总效用比较", transform=ax.transAxes,
            color=white, fontsize=17, fontproperties=bold_prop, ha="left", va="top")
    ax.text(0.025, 0.095,
            "U = −(Q + T + L + P + M + E) / 6；数值越大越好。每列使用相同到达轨迹和评价契约。",
            transform=ax.transAxes, color="#c8c8c8", fontsize=9.7, fontproperties=regular_prop,
            ha="left", va="center")
    ax.text(0.025, 0.045,
            "证据范围：冻结模型池、离散副本域与声明路由域内的有限搜索；不等同于连续物理全局最优。",
            transform=ax.transAxes, color="#8f8f8f", fontsize=8.8, fontproperties=regular_prop,
            ha="left", va="center")
    fig.subplots_adjust(0, 0, 1, 1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"global-optimality-table.{ext}", dpi=600, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main(output: Path):
    data = load_rows()
    render(data, output)
    table_rows = []
    for c in ORDER:
        table_rows.append({"candidate": c, "label": LABELS[c], "seed43": data[c][43],
                           "seed44": data[c][44], "seed45": data[c][45], "mean": data[c]["mean"]})
    (output / "table-data.json").write_text(json.dumps({"rows": table_rows}, ensure_ascii=False, indent=2) + "\n")
    (output / "README.zh.md").write_text(
        "# 静态部署组合效用表\n\n"
        "本图按已校验的 15 条 GPU 确认记录生成：五个物理候选、三个独立 seed。\n\n"
        "‘算法选中方案’来自冻结搜索；其余四行是搜索后登记的最强其他部署、同部署最强其他路由、纯专控制和纯通预算候选。\n\n"
        "图中的比较支持有限搜索域内的候选排名。搜索证书、上界和 DFS 一致性见静态研究汇总；不把有限预测域证书表述为整个连续 GPU 物理空间的全局最优。\n"
    )
    manifest = {
        "source_csv": str(SOURCE.resolve()),
        "source_csv_sha256": digest(SOURCE),
        "search_results": str(SEARCH.resolve()),
        "search_results_sha256": digest(SEARCH),
        "report": str(REPORT.resolve()),
        "report_sha256": digest(REPORT),
        "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "script_sha256": digest(Path(__file__).resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(table_rows), "source_sha256": manifest["source_csv_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
