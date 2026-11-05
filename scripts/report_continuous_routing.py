"""Summarize actual Phase 1 records without selecting algorithms on heldout U."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def generate(source, inner, output, additional=None):
    source, inner, output = Path(source), Path(inner), Path(output)
    if output.exists():
        raise FileExistsError(output)
    main = json.loads((source/"summary.json").read_text())
    costs = json.loads((inner/"summary.json").read_text())
    records = main["records"]
    lines = ["# 连续路由 Phase 1 实际结果", "",
        "全部数值来自 CPU synthetic fixture；不证明真实 GPU 预测精度、质量或部署收益。", "",
        "## 端到端配对结果", "",
        "同一负载、训练轨迹、效用与资源约束；三个优化 seed。以下时间为各算法运行时间中位数，U 为三 seed 中位数。", "",
        "| 负载 | 算法 | 中位 U | 中位秒数 | 中位完整预测 | 有限网格认证次数 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for workload in sorted({r["workload"] for r in records}):
        for algorithm in sorted({r["algorithm"] for r in records}):
            selected = [r for r in records if r["workload"]==workload and r["algorithm"]==algorithm]
            values = [r["utility"] for r in selected if r["utility"] is not None]
            utility = f"{statistics.median(values):.9f}" if values else "UNKNOWN"
            certified = sum(json.loads((source/r["raw_file"]).read_text()).get("finite_grid_certified",False) for r in selected)
            lines.append(f"| {workload} | {algorithm} | {utility} | {statistics.median(r['elapsed_seconds'] for r in selected):.3f} | {statistics.median(r['full_prediction_calls'] for r in selected):g} | {certified}/{len(selected)} |")
    by_inner = {name:[r for r in costs["rows"] if r["inner"]==name] for name in ("enumeration","milp")}
    ratios = [m["seconds"]/e["seconds"] for e,m in zip(by_inner["enumeration"],by_inner["milp"])]
    same_calls = all(e["predictions"]==m["predictions"] for e,m in zip(by_inner["enumeration"],by_inner["milp"]))
    lines += ["", "## 固定路由内层", "",
        f"独立枚举与 MILP 的 {costs['pairs']} 对固定路由全部同值、同认证。MILP/枚举耗时比中位数 {statistics.median(ratios):.3f}；完整预测次数全部相同：{same_calls}。", "",
        f"局部连续扰动诊断中，{costs['plateau_assignment_classes']} 类实际请求分配对应 {costs['plateau_utility_classes']} 类效用。相同请求映射不能直接作为全预测缓存等价证明。", "",
        "## heldout 与停止状态", ""]
    stop_counts = {}
    for r in records:
        stop_counts[r["status"]]=stop_counts.get(r["status"],0)+1
    lines.append("实际终止状态：`"+json.dumps(stop_counts,ensure_ascii=False)+"`。")
    lines += ["", "每个训练赢家在两条 validation 与两条 test 轨迹上冻结 D/r 后评价；完整数值和独立成本位于对应 raw JSON 的 holdout 字段，未用于挑选赢家。", "",
        "旧 Joint MILP 的 wall-time 曲线只展示终点；原实现没有每次改进的时间戳，未补造过程时间。DFS state budget 是更严格的额外限制，需同时阅读原始停止原因。", "",
        "## 阶段判断", "",
        "1. 两种内层的正确性链已建立；一般黑盒下，MILP 的身份反馈并不自动减少预测调用。",
        "2. 连续搜索结果和网格结果按实际预算报告；高于某个网格结果也不表示连续全局最优。",
        "3. 保留枚举作为当前小域推荐内层，MILP 作为可运行的对照及未来安全界入口。",
        "4. 暂不推进 BO/MADS 或混合算法；先确认完整真实 oracle 的支持域，以及能够降低内层成本的有效结构。", "",
        f"主对比总 wall time（含循环、验证与保存）：{main['total_wall_seconds']:.3f} 秒。主实现摘要：`{main['implementation_sha256']}`。", "",
        f"原始结果：`{source.resolve()}`；固定路由/平台诊断：`{inner.resolve()}`。"]
    if additional:
        lines += ["", "## 额外多轨迹与维度检查", ""]
        for path in additional:
            path = Path(path)
            summary = json.loads((path/"summary.json").read_text())
            lines.append(f"`{path.resolve()}`：{len(summary['records'])} 次运行，总 wall {summary['total_wall_seconds']:.3f} 秒。")
            lines.append("")
            for row in summary["records"]:
                lines.append(f"- {row['workload']} / {row['algorithm']} / seed {row['seed']}：U={row['utility']}，calls={row['full_prediction_calls']}，status={row['status']}。")
            lines.append("")
    output.write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--inner",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--additional",nargs="*")
    args=parser.parse_args()
    generate(args.source,args.inner,args.output,args.additional)
