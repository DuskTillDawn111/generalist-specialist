"""Report prediction scopes separately; never compare fixture U to GPU U."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    all_rows, paragraphs = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, label, directory in zip(axes,("New stage FIFO: CPU fixture","Legacy replay: five models"),
                                     (args.paper.resolve(),args.legacy.resolve())):
        report = json.loads((directory/"report.json").read_text())
        rows = report["rows"]
        groups = [[r for r in rows if r["algorithm"] == method]
                  for method in ("joint","joint+de","joint+local")]
        medians = [statistics.median(r["total_seconds"] for r in group) for group in groups]
        ax.bar([0,1,2],medians)
        ax.set(xticks=[0,1,2],xticklabels=["Joint","Joint + DE","Joint + local"],
               ylabel="Total seconds (seed + refinement)",title=label)
        table = ["| 算法 | 总时间范围 s | 最佳 U | 预测 bundle 数 | MILP 次数 | 精修路由数 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for group in groups:
            first=group[0]
            values=[r["utility"] for r in group if r["utility"] is not None]
            table.append(f"| {first['algorithm']} | {min(r['total_seconds'] for r in group):.3f}–{max(r['total_seconds'] for r in group):.3f} | {max(values):.9f} | {min(r['prediction_bundles'] for r in group)}–{max(r['prediction_bundles'] for r in group)} | {min(r['milp_solves'] for r in group)}–{max(r['milp_solves'] for r in group)} | {min(r['route_evaluations'] for r in group)}–{max(r['route_evaluations'] for r in group)} |")
        paragraphs.append("\n".join(table))
        for row in rows:
            all_rows.append({"prediction_mode":report["prediction_mode"],
                             **{k:v for k,v in row.items() if k!="engine_stats"}})
    fig.tight_layout()
    fig.savefig(out/"runtime.png",dpi=160)
    plt.close(fig)
    with (out/"comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    diag=json.loads((args.paper/"supplied-deployment-diagnostic.json").read_text())
    best=diag["result"]["winner"]
    initial=diag["history"][0]
    text=f"""# Joint MILP＋固定部署连续路由的重新评估

2026-11-02。本轮先补齐连续精修到 `paper_profiled_fifo_v1` 的接线，完成有界软件实验；真实画像训练与独立精度验证按用户要求暂时搁置，仍保留为未完成事项。

## 结论

**需要重评，旧效用与最优路由不能自动继承到新预测器。** 原有连续精修依赖的 SearchSession 只接受历史校准模式，之前主 MILP 接通新模式尚未覆盖该组合链路。本轮修复这个具体缺口，原 JointMilpOptimizer 和 UtilityOptimizer 的算法源码没有更改，六项效用和硬约束保持原定义。

目前新模式只具备 CPU 手算画像，因此完成的是软件机制重评，尚无真实域的新预测器优劣结论。另重跑旧预测器五模型小对照以核对接线兼容性；两组的模型、画像、轨迹和参考值不同，U 数值不能跨组比较。

## 接线与边界

- SearchSession 对新模式执行显式 schema/画像合同检查，仍保留 CPU fixture opt-in 和未知证据处理；没有继承旧校准资格。
- 连续 oracle 的版本摘要现在包括 paper_forecast、完整 forecasting 模块和资源合同。阶段模拟器变化会使 oracle 失效，不再只跟踪旧标量预测路径。
- Joint 初始解、完整初始预测、固定部署及连续比例热启动保留。精修直接评价固定 D，零比例仍驻留，没有部署邻域或反复部署 MILP。
- 结果增加预测模式、固定部署路由维度和新事件引擎真实调用计数。预测 bundle 包括 nominal/1.25/2 三个情景查询；缺失压力画像仍 UNKNOWN，名义搜索不因此被错误升级为 robust。
- 全部初始求解、会话建立及精修时间计入总时间。每条流水线限 4 秒、32 个路由点（新软件样例）或24点（旧预测对照）；运行中一次原子评价可能略超截止时间，实际用时如实计入。

## 新阶段共驻/FIFO：软件样例

1 任务、2 模型、3 个非空部署、batch1、K=2、同一条2请求冻结轨迹。DE与局部搜索种子11/29/47；所有方法采用同一新模式和相同画像。

{paragraphs[0]}

Joint 选 G-only，固定部署的路由维度为0。两种精修都复用了初始预测，未增加完整预测或 MILP 调用，最终 U 一致。这不能用于判定 DE 是否有效，因为部署内没有第二个可接收流量的模型。不同短运行的耗时波动不能解释为算法加速。

为检查非零路由维度，单独固定已支持的 G+S 部署做32点 DE 诊断，覆盖端点和非旧网格比例，不改变画像或效用。该诊断**不是 Joint 选出的赢家，也未纳入主表**：

- 实际事件模拟 {diag['engine_stats']['event_engine_calls']} 次，部署 MILP 0次；达到路由评价上限，未证明连续全域最优。
- 起点 U={initial['best_feasible_utility']:.9f}，找到的最好 U={best['best_feasible_utility']:.9f}。
- 最好连续比例为 `{json.dumps(best['routing_shares'],ensure_ascii=False)}`；未重新量化到 K=2。
- 仍低于 Joint 的 G-only U=-3.015777778，不能称为系统改进。
- 该两请求软件例中，连续质量项/资源期望可能随比例变化，而有限 FIFO 分配不变；不能把同映射下的 U 差自动归因于 GPU 干扰改善。

## 旧预测器：五模型兼容对照

模型 G2、Text、image、OCR、ASR，合法状态均为0/1副本，K=8，原四任务负载、1800秒冻结轨迹seed42、原输入序列/画像/效用。这里只调整模型池和副本域，原预测门禁仍开启；预测来自 calibrated_replay_v2，不是新 GPU 测量。

{paragraphs[1]}

Joint 赢家仍为 G2+image+OCR：ASR/Text给G2，image/OCR全给对应S。固定部署有2个路由自由度；3个搜索种子的 DE 和局部精修均未超过初始 U=-0.309302874。DE实际评价20–24点，局部17点，精修阶段新增 MILP 都为0。DE包含时间或路由预算停止，不能据此证明连续域没有更优解。

以seed29为例，DE评价了15个旧网格之外的点，最好 U=-0.363939879；局部评价14个网格之外的点，最好 U=-0.309302875，略低于初始方案。未发生收益被覆盖的情况；赢家始终保留。报告原始bundle而非与历史可能不同口径的“预测调用”直接相除：一次bundle含3个scenario查询，不能将其与名义事件模拟次数混算。

![两组各自的运行时间；两组效用不跨域比较]({out/'runtime.png'})

## 未完成事项与恢复条件

真实实测画像补齐和验证暂时搁置，不会被本轮软件通过覆盖。跟踪项：匹配执行配置的阶段/完整活动上下文进度、handler/receipt/READY 生命周期、整板功率与全驻留显存规律；当前只有接线与软件样例，仍缺实际可转移预测规律。此前346条实测请求复核及缺项保留在 `artifacts/paper-measured-pilot-20261102/REPORT.md`。

新模式具备这些真实画像后，可用本轮新增 runner 原样重评 Joint、Joint+DE、Joint+local，逐次检查相同轨迹上的 U、实际 FIFO 映射、连续比例与端点、完整预测次数、求解时间及种子稳定性。**在此之前不能将旧五模型的结论改称新预测器结论。**

## 原始文件与复现

- 新模式：`{args.paper.resolve()}`，包含Joint结果、6条完整精修流水线、路由映射和单独固定部署诊断。
- 旧对照：`{args.legacy.resolve()}`，相同结构；部署邻域关闭。
- 本目录comparison.csv、runtime.png为汇总。每组effective-config.json、plan.json和source-manifest.json保留有效域、预算与源码身份。

```bash
python experiments/reevaluate_joint_refinement.py --output /tmp/joint-paper-new \\
  --diagnostic-deployment '{{"g":1,"s":1}}'
python experiments/reevaluate_joint_refinement.py \\
  --config artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json \\
  --models generalist_2 text image ocr asr --replicas 1 --routes 24 \\
  --output /tmp/joint-legacy-new
```

没有运行 GPU、补采、拟合、部署邻域、GP-BO 或 MADS。
"""
    (out/"REPORT.md").write_text(text)
    print(out/"REPORT.md")


if __name__=="__main__":
    main()
