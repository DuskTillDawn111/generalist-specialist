"""Render the bounded measured-data diagnostic, with missing results explicit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    out = args.input.resolve()
    r = json.loads((out / "report.json").read_text())
    a, b = r["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = [0, 1]
    axes[0].bar([v-.17 for v in x], [c["predicted_U"] for c in (a, b)],
                width=.34, label="Old predictor")
    axes[0].bar([v+.17 for v in x], [c["measured_U"] for c in (a, b)],
                width=.34, label="Native measurement")
    axes[0].set(title="Utility: held-out conditions", ylabel="U (higher is better)")
    axes[0].legend(fontsize=8)
    axes[1].bar([v-.17 for v in x], [100*c["mean_relative_error"] for c in (a,b)],
                width=.34, label="Normalized mean")
    axes[1].bar([v+.17 for v in x], [100*c["p95_relative_error"] for c in (a,b)],
                width=.34, label="Normalized P95")
    axes[1].set(title="Old predictor error", ylabel="Absolute relative error (%)")
    axes[1].legend(fontsize=8)
    cross = r["crossed_routes"]
    axes[2].bar(x, [c["A_minus_B_U"] for c in cross])
    axes[2].set(title="Same-trace route comparison\nOLD PREDICTIONS ONLY", ylabel="Route A U - Route B U")
    for ax in axes[:2]:
        ax.set_xticks(x, ["A / seed46", "B / seed47"])
    axes[2].set_xticks(x, ["seed46", "seed47"])
    fig.suptitle("One deployment; new paper predictor NOT evaluated", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "baseline-comparison.png", dpi=160)
    plt.close(fig)
    training = r["legacy_training_inventory"]
    requests = sum(c["coverage"]["requests"] for c in training)
    text = f"""# 论文预测器的最小实测复核（2026-11-02）

本轮完成了旧基线的小范围实测复核和新接口的数据适用性检查；**新预测器独立精度验证尚未完成，未运行优化器效用比较**。没有新增 GPU 测量、拟合或更改主优化器/效用/硬约束。

## 范围与复现

固定部署为 generalist_2、image、ocr 各一个副本，原四任务各 0.0748360842705697 QPS。A：image 给 S 75%、OCR 给 S 100%；B：image 给 S 100%、OCR 给 S 75%。两次原生实验均 600 秒，seed 分别为 46、47。

```bash
python experiments/validate_paper_measured.py --output /tmp/paper-measured-new
python scripts/report_paper_measured.py --input /tmp/paper-measured-new
python -m unittest tests.test_paper_measured_validation -v
```

原始采集无需重新运行，脚本只读取明确绑定的 10 份旧 run 和 2 份较新采集审查。源文件指纹见 source-manifest.json。旧验证门禁保持开启。预测只接收公开轨迹/输入序列/原画像，留出请求耗时只用于事后误差计算，不进入预测或拟合。

## 旧预测器对实测

| 条件 | 请求数 | 旧预测 U | 实测 U | U 绝对误差 | Handler 相对误差 | Mean 相对误差 | P95 相对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | {a['offered_requests']} | {a['predicted_U']:.9f} | {a['measured_U']:.9f} | {a['utility_absolute_error']:.6f} | {a['handler_relative_error']:.2%} | {a['mean_relative_error']:.2%} | {a['p95_relative_error']:.2%} |
| B | {b['offered_requests']} | {b['predicted_U']:.9f} | {b['measured_U']:.9f} | {b['utility_absolute_error']:.6f} | {b['handler_relative_error']:.2%} | {b['mean_relative_error']:.2%} | {b['p95_relative_error']:.2%} |

Mean/P95 误差使用原目标的任务权重与参考值归一化；不是合并所有请求后的单一 P95。Handler 沿用历史核验口径 worker_latency_ms，包含 handler 内部开销，不能当成纯 GPU 执行时间或分段服务律。实测六项 U 用原 Q/T/M/E 和逐请求重算 Mean/P95 复核，未重新采样峰值或功率。

346 条请求均逐条核对 arrival_index、task、pool_id、实际 alias 和计划到达时刻，无丢弃、截短或改变分母。原独立标量 FIFO 的逐请求复算与正式 predict 的各任务 Mean/P95 一致。逐条记录在 requests.csv，各任务服务/等待/Mean/P95 在 per-task.csv。实测 worker queue wait 包含运行时传输/队列口径，不能把 E2E 减去它解释成纯 GPU 执行。

A/B 的实测 U 差为 {r['observed_condition_delta_U']:.9f}，但 seed、输入顺序和到达轨迹不同，**不能把这个差值直接当成路由的因果增益**。为分开这个因素，额外只做两次预测，把 A/B 路由分别放到同一条 seed46 和 seed47 轨迹：

| 冻结轨迹 | A 路由 U | B 路由 U | A−B |
|---|---:|---:|---:|
| seed46 | {cross[0]['route_A_U']:.9f} | {cross[0]['route_B_U']:.9f} | {cross[0]['A_minus_B_U']:.9f} |
| seed47 | {cross[1]['route_A_U']:.9f} | {cross[1]['route_B_U']:.9f} | {cross[1]['A_minus_B_U']:.9f} |

这两条轨迹下旧预测均偏好 A，但差值显著随轨迹改变。交叉组合没有相应实测，不能授予新路由或新预测器验证资格。

![旧预测与实测对比]({out / 'baseline-comparison.png'})

## 为什么尚不能验证新的阶段共驻/FIFO 模式

1. 两份留出 run 的 346 条请求都有 handler 起止与 parent 接收时间；ordered segments 和 explicit worker READY 均为 0。handler end、parent receipt 和下一条 handler start 均不能替代 READY。
2. 原 {len(training)} 条训练条件共 {requests} 条 measurement 请求，没有目标 G2+image+OCR 三模型部署，训练记录也不具备上述阶段/READY 字段。原有全局系数及整板资源回归不能冒充新模式的活动上下文定律。
3. 本轮只额外检查 public181 的前两份真实训练采集审查。它们各有 181 条 measurement 和 18 条 priming，存在更细协议及 READY 观测，但绑定 text 双 owner、G2、OCR 的另一执行配置，且保留非空 prior history。两份审查的独立预测精度、主队列 provider 和最终收益资格均为 false；当前模式仅支持 fresh/drained 初态，不能直接粘接这些 busy-state 记录。
4. `ProfileProvider` 当前按精确输入 profile/segment 和完整活动 alias 集合查进度、功率和显存。这验证了机制接线，却尚未提供能从少量实测推到未见活动组合的规律。**数据适配器和事件引擎接通不等于可用的动态干扰模型已经训练完成**。单纯多补两条总耗时，也不能解决可辨识性与未见上下文覆盖。
5. 当前只读 GPU 查询结果：`{r['hardware_access'].get('error', r['hardware_access'])}`。本轮没有更改设备权限或启动补采。

因此新预测器状态为 NOT_RUN_MISSING_MATCHED_TRAINING_PROFILES，U=null；不是已运行精度失败，也不是验证通过。没有把缺字段填零、将 CPU 样例改标为 measured，或读取留出标签补画像。

## 下一步的具体边界

先做训练数据到预测规律的连接，继续限定一个部署和两种路由。优先从已有细粒度训练采集导出可用的 segment/生命周期观测，逐项保留执行配置、输入身份、完整前史和时钟证据；复用实际已有训练实现拟合可辨识的进度规律，明确哪些活动上下文仍未知。需要处理该数据域的完整初态，或另选有真实 drained 证据的记录，不能强制清空其队列后验证。

首次验收只要求：**一份实际训练数据导出的非 fixture provider 能对两条完整、独立留出轨迹给出有根据的预测**。缺阶段/READY/资源定律时先列出准确缺口；无法辨识时停止，不启动全池测量。若必须补采，先限定 1 个部署、2 个训练条件；覆盖检查通过后再执行 2 个同轨迹路由留出条件。这个上限是一次诊断预算，不保证少量样本足以支撑任意输入/重叠状态；旧 346 条留出已经被阅读，不再称为未来的盲测。

只有误差及路由排序检查有明确结果后，才比较“旧 MILP+旧预测器 / 旧 MILP+新预测器 / J3O-inspired+新预测器”，支持域、轨迹、原六项 U 和约束保持一致，初始解时间计入总耗时。当前未进入该比较，也未安排 GP-BO/MADS 或长时间扫描。

## 验证成本与文件

本轮数据复算为 4 次旧预测，耗时 {r['elapsed_seconds']:.3f} 秒（不含开发、测试、报告绘制）。另新增 4 个定向测试，检查丢失/重复请求、输入/路由/轨迹错配、失败结果和 receipt 冒充 READY；全部通过。零 GPU 运行、零新拟合、零优化器搜索。

- report.json：完整结果与明确未完成项。
- requests.csv：346 条完整实测/预测请求记录。
- per-task.csv：服务、等待、Mean/P95。
- same-trace-routing.csv：两条冻结轨迹的交叉路由预测。
- A-old-prediction.json / B-old-prediction.json：正式旧预测器输出。
- source-manifest.json：使用的原始数据与脚本指纹。
"""
    (out / "REPORT.md").write_text(text)
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
