# TwoFIFO 队列预测接入状态（CPU 阶段）

本阶段把“只看当前积压”改成了完整全过程预测接口，并保留了用户要求继续借鉴的机制。公开到达日历先形成完整 offered cohort；预测随后按 sender FIFO、worker FIFO、RESULT reader/collector FIFO、READY 释放、固定 measurement window 和 drain 逐事件推进。请求预测使用剩余 solo/resource work，而不是把队列长度直接换算为等待时间。

已接入的队列机制如下：

* **InferLine**：完整未来到达表、事件驱动排队、短期工作量与多尺度 envelope。已知全过程到达表优先于随机到达率；AlpaServe 的 CV/Gamma 情景只在未来到达确实未知时作为预声明情景，不能覆盖已知日历。
* **Clockwork**：显式 sender/worker/reader/collector 状态、FIFO 队首阻塞、最老等待、READY 和恢复片段。settled prefix seed 保留上一阶段的 worker lifetime、发送时钟、reader/collector 时钟和 producer counter。
* **Sinan**：短期队列状态与长期恢复风险分开；恢复风险只作诊断，不能改写原 `rho`、`g(n)` 或六项评分。
* **项目扩展**：batch 等待、padding、microbatch、公共权重子矩阵的额外准备/残余 GEMM、resource-feasible HOL、多尺度 offered work，以及可见状态滚动校正接口均有明确字段。缺少真实执行证据时，整项保持 UNKNOWN。

新增 `twoFIFO_queue_envelope_companion_v1.py` 是独立诊断层。它复用原 `forecast_case` 的 public compiler、settled seed 和完全相同的 draw IDs/weights，为每个生成世界计算有限速率的多 owner FIFO 外包络。它会保留所有请求和未知概率质量，输出 START/READY/RESULT reader、完整 cohort P95 和 fixed-drain 未完成比例的范围；它不生成潜在点、不分配未经校准的概率、不重算或填补 M/E，也不参与候选选择。原 point prediction 和六项评分文档保持不变。

新增 `twoFIFO_candidate_ranking_v1.py` 完成多候选排序。所有候选共享公开 arrival/input cohort、window/drain 和同一 joint draw domain；排序逐对调用原 `compare`，Q/T/L/P/M/E 仍各占 `1/6`。未知世界保留原始概率质量，conditional mean 仅作诊断；只要有未知效用，就不给出完整域赢家。M 仍采用 startup→drain owned sampled peak 并只应用一次安全因子，E 只计 measurement、startup 按 3600 秒摊销一次。

CPU 验证收据：

* queue envelope companion：7 项新增测试通过；加上 outer envelope、generated whole-block、rolling visibility 回归共 27 项通过；
* candidate ranking：作者 15 项、原 ensemble 16 项、独立 root review 4 项通过；独立收据为 `artifacts/predictor-twoFIFO-candidate-ranking-root-review-v1-20261028-cpu01/report.json`；
* 这些收据是契约、算术和因果边界验证，不是 GPU 预测准确率，也不是最终分流收益。

仍未通过的门槛：USHER 的计算型/显存型互补、Roomie kernel sequence/occupancy 对齐、batch 后端和公共权重真实 cache reuse 尚未获得同硬件原生证据；busy native device progress、source-rebound exact12 采集和独立 validation 也未完成。因此当前不启动 GPU fit、独立预测或四组收益实验。
