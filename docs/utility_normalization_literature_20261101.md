# 六项效用归一化的文献依据与统一参考值

核查日期：2026-11-01。结论：**六项成本等权相加是本项目的研究设计，没有在本次核查的论文中找到完全相同的评价规范。** 各项使用常见指标、流量权重和无量纲比值，但不能据此声称这套总效用、六个 `1/6` 或原来的四个延迟分母已获文献验证。

后续补充：按“六项归一化均有文献方法来源”的要求，已完成[新的六项重设计](utility_normalization_redesign_20261101.md)，新增核验 OECD/JRC 的具体范围/参考比值公式、DRF 的资源份额、Zeus 的能量基线及 MLPerf Power 测量原则，并找到 μ-Serve 直接将尾延迟除以SLO的原文。2026-11-03 已[切换软件与正式入口](utility_normalization_switch_20261103.md)，真实参考值待独立冻结；本页的1秒配置作为历史版本保留。

## 公式与来源边界

设 `alpha_t=lambda_t/sum(lambda)`，当前六项为：

```text
U = -(Q + T + L + P + M + E)/6
Q = sum_t alpha_t * loss_t
    loss_t = 1-EM (QA), 1-accuracy (image), CER (OCR), WER (ASR)
T = 1 - N_completed_within_window/N_offered
L = sum_t alpha_t * mean_e2e_latency_ms_t/b_t
P = sum_t alpha_t * p95_e2e_latency_ms_t/b_t
M = safety_factor * owned_sampled_peak_memory_MiB/GPU_budget_MiB
E = (J_run/H + J_startup/A)/P_ref
```

M 的当前实测口径为启动至排空的 owned 采样峰值，安全系数只乘一次；规划代理的适用范围另行标识。公式实现见 `gsdeploy/prediction.py::components`，原生观测组装见 `gsdeploy/evaluation.py`，能耗见 `gsdeploy/energy.py`。

|项|文献可支持的部分|本项目仍须自行说明的部分|
|---|---|---|
|Q 质量损失|J3O，PDF 第 4 页，按归一化任务需求加权系统质量；Proteus，PDF 第 8 页，按家族最佳精度归一化不同变体的精度|我们直接混合 `1-EM/1-accuracy/CER/WER`，没有采用 Proteus 的最佳变体分母；不同任务损失的一单位并不保证同等用户价值|
|T 窗口内未完成比例|MuxServe，PDF 第 7 页，报告吞吐与 SLO attainment；AlpaServe，PDF 第 11 页，主要评价满足目标的请求比例|我们的 T 按测量窗口结尾计完成数，并非逐请求延迟 SLO 违约率；不能把 SLO attainment 公式直接说成 T 的来源|
|L 平均延迟比值|USHER、Proteus、MuxServe、AlpaServe 使用独立执行延迟的倍数确定/扫描 SLO|把 `mean/b_t` 连续惩罚并纳入总效用，是本项目设计；SLO 倍数法支持参考尺度的选择思路，不等于支持这条效用公式|
|P 尾延迟比值|InferLine，PDF 第 2 页，使用 P99 延迟 SLO；Cascadia，PDF 第 6 页的例子使用 P95 延迟与质量惩罚|我们采用各任务 P95 的流量加权比值，且与均值共用 b_t；P95 而非 P99、与均值同时等权计入，均属设计选择|
|M 显存占预算比|J3O，PDF 第 4 页，将内存等资源作为容量约束；占用/预算可直接消除单位|论文未支持我们这个 owned 采样峰值、安全系数或把显存作为 `1/6` 软成本的完整组合；显存硬合法性仍须单独满足|
|E 功耗参考比|μ-Serve，PDF 第 4 页，展示满足 SLO 前提下减少功耗的研究动机|`J/H` 是平均功率的单位换算；`J_startup/A` 的摊销、`P_ref=450 W`、`A=3600s` 和 `1/6` 权重是项目约定，不是 μ-Serve 提出的规范|

因此，这里“归一化”的准确含义是**化为无量纲或损失比例**，不是把六项都映射到 `[0,1]`。L/P 可以任意大，CER/WER 也可能超过 1，E 含启动项也可超过 1。六项等权不等于实际影响相同；L/P 两项合计占两个系数，且彼此相关。只显示总分会掩盖这种偏好，报告必须同时保留原始单位、六项值和各项贡献。

## 与参考延迟直接相关的原文

以下页码统一使用**本地 PDF 物理页码**，包含封面；引文来自已下载全文，不将引言引用误算为本论文的实验设置。完整摘录、文本位置、来源文件和 SHA256 保存在本地审计目录的 `literature-evidence.json`。

|论文与位置|原文片段|能得出的结论|
|---|---|---|
|Proteus，§6.1.2，PDF p.8|“profile its fastest model variant that can run on a CPU with the smallest batch size”; “set the latency SLO of the model family to be 2× of its profiled latency”|同一模型家族共享由固定基线测得的目标；正文还明确说给所有异构应用同一 SLO 并不实际，并在 §6.6 分析敏感性|
|[USHER (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/shubha)，§2.1，PDF p.4|“the SLO of a model is taken as double the average inference latency of a single request in a Nvidia V100 GPU”|提供“单请求基线的 2 倍”的直接依据；其单位是模型，移植到可替换模型比较时应另行固定每任务共同基线|
|MuxServe，Metrics，PDF p.7|“scale the latency to different multiplies of single device execution latency (i.e. SLO scale)”|用独立执行延迟的倍数做敏感性评价；未给我们的四任务规定 1 秒|
|[AlpaServe (OSDI 2023)](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)，§3 / §6.2，PDF p.6 / p.11|“scale the SLO to different multiplies”; “default SLO ... 5× inference latency”|USHER 所引文献也采用基线倍数，并非所有论文一律使用 2 倍|
|μ-Serve，PDF p.4|“serving gpt2-large can meet the SLO (1000 ms)”|1 秒确实在特定模型实验中出现过；只能作为个例，不能外推为 QA、分类、OCR、ASR 的统一公认目标|
|[Clockwork (OSDI 2020)](https://www.usenix.org/system/files/osdi20-gujarati.pdf)，摘要，PDF p.2|“meeting 100 ms latency targets for 99.9999% of requests”|其他任务采用明显不同目标，进一步说明没有跨任务统一 1 秒的标准|
|Cascadia，§3.3，PDF p.6|用最大/最小模型质量定义 `z1*` / `z2*`，质量惩罚除以 `z1*-z2*`；“normalize the quality shortfall so the penalty is dimensionless”|归一化多目标惩罚有明确先例；其范围归一化、质量约束与惩罚系数都不同于我们的六项等权，不能冒称同一方法|

USHER 该段将方法追溯至 [4] AlpaServe 和 [14] GPUlet；本次已直接核对 AlpaServe 全文。Proteus 该段引用 [35] INFaaS 和 [36] Nexus。上表对 Proteus 自身实验设置的引用成立，并不表示这些被引论文的全部细节都已在本次重新核查。

## 本次生效的后续配置

按用户要求，后续真实四任务比较统一使用：

|任务|旧 b_t（ms）|新 b_t（ms）|同一原始延迟的 L/P 惩罚变化|
|---|---:|---:|---:|
|QA / text|1000|1000|×1|
|图像分类|250|1000|×0.25|
|整页 OCR|10000|1000|×10|
|ASR|3000|1000|×3|

这是**统一 1 秒的研究比较尺度**，不是实测性能基线、经验证的业务 SLO 或硬截止时间。mean/P95 同时使用这组值，同一任务的所有候选部署、模型、种子共用分母。不能每个模型除以自己的延迟，否则慢模型可能因分母更大而获利。

- 机器可读目标：[utility_objective_20261101.json](../configs/utility_objective_20261101.json)。
- 静态研究配置：[system_one_second_20261101.json](../configs/system_one_second_20261101.json)。保留既有静态 `calibrated_replay_v2` 的候选域/负载/画像，只替换参考值；catalog 的提取来源和哈希见审计目录 `config-provenance.json`。此配置的校准适用性限制仍然存在。
- 资源控制后续设计：[resource_spillover_study_20261101.json](../configs/resource_spillover_study_20261101.json)。沿用原资源实验的物理域和负载，改为新目标与四项 1000ms。该设计仍为 CPU 准备材料。

旧 `20261024` 资源设计、原始测量、旧搜索证书、SHA 绑定的研究协议与合成测试 fixture 保留各自身份；新实验应从上面新入口读取。改分母不改变实际延迟，也不能将旧最优部署称为新目标下的最优部署。

## 更有文献依据的可选后续方案

如果需要任务难度可比的尺度，推荐借鉴 Proteus 的固定家族基线思想，预先定义 `b_t=k*s_ref,t`：为每项任务选择一个共同参考部署，在同一 GPU、输入协议、输出长度、低负载和测量边界下测量 `s_ref,t`。冻结参考部署、样本、重复测量方法和汇总统计量；所有可替换模型均使用此共同分母。`k=2` 可作为来自 Proteus/USHER 的一个实验点，并预先扫描 `k∈{1,2,4,8}`；AlpaServe 的 5 倍则说明倍数本身也需说明，而非一条普适定律。

这是**待另立协议验证的替代方案**，本次没有据事后收益挑选参考模型/倍数。旧单模型 handler/service-time 画像可做诊断，但与端到端服务时间边界不同，不能无说明直接充当 e2e SLO。生成模型的 TTFT/TPOT 与整请求耗时、音频时长相关的 ASR RTF 也不是同一指标，不直接混用。

当前更稳妥的论文表述是：“本研究将六种成本无量纲化并等权标量化；四任务使用预先指定的共同 1 秒延迟尺度，报告原始指标及尺度敏感性。该尺度表述比较偏好，不声称是通用业务 SLO。”

## 离线核验与复现

本次对既有静态确认的 15 次运行（5 候选 × 3 seeds）和既有 spillover 的 15 组配对按共同 `0.5/1/2/5s` 重评分，保留负结果、原资格条件与固定路由。没有启动 GPU、重新寻优或重放会随效用改变决策的在线策略。

共同 1 秒时，原静态选中方案对四个既有对照的配对 ΔU 在三个 seeds 中均为正，但这不证明新目标全域最优；共同 0.5 秒时，它对纯专用方案的三个配对 ΔU 均为负，说明结论对尺度确实敏感。最近耦合 spillover 的 seeds 303/304 在 1 秒下分别为 `−0.02800846`、`−0.03720749`，仍未显示正收益。不同研究阶段不合并成一次统计重复。

```bash
python -m scripts.audit_latency_normalization_20261101 \
  --output artifacts/utility-normalization-audit-20261101/rescoring
python -m gsdeploy.resource_study prepare \
  --design configs/resource_spillover_study_20261101.json \
  --output artifacts/utility-normalization-audit-20261101/resource-packet.json
```

输出目录/文件须尚未存在。重评分摘要见本地 [report.zh.md](../artifacts/utility-normalization-audit-20261101/rescoring/report.zh.md)，逐任务观测、原分数复算、各尺度分数及来源哈希见相邻 `analysis.json`。结果只能说明既有观测在新偏好下的分数，不能替代新目标下的搜索和独立确认。
