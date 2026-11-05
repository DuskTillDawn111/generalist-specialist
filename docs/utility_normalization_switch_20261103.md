# 六项归一化 v2 正式切换

2026-11-03 已将新研究入口切到 `six_term_reference_v2`，预测、原生评估和在线整段评分共用这一版本。六项各 `1/6`，稳定性与可执行能力规则保留。真实独立参考延迟、完整基准 episode 能量仍为 `null`，因此当前真实配置不会生成新 U、排名或最优证书。

## 生效公式与来源

`alpha_t=lambda_t/sum(lambda)`，`U=-sum(w_i*C_i)`。OCR/ASR 先累计编辑错误和参考单位，再计算原生损失。

| 项 | 当前公式 | 文献与适配 |
| --- | --- | --- |
| Q | `sum alpha_t*(native_loss_t-a_t)/(z_t-a_t)` | OECD/JRC 2008 §5.3；Cascadia ICLR 2026 §3.3。冻结可信开发路径的损失范围；连续质量成本是本项目适配，允许超界与负值 |
| T | `1-N_good/N_offered` | MuxServe ICML 2024 的 SLO attainment，USHER OSDI 2024 的 goodput。执行成功且响应时间 `<=D_t` 为 good；所有 offered 请求保留在分母 |
| L | `sum alpha_t*mean_e2e_ms_t/D_t` | OECD/JRC §5.4 的参考比值，USHER 与 Proteus ASPLOS 2024 §6.1.2 的基线倍数 SLO |
| P | `sum alpha_t*p95_e2e_ms_t/D_t` | μ-Serve USENIX ATC 2024 §4.2 直接使用 P99/SLO；本项目适配为 P95 |
| M | `raw_owned_startup_to_drain_sampled_peak_mib/B` | DRF NSDI 2011 §4 / Algorithm 1 的用量/容量；本项目适配为 GPU owned 显存采样峰值 |
| E | `J_episode/J_reference_episode` | Zeus NSDI 2023 Fig.1 的基准能量比；MLPerf Power HPCA 2025 §IV 的边界对齐和功率积分原则。适配为推理 GPU 板卡 episode |

逐条原文、页码、链接与限制见[完整文献设计](utility_normalization_redesign_20261101.md)。这些来源支持各项变换和测量原则；完整六项组合、等权与参考部署政策属于本项目设计，不能称为某论文原式。

`D_t=2*isolated_reference_mean_e2e_ms_t`，预声明敏感性倍数 `1/2/4/8`。任务之间可以不同，同一任务所有候选共用。拒绝以旧 handler 时间、各候选自己的耗时或猜测值替代。窗口后、自己 deadline 内完成的请求仍可为 good。

M 的安全系数继续用于约束，评分用原始用量。E 包含启动加载、priming、warmup、正式窗口、排空；取消旧参考功率与启动摊销公式。测量范围是 GPU 板卡，非完整机器能耗。只有窗口能量或缺排空遥测时 E 未知。

## 当前入口与证据状态

- 静态四任务：[system_reference_v2_20261103.json](../configs/system_reference_v2_20261103.json)。加载原保留模型和输入池，再叠加独立参考文件，不改旧 catalog。
- 参考契约：[normalization_reference_v2_20261103.json](../configs/normalization_reference_v2_20261103.json)。质量锚点来自已保留、可执行通用/专用路径的开发池，保存路径损失与来源哈希；不称为独立确认质量。独立参考 e2e 和基准能量尚待测量。
- 目标契约：[utility_objective_reference_v2_20261103.json](../configs/utility_objective_reference_v2_20261103.json)。保存六项公式、权重和逐项来源。
- 资源研究：[resource_spillover_study_reference_v2_20261103.json](../configs/resource_spillover_study_reference_v2_20261103.json)。其请求率/H 与静态域不同，因此使用自己的系统、目标和参考文件，不能借用另一个工作负载的能量分母。
- 完整 CPU 软件样例：[paper_fifo_reference_v2_cpu_fixture.json](../configs/paper_fifo_reference_v2_cpu_fixture.json)。参考值显式标为合成，不授予原生测量资格。

真实配置当前 `plan` 返回 `NORMALIZATION_REFERENCES_PENDING`、`winner=null`、`optimality_certified=false`。缺项不补零，不沿用旧分母；预测可展示已有原始量，但没有完整新 U。

参考值冻结后，也不能把旧 `calibrated_replay_v2` 的窗口平均功率模型当作完整 episode 模型。旧适配器若不能给出原始峰值和完整 episode 能量，仍返回未知，需要新候选测量或满足新边界的预测画像。

## 参考冻结流程

1. 在确认候选之前，固定开发质量路径、单任务参考路径、GPU/检查点、输入/输出协议、工作负载、H、warmup、priming、drain 与能量对照部署。参考文件中的基准部署目前是预声明候选，须实测稳态可行后才能冻结；若不满足，需要新版本重新声明。
2. 每任务用指定单副本专用路径做独立、无竞争、无排队 e2e 开发运行；不能以已经混部的服务时间画像代替。参考种子与保留的确认种子分开。
3. 固定覆盖四任务的基准部署及专用路由，重复运行相同 episode 协议，保存从 startup 到 drain 的原始整卡功率序列。工具要求重复、不同种子的 episode，取均值；重复次数本身不保证统计精度，仍需报告样本和区间。
4. 写原生记录清单，运行 CPU 冻结工具。它验证 cohort、身份/协议/GPU、参考路径隔离、稳态可行性、功率覆盖和种子隔离，从记录计算分母并保存哈希，不接受手填标量作为测量依据。

清单格式（路径相对清单文件；这些是待采集记录的示例名称）：

```json
{
  "schema": "gsdeploy.normalization_measurements.v1",
  "split": "normalization_development",
  "confirmation_seeds": [43, 44, 45],
  "task_runs": {
    "text": ["reference/text-201.json"],
    "image": ["reference/image-202.json"],
    "ocr": ["reference/ocr-203.json"],
    "asr": ["reference/asr-204.json"]
  },
  "episode_runs": ["reference/baseline-205.json", "reference/baseline-206.json"]
}
```

```bash
# 当前参考状态；只做 CPU 读取
python -m gsdeploy.reference_freeze --config configs/system_reference_v2_20261103.json

# 采集完成后，由原生记录生成新的冻结参考文件
python -m gsdeploy.reference_freeze \
  --config configs/system_reference_v2_20261103.json \
  --measurements artifacts/reference-development/measurements.json \
  --output artifacts/reference-development/normalization-frozen.json

# 完整新版软件样例
python -m gsdeploy plan --config configs/paper_fifo_reference_v2_cpu_fixture.json

# 当前资源域 CPU 准备
python -m gsdeploy resource-study prepare \
  --design configs/resource_spillover_study_reference_v2_20261103.json \
  --output artifacts/reference-v2-resource/packet.json
```

冻结工具输出新参考路径与哈希。复制当前系统配置到新版本，令 `normalization_reference` 指向该文件；`latency_reference_ms` 保留 null 时由 loader 从冻结参考填入，显式填写且冲突会报错。旧 pending 文件不覆盖，旧版本与凭据不改写。冻结参考还绑定实际 warmup/priming/drain 协议；评分证据协议不一致时 E 未知，原生启动会拒绝该不匹配。

## 实现和验证边界

公共公式和契约在 `gsdeploy/normalization.py`，参考冻结在 `gsdeploy/reference_freeze.py`。版本与参考 SHA-256 进入评分；连续问题的目标身份也绑定参考 SHA-256。候选和资源控制对比不能各自换分母。

MILP 改为支持负的归一化 Q 下界，T 下界使用任务 deadline，独立预测 M 下界去除安全因子，旧 E 分解不迁移。DFS 与连续路由保留有效的全局质量下界，尚未重新证明的新尺度细化界明确禁用。有限域证书只适用于当前数学预测域，不是 GPU 最优性证明。

CPU 验证覆盖六项手算、质量超界、CER/WER 不裁剪、共同任务参考、deadline 边界与失败分母、原始显存/安全约束分离、排空能量、缺项状态、配置版本冲突、原生离线评分接线、MILP/DFS 与穷举及冻结工具拒绝伪测量/种子泄漏。另运行历史运行时、预测、MILP、连续界、搜索会话、资源实验包与系统回归。测试数量及结果见本次交付说明；没有新增 GPU 运行或真实收益结论。

本次实际通过：主回归 **193 项**；最终协议/身份修改后复跑相关 **110 项**；新版公式与冻结工具专项 **17 项**。三组存在重叠，不作加总。`compileall` 与差异空白检查通过；资源 CPU 实验包 16 个任务结构校验通过。日志、合成计划和校验记录保存于 `artifacts/utility-normalization-switch-20261103/verification.json`。
