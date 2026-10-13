# Module 24A：Selected Pool Verification & Utility Prediction Error Calibration

本轮完成 CPU 优化器复核、历史分组校准与留出评估、HIGH 最小共置诊断和相同池重新优化。没有执行 Module 24B；不能宣称跨模型组合泛化或正效用三阶段成立。

## 原优化器复核

固定 Qwen2.5-Omni、Whisper-large-v3、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq；B4=34567.5534 MiB。Text=4、Image=2、OCR=5、LID=.03、VQA=.2、ASR=.02/.80/1.52 QPS。质量准入、服务画像、实测副本域和 Module20A 的 Q/T/L/P/M 权重 0/.25/.30/.30/.15 保持原值。完整枚举 288 个副本状态，71 个满足覆盖和预算的部署都自由竞争；Text/Image/ASR 全部允许自主选路。多起点、两组 DE、Powell、ASR 密网格、独立 top-five 加密检查，路由数值收敛但没有连续全局证书。

| 阶段 | 原始浮点选择 | G ASR 比例 | 预测 U | 数值/实际含义 |
|---|---|---:|---:|---|
| LOW | G-only | 1.00000000 | -0.48665265 | G-only |
| MID | G-only | 1.00000000 | -0.52060668 | G-only |
| HIGH | G/S | 0.00011528 | -0.53680640 | S-only numerical tie; zero G-ASR simulated requests. Tiny positive G flow is retained, not credited as useful split. |

HIGH 原自由 MILP 两个 rho cap 均选择 S-only。补充搜索的极小正 G 流在 120s 周期模拟中产生 0 个 G-ASR 请求；与 S-only 的 U 差只有浮点精度，保留原始路由但不记作有意义的混合收益。LOW/MID 均为 G-only，因此原算法没有自主形成要求的三阶段。低负载/极小正路由在有限窗口可能零请求，这是旧预测器的限制。

| HIGH 路由 | 预测 U | mean / P95 ms | max rho | ASR residual capacity QPS (G/S) |
|---|---:|---:|---:|---|
| G-only | None | 305.427 / 2547.724 | 1.027476 | {'qwen': 1.477992724623509, 'openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1': 2.020559075441307} |
| S-only | -0.5368064004650557 | 68.474 / 494.913 | 0.752267 | {'qwen': 1.477992724623509, 'openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1': 2.020559075441307} |
| GS18_diagnostic | -0.5660610840040806 | 73.026 / 494.913 | 0.616859 | {'qwen': 1.477992724623509, 'openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1': 2.020559075441307} |

G-only rho>=1 时 U=null；mean/P95 仅为不稳定诊断，不可排序。HIGH 每个合法部署的 mean/P95、路径排队、容量和服务假设在 original_optimizer_audit.json 中完整保留。原始 standalone ASR 为 Qwen=654.078 ms、Whisper=494.913 ms；常数服务、周期到达、family round-robin，不能认证 Poisson 尾部或共置干扰。

## 历史训练与留出

读取 Module15–22 的全部 119 个原始 GPU worker/task 会话、请求队列、E2E、显存和系统效用；Module20 没有原始 GPU 会话，Module21D 是 Module21 的 CPU 重分析，未重复当成新观测。开发 109 会话，留出 10 会话。Module16 的两个完整组合连同 Module17 中同组合的所有 workload/会话统一留出；训练/测试组合不相交。Module19/22 负例均是开发验收，不能充作留出成功。

B 使用按模型组合/model/task/count 的中位 log 服务比、33 个经验服务分位、路径队列残差；先在整个 combo/workload/route 组内合并会话，再让每组等权。完整结束的 unstable 会话可提供未截尾 service 观测，但不能训练 stationary queue offset；不完整会话不参与正常延迟/U 拟合。未知组合只借用 model/task/count 参数并标为未验证迁移；未知模型路径沿用 standalone 并保留未知。不使用所有模型共享干扰系数。U 仍用原权重重新计算，实际 U 残差完整输出，不通过任意奖励项制造收益。

| 真正整组合历史留出 | A 原模型 | B 历史校准 |
|---|---:|---:|
| service_mean_ms_MAE | 168.6240964249293 | 138.7496764652202 |
| service_P95_ms_MAE | 423.18811041052624 | 316.5437499886965 |
| queue_mean_ms_MAE | 462.76135474154654 | 461.53185771495 |
| system_mean_ms_MAE | 712.9986731317142 | 697.0972013157269 |
| system_P95_ms_MAE | 3229.6187211290076 | 3181.245639139347 |
| U_MAE | 12.748648040076912 | 12.481812466080445 |
| false_predicted_stable | 3 | 3 |
| matched_contrast_groups | 0 | 0 |
| delta_direction_accuracy | None | None |

开发集 ΔU 方向准确率：A=0.737 (19 组)，B=0.688 (16 可预测组)。B 对部分实际可用方案 abstain，分母不同，不能直接把条件 MAE 降低称作全面改进。排序相关、最高预测收益方案的真实收益/后悔值、服务/E2E/U 误差、假 stable 和计算耗时均见 prediction_model_comparison.csv。

留出没有相同 workload 下的独立替代部署，独立排序样本=0：ΔU 方向、收益排序能力未获独立验证。服务时间误差有所下降，但队列/U 误差和假 stable 问题仍在；不能宣称新预测器具备跨组合泛化能力。

| 开发负例验收 | 实际 ΔU | A 预测 ΔU | B 预测 ΔU |
|---|---:|---:|---:|
| M19 asr_high rep1 | None | -0.6087346291743494 | None |
| M19 text_high rep1 | -0.15637940369211412 | -0.0617199897241848 | -0.14016617909533724 |
| M19 text_high rep2 | -0.145498127208473 | -0.0617199897241848 | -0.14016617909533724 |
| M19 text_high rep3 | -0.18875814103775246 | -0.0617199897241848 | -0.14016617909533724 |
| M19 image_high rep1 | -0.01878079398854854 | -0.007769409862400045 | None |
| M19 image_high rep2 | -0.018155701398068214 | -0.007769409862400045 | None |
| M19 image_high rep3 | -0.01904264541812123 | -0.007769409862400045 | None |
| M19 lid_high rep1 | -1.6057521205543042 | -0.38719992519991187 | -2.4473830393209597 |
| M19 lid_high rep2 | -2.90731323302787 | -0.38719992519991187 | -2.4473830393209597 |
| M19 lid_high rep3 | -2.2104159602699056 | -0.38719992519991187 | -2.4473830393209597 |
| M19 lid_mid rep1 | -0.09598670895644945 | -0.014863816490121029 | -0.09016051056484553 |
| M19 lid_mid rep2 | -0.17132019410880472 | -0.014863816490121029 | -0.09016051056484553 |
| M19 lid_mid rep3 | -0.05670641968889356 | -0.014863816490121029 | -0.09016051056484553 |
| M22 HIGH rep1 | -0.5553671123302517 | -0.22495835159707211 | -1.0920877236922628 |
| M22 HIGH rep2 | -0.4656964654416713 | -0.22495835159707211 | -1.0920877236922628 |
| M22 HIGH rep3 | -0.6301518451254338 | -0.22495835159707211 | -1.0920877236922628 |

## HIGH 共置诊断（开发数据）

状态 COMPLETED_DIAGNOSTICS_WITH_RETAINED_STARTUP_FAILURE；GPU 启动尝试 3（包含保留的启动失败），每种最多 2 次、正常请求测量每种最多 1 次。六任务率和相同五模型各 1 驻留固定，ASR S-only / 原探索性 G/S=18%；同一冻结 Poisson task/sample/uniform 轨迹，warmup=10s、measurement=120s、drain 上限60s。这两种是诊断配置，不代表优化器自主选择 18% G/S，也不构成 Module24B 成功证据。

- S-only: eligible=True, unstable=False, U=-2.2566722401185606, 旧预测 U=-0.5368064004650557, 实际-旧预测=-1.719865839653505, safe memory=30917.28728265385 MiB, mean/P95=201.26692192066668/1425.4552912 ms。
- GS18: eligible=True, unstable=False, U=-1.7798739574615303, 旧预测 U=-0.5660610840040806, 实际-旧预测=-1.2138128734574498, safe memory=31308.307874903567 MiB, mean/P95=202.534256504/1421.3435097 ms。

ASR service 分布、逐 worker/task queue wait、busy、backlog、显存和完整请求/遥测/失败证据已保存。S-only 中闲置的 Qwen ASR 只报告同驻留 sequential path probe，不能冒充并发 HIGH 分布。原始完整计划请求保留；unstable/incomplete 的正常 U=null，完成子集只作明确诊断。两次短诊断不能评价稀疏 LID 的可靠 P95，不能形成正式收益置信区间或独立排序验证。

## 重新优化和后续 gate

| 阶段 | 旧选择 | 校准后选择 | 对最佳纯路由预测 ΔU | 不确定性（经验 stress，非 CI） |
|---|---|---|---:|---|
| LOW | G-only | G-only | 0.0 | [-46.260973373976896, 46.260973373976896] |
| MID | G-only | G-only | 0.0 | [-46.260973373976896, 46.260973373976896] |
| HIGH | S-only numerical tie; zero G-ASR simulated requests. Tiny positive G flow is retained, not credited as useful split. | G/S | 0.03122360277222258 | [-46.22974977120467, 46.29219697674912] |

自主三阶段（排除零请求伪混合）=False。HIGH 可信正收益余量=false；独立 GPU 正式验证条件=false。B 的开发探针参数及其历史留出复评已单独标记 B_AFTER_DEVELOPMENT_PROBES，探针始终不作为测试目标。

不提出 Module24B 正式配对执行计划：需要另外收集/冻结独立的同 workload 部署配对排序证据，且跨至少两个模型组合、至少三组独立对比，并解决大队列误差和假 stable。当前结果仅支持继续预测器开发。Module24A 到此停止。

## 完整性与复现

历史 15972 个文件 metadata 不变=True，实际使用科学输入 SHA256 不变=True；每个原始 GPU 来源有 SHA256。Module19/22 负结果、历史实验和原始 Module20 代码都保留。

原 optimizer 的原有 CPU 测试、replay 一致性、数据组合隔离、未知路径回退和不稳定 U/null 检查见 verification.log。原始与校准预测器的实际 CPU 单次耗时见 predictor_runtime_benchmark.csv；comparison 表中的耗时包含诊断统计，优化 wall time 单独报告。CPU 搜索与拟合用 envs/data_env；GPU controller 用已有 gsdeploy 环境。首次离线报告生成的缺失字段错误、controller 缺少 psutil 的 prelaunch 日志、Whisper READY 未声明 ASR 的 startup 失败均保留。启动失败没有请求测量；只补齐 READY 能力元数据后在用户每路由两次上限内继续。保存路径错误仅移动本轮新增失败文件到 Module24 目录，未改动任何既有历史文件。
复现步骤：python -m module24.audit → python -m module24.calibration → python -m module24.probe → python -m module24.finalize。CPU 步骤设置 CUDA_VISIBLE_DEVICES 空字符串、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1。已完成步骤拒绝静默覆盖；不要重跑 GPU 诊断。

## 最终候选的实测覆盖范围

探针前冻结历史模型的排序留出见 ranking_holdout_validation.csv；开发探针更新后的分组排序另见 ranking_after_development_probes.csv，Module24 行仅为 DEVELOPMENT。

| 阶段 | 新候选副本数 | workload 在开发中出现 | 精确部署+workload 已测 | 未获当前共置校准的路径数 |
|---|---|---|---|---:|
| LOW | Qwen×1, RoBERTa×2, ConvNeXt×2, PARSeq×2 | False | False | 5 |
| MID | Qwen×1, RoBERTa×2, ConvNeXt×2, PARSeq×2 | False | False | 5 |
| HIGH | Qwen×1, Whisper-large×1, RoBERTa×2, ConvNeXt×2, PARSeq×2 | True | False | 6 |

未覆盖 workload/副本的回退是带未知误差的名义预测；不能用缺少校准的副本看起来更快来认定收益。整组合留出、服务测量、路由配对和复制数支持分别报告。六个新旧阶段候选已从保存的路由独立重算 U/mean/P95；全部一致。全部 GPU 进程/显存已释放，原始历史文件未变。

原算法/新算法部署差异还需在同一个预测器下比较；上表的 ΔU 是新选择与最佳纯路由的对比，以下另报告新选择−旧选择（实测 ΔU 未测）：

| 阶段 | A 预测新−旧 ΔU | B 预测新−旧 ΔU | B 经验 stress 区间（非 CI） |
|---|---:|---:|---|
| LOW | -0.04326655503274024 | 0.5826494022902341 | [-45.67832397168666, 46.84362277626713] |
| MID | -0.04114008372849687 | 0.5441927940673666 | [-45.71678057990953, 46.80516616804426] |
| HIGH | -0.05061635551035748 | 1.05329650907897 | [-45.20767686489793, 47.314269883055864] |

单次开发诊断：GS18−S 的实际 ΔU=0.476798，旧预测 ΔU=-0.029255。收益分解 mean=0.136153、P95=0.342342、throughput=0.000000、memory=-0.001697；汇总 pooled mean/P95 几乎不变，不能替代按任务参考值和固定 offered 权重定义的 U。只有一次短配对，CI 未估计，独立验证=false。

探针更新后的 B 对同驻留 GS18−S 的预测 ΔU=-0.007317，仍与实际方向相反；路径中位 service/queue 残差不足以预测路由引起的全系统干扰变化。

A 表示重新计算的固定 Module20A 周期预测器。Module22 原实验日志使用 Module21 的 Poisson/FFT 预测器，不能混作 A；两种原预测均保留在 utility_prediction_residuals.csv 和 negative_case_development_acceptance.csv 的 historical_logged_* 字段。

同一 workload 内的方案排序另见 within_workload_ranking_validation.csv：先合并整次重复会话，再比较部署/路由配置的真实 U，报告 top-1 命中、收益后悔值、配置拒绝预测数和秩相关。留出依然没有可用排序组；开发排序不是独立泛化证据。

| Module22 HIGH 负例 | 原实验日志预测 ΔU | 固定20A重算 ΔU | 实际 ΔU |
|---|---:|---:|---:|
| rep1 | 0.04300361645452233 | -0.22495835159707211 | -0.5553671123302517 |
| rep2 | 0.04300361645452233 | -0.22495835159707211 | -0.4656964654416713 |
| rep3 | 0.04300361645452233 | -0.22495835159707211 | -0.6301518451254338 |
