> 2026-10-18 更新：当前按持续任务执行，真实状态见 `artifacts/continuous-gpu-study-20261018/continuation-plan.json`，结果与拒绝项见 `docs/repair_status_20261018.md`。正在运行的是训练及新校准链，完整系统矩阵尚未开始。加入七项完整校准显存屏障后，基础验证清单为 **90 项**。下方旧实验目录和命令属于历史准备示例；源码、配置或候选改变后，必须重新生成冻结包，不可直接复用旧回执或旧运行命令。

# 四任务实验顺序与启动入口

2026-10-18 当前状态见 [原生适配修复与验证](repair_status_20261018.md)。MiniCPM ASR 已完成 64 条修复验证，最新分类验收 18 项全部采集完成；分类、旧 QA 和 OCR 的质量仍未全部达标，模型池未准入。新 QA 的下载和有限校准使用独立状态文件，不能把历史监督文件当作当前运行状态。

以下保留早期实验设计和命令作参考。旧清单已绑定旧源码，不能直接重跑或修改哈希绕过检查；恢复正式实验必须使用新冻结包。基础清单在加入 7 个完整校准显存检查后为 90 项，完整多负载矩阵仍待合法模型池与实测容量。

2026-10-17：首轮已停止，通用显存不符合原定模型池条件。新显存优化配置已实现，
仅做 CPU 验证，后续 GPU 入口见 [显存优化与待测入口](compact_memory_optimizations.md)。
首轮修正与阶段记录见 [GPU 实验记录](gpu_experiments.md)。
以下保留原 CPU 准备包的启动说明；实际运行使用新冻结实验包，见
`artifacts/compact-gpu-results/latest.json`。

本轮只完成 CPU 前置准备，未启动 GPU。当前入口为
`python -m gsdeploy.profiling.compact_experiments`，默认准备清单，执行须显式 `--execute`。
LID/VQA 不在本轮；候选、原生配置和统一协议由 study/execution/protocol 三份配置固定。

## 已准备的实验阶段

| 阶段 | 作业数 | 输入与用途 |
|---|---:|---|
| smoke | 7 | 各模型每任务 3 条校准；检查加载、量化、完整输入、EOS、输出与显存 |
| quality | 7 | 16 条路径、6400 个独立质量请求；每样本一次；不重复扩大样本数 |
| profile | 21 | 7 模型 × n=1/2/3，配置内各 3 轮；完整校准集、最少 64 请求/任务和 30 秒 |
| matched_cost | 48 | 16 路径 × 种子 42/43/44；同任务的 G/S 使用完全相同的输入池与固定到达轨迹 |

共 83 个条件作业。n2/n3 依赖 n1；真实内存超预算、加载/输出失败均保留失败记录，
不把“列入清单”写成可行副本。通用与专用都开放多副本；不按 n 倍单副本容量替代实测。
每次只运行一个 GPU 作业，避免其他进程污染整卡能耗。未运行阶段不填补结果。

matched_cost 采用固定 H=600s、A=3600s，初始单任务请求率：
QA 0.1、图像 0.2、OCR 0.02、ASR 0.05 req/s，三个模型类别承受相同绝对需求。
这是启动前固定的低负载成本探针，不保证所有候选稳定；不稳定则该对照不能准入，
需要另立同口径更低负载的完整比较。48 个正式窗口本身共 8 小时，另有加载、排空和其他阶段。
低速 OCR 只有约 12 个正式请求，不能用它认证长期尾延迟；正式质量仍使用独立 200 页。

## 准备与单个作业运行

准备在 CPU 执行，不加载模型。所有输出使用新目录：

```bash
python -m gsdeploy.profiling.compact_experiments prepare \
  --inventory artifacts/four-task-small-pool-20261015/final-preparation/resources/resource-inventory.json \
  --classmap /root/autodl-tmp/gsdeploy-assets/datasets/imagenet-class-index.json \
  --output NEW_EXPERIMENT_PACKET
```

完成前置检查后，固定实验包位于：
`artifacts/four-task-small-pool-20261015/final-preparation/experiments/experiment-manifest.json`。
包内绑定源码配置、权重清单、原生 worker、解释器、每个作业、输入和种子的哈希。
当前 `execution_catalog=null`；这是采集准入证据的入口，不绕过正式运行 API 的准入要求。

以下仅为用户恢复 GPU 实验后的命令，本轮未执行：

```bash
python -m gsdeploy.profiling.compact_experiments run \
  --manifest artifacts/four-task-small-pool-20261015/final-preparation/experiments/experiment-manifest.json \
  --job smoke-generalist_1-n1 \
  --output artifacts/compact-gpu-results/smoke-generalist_1-n1 --execute
```

它自动使用该候选绑定的独立解释器。其他 ID 见清单；所有结果放在同一个父目录，
以便读取前置作业回执。先逐个 smoke，再独立 quality 和 profile-n1，通过后才尝试更多副本及成本探针。
加载失败、超显存、截断或未知 EOS 时先处理该配置，不让无效配置进入模型池。

全部单模型对照完成后，CPU 汇总：

```bash
python -m gsdeploy.profiling.compact_experiments assess \
  --manifest artifacts/four-task-small-pool-20261015/final-preparation/experiments/experiment-manifest.json \
  --results artifacts/compact-gpu-results --output artifacts/compact-pool-screen.json
```

每个 S 必须逐一通过三个 G 的质量及成本约束；缺证据不能通过。
筛选报告仍不是整套部署 catalog；不同精度配置不能继承原配置的质量记录。

## 共驻、分流和静态搜索

`prepare-plan` 可在 CPU 验证明确的部署/副本/路由，继续生成相同采集器使用的作业。
计划 JSON 包含 `deployment`、`routing`、`rates`、`duration_seconds`、`seed`；路由用每秒请求数，
不能混用比例。它支持 G 常驻、G+S 常驻但不分给 S、交给 S、G/S 分流、全专用和真实副本。

```bash
python -m gsdeploy.profiling.compact_experiments prepare-plan \
  --manifest PREPARED_EXPERIMENT_MANIFEST --plan EXPLICIT_PLAN_JSON --output NEW_PLAN_PACKET
```

在固定输入需求下比较完整方案，不能把各模型独立耗电相加。记录整卡加载/常驻/运行/排空功率、
峰值/驻留显存、全部预登记到达请求、队列、端到端 mean/P95/P99、窗口吞吐与有界排空。
失败请求保留；功率采样缺覆盖、外部 GPU 进程、未完成/不稳定均使证据失效。
正式能耗用固定到达窗口 J_run/H + 启动 J_start/A，排空电量另报。

容量/质量通过后冻结共同参考方案 C_ref，再用 0.2/0.5/0.8/1.0/1.2 倍输入扫描静态强度，
并在固定总流量下扫描任务偏斜。每个比较点所有方案使用相同绝对到达率、种子、H 和输入池；
不能按各自容量分别设需求。短窗口尾延迟不充分时统一延长该比较点。

从合法共驻/副本测量构建可追溯 catalog，使用现有联合 MILP、统一预测器、运行时和五项效用。
先在可完整校核的小有限域与穷举对照，再扩域；未知共驻容量或能耗保持未知。
无“必须保留一个通用”约束。低负载单通用、偏斜时专用、拥塞分流及高负载全专用/副本
都是待验证现象，不能通过限制通用、添加空算力或伪造能耗保证出现。
