# Generalist–Specialist

单 GPU 上通用模型与专用模型的联合部署和请求路由系统。输入模型画像、任务流量与显存预算，输出模型/副本部署及路由，并使用同一套运行时记录和评估请求。

```text
configs → DeploymentSystem → optimizer + predictor → plan
                              ↓
samples + workers → runtime → request records → evaluation
```

## 快速开始

Python 3.10+，CPU 规划使用 NumPy 和 SciPy/HiGHS：

```bash
python -m pip install -e .
gsdeploy plan --config configs/demo.json --output artifacts/plan.json
gsdeploy predict --config configs/demo.json --plan artifacts/plan.json
gsdeploy run --config configs/demo.json --plan artifacts/plan.json \
  --samples examples/demo_samples.jsonl --duration 1 --warmup 0 \
  --output artifacts/run.json
gsdeploy evaluate --run artifacts/run.json
```

未安装命令行入口时，可将 `gsdeploy` 替换为 `python -m gsdeploy`。输出文件不覆盖已有实验记录；重新执行时使用新目录。

示例的 `mock` 后端使用真实进程通信和测量流程，但不加载模型、不使用 GPU，也不产生模型质量或真机性能证据。正式效用 `U` 保持为空，另给出诊断算术值。

## 项目结构

```text
gsdeploy/       可安装的核心包；含 backends、runtime、profiling、forecasting
configs/        模型池、研究协议、worker 配置与依赖锁
examples/       CPU mock 最小运行输入
experiments/    调用公开 API 的实验入口
scripts/        输入准备、校准、搜索、报告及测试入口
tools/          按日期和课题保留的研究实现与复核工具
tests/          可移植回归测试与需要本地证据的研究测试
docs/           架构、协议、研究记录与源码快照
artifacts/      本地实验产物（不进入 Git）
```

目录职责与 Git 边界见[项目维护说明](docs/project_structure.md)；
[文档导航](docs/README.md)汇总研究入口与完整索引。
脚本、配置、研究工具和测试分别见各目录的 README。

核心包不导入编号实验目录，不从历史 `results/` 动态加载实现，也不依赖固定研究服务器路径。新增算法与功能在 `gsdeploy/` 中开发；实验调用公开接口。

## 资源控制研究：CPU 准备

2026-10-24 新增逐 worker 资源配置、整请求并发准入及六项等权实验包准备入口。
下列命令仅做 CPU 校验，不启动 GPU、MPS 或模型：

```bash
python -m gsdeploy resource-study prepare \
  --design configs/resource_spillover_study_20261024.json \
  --output artifacts/resource-control-cpu-20261024/experiment-packet.json
python -m gsdeploy resource-study validate \
  --packet artifacts/resource-control-cpu-20261024/experiment-packet.json
```

整请求准入已可用 CPU mock 验证；MPS 上限目前准备启动环境，静态 SM／Green Contexts
目前校验配置声明，均不代表已经验证硬件分区。新资源域的预测、原生运行与最优证书
仍待重新验证，详见[CPU 实现及接续说明](docs/resource_control_cpu_implementation_20261024.md)。

2026-10-27 已接入[全过程到达与执行段预测器](docs/full_horizon_predictor_20261027.md)：
完整未来负载情景、阶段／kernel 序列重叠、整请求 FIFO 准入仿真及六项效用评分。
入口为 `python -m gsdeploy.forecasting {forecast,fit,assess}`，只执行 CPU 计算。
真实共驻画像、联合 batch／权重合并后端及 GPU 准确率验证仍待完成。

## 当前四任务实验

当前研究范围为**上下文问答、图像分类、整页 OCR、ASR**。LID、VQA 已退出本轮实验。
七模型及其 16 条任务路径已冻结，配置池采用七模型全池与纯通、纯专结构控制。
质量、吞吐、平均时延、P95、显存、能耗进入六项效用，各占 1/6；取消绝对精度准入。

[`configs/study.json`](configs/study.json) 固定四任务、紧凑模型候选及不可变版本。
它是候选筛选协议，不是已有准入证据的运行 catalog。准备入口校验已有输入的任务、
数据版本及哈希，只冻结四任务清单，不加载 GPU，也不把未知指标补成旧画像：

```bash
python -m gsdeploy.profiling.study --config configs/study.json \
  --source EXISTING_CALIBRATION_DIRECTORY --output NEW_STUDY_DIRECTORY
```

本次静态部署研究使用 160 个唯一输入：QA 32、图像 64、OCR 32、ASR 32。
已记录 61 个成功副本画像和 6 个失败条件；缺测任务×副本状态仍为未知。
H600 校准与 H1800 独立确认分开，实际质量重评 worker 输出，能耗测量整张 GPU。
当前 GPU 串行校准与来源绑定见[GPU 推进记录](docs/static_joint_gpu_progress_20261022.md)。
早期候选筛选、输入准备和适配历史保存在[四任务协议](docs/compact_protocol.md)、
[CPU 准备记录](docs/cpu_preparation.md)及[GPU 实验记录](docs/gpu_experiments.md)。

## 规划与路由

默认联合 MILP 同时选择模型、副本数和任务分流，无须预先固定部署。K=1 为单路径，K>=2 使用分母 1..K 的嵌套分流网格。MILP 以有效成本下界选择候选，统一预测器计算完整效用并反馈；不会把服务时间代理最优当作完整效用最优。搜索、预测与运行共用请求轨迹和路由器，保留质量、实测副本域、显存及 `rho<=0.8` 约束。时间、迭代或列预算耗尽明确报告当前方案和界；有限域最优不代表真机或连续路由最优。

K>=2 默认补充由容量 MILP 求出的可行流量见证，具体点保存到计划；`--no-capacity-witness` 选择纯网格域。新研究 MILP/DFS 使用相同的分母 1..K 路由域与可选见证，首轮均关闭见证。DFS 支持独立完整枚举或有效界剪枝、六项评分及未知证据。共驻画像可通过 `performance_mode=profiled` 按完整部署和路由接入；缺画像保持未知。详见[同域 DFS 与 CPU 基准](docs/static_joint_dfs_benchmark_report_20261021.md)。

```bash
gsdeploy plan --config configs/low18.json --output artifacts/low18.json
gsdeploy plan --config configs/static_replicas.json --output artifacts/static-replicas.json
gsdeploy plan --config configs/asr32b.json --output artifacts/asr.json
python experiments/candidate_pool.py --config configs/holdout.json \
  --output artifacts/pool-comparison.json
```

`configs/catalogs/a.json`、`b.json`、`c.json` 保存三个历史嵌套候选池的单副本冻结参数。`c_replicas.json` 接入同模型已有双/三副本实测参数；`static_replicas.json` 的负载已缩减为四任务，但仍使用旧任务画像，仅作软件与历史基线对照，不能当成新问答/整页 OCR 的小模型池结果。代表任务容量转移仍是规划代理。K=1 的结果只覆盖声明的预测器和有限域，不代表任意连续路由或真实 GPU 上的全局最优。

## Python API

```python
from gsdeploy import DeploymentSystem, SystemConfig

system = DeploymentSystem(SystemConfig.load("configs/low18.json"))
plan = system.plan()
winner = plan["winner"]
prediction = system.predict(winner["deployment"], winner["routing"])
```

原生 Qwen、Phi、MiniCPM、Whisper、分类器和 PARSeq 适配器在 `gsdeploy/backends/`，模型依赖及检查点由独立环境配置。见[原生后端与执行](docs/runtime.md)。

`experiments/native_acceptance.py` 按显式提供的 `--config` 中的已准入路径验收，默认只准备计划和预检报告。原生运行自动检查 GPU，正式效用还需通过检查点和质量记录核验；未提供证据时保留诊断结果。准备中的 `study.json` 不能传给运行接口冒充已准入配置。

新研究使用 `six_term_quality_v1`，T/L/P/M/E/Q 各 1/6，质量进入效用并移除绝对精度门槛。历史 `configs/static_replicas.json` 保留五项语义，demo 能耗为 synthetic fixture。缺质量或整方案能耗时完整 U 为空，不补零、不重新分配权重。见[效用规范](docs/utility_specification.md)。

2026-10-21 的七模型、四任务真实画像已冻结在 `artifacts/static-study-v6-20261021/inputs/`。
2026-10-22 已完成 GPU 校准、七模型联合搜索和 15/15 条独立确认。
选中方案为 MiniCPM-o、图像专用、OCR 专用各一副本；三个确认种子中均高于四个已测竞争方案。
最优证书覆盖冻结预测器及 K8 离散域；实测只覆盖预声明候选，三种子不支持显著性结论。
完整方法、实测表及限制见[研究说明](docs/static_study_methods_and_results_20261022.md)，
带来源凭据的最终汇总见[最终报告](artifacts/static-study-gpu-20261022/study-summary-final-v1/report.zh.md)。
当前运行与命令见[静态工作流](docs/static_study_workflow_v6.md)；
此前的 CPU 阶段记录见[CPU 工作流交付](docs/static_joint_cpu_delivery_20261021.md)。

计划保存时长、种子与到达过程。运行默认继承，显式改变时重新预测并标明原最优结论不再适用。可通过 `plan --solver milp --duration 120 --seed 42 --route-grid 4 --milp-seconds 30 --milp-iterations 1000 --search-policy nominal` 冻结搜索条件。`robust` 则比较同一请求轨迹的 1/1.25/2 倍服务时间。

## 验证与历史

```bash
python scripts/run_cpu_tests.py
```

该入口与 CI 一致，只需基础 Python 依赖和仓库内的 fixture。
需要本地实验产物、额外环境或硬件的研究测试另见[测试说明](tests/README.md)。

CPU 测试验证独立完整枚举一致性、请求守恒、任务顺序、网格嵌套、剪枝、预算状态、能耗缺失规则与真实 mock 子进程。历史输入继续验证合法性；旧分数保留为历史记录，修复后不再复现旧到达流错误。历史实现不会进入运行时依赖。原阶段目录、旧 V3 等基线和精选材料完整保留在 [archive/research-history](https://github.com/DuskTillDawn111/generalist-specialist/tree/archive/research-history)。原研究工作区的权重、数据和原始观测保持原样。

[系统架构](docs/architecture.md) · [迁移与版本边界](docs/history.md) · [源码来源](docs/source_provenance.json)
