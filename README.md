# Generalist–Specialist

单 GPU 上通用模型与专用模型的联合部署和请求路由系统。输入模型画像、任务流量与显存预算，输出模型/副本部署及路由，并使用同一套运行时记录和评估请求。

```text
configs → DeploymentSystem → optimizer + predictor → plan
                              ↓
samples + workers → runtime → request records → evaluation
```

## 快速开始

Python 3.10+，CPU 规划只需要 NumPy：

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
gsdeploy/       统一 API、配置、优化、预测、路由、执行和评估
configs/        可移植的模型池、工作负载及 worker 配置
examples/       最小完整运行示例
experiments/    按研究问题组织的实验入口
tests/          冻结决策回归、约束与运行时测试
docs/           架构、真实后端与历史版本说明
```

核心包不导入编号实验目录，不从历史 `results/` 动态加载实现，也不依赖固定研究服务器路径。新增算法与功能在 `gsdeploy/` 中开发；实验调用公开接口。

## 规划与路由

默认算法以归一化效用 U 优化部署和任务分配。K=1 使用模型/任务子集缓存与 Pareto 动态规划；固定部署下支持 K>1 的离散分流网格。默认保留经验画像、质量准入、显存和 `rho<=0.8` 约束。

```bash
gsdeploy plan --config configs/low18.json --output artifacts/low18.json
gsdeploy plan --config configs/asr32b.json --output artifacts/asr.json
python experiments/candidate_pool.py --config configs/holdout.json \
  --output artifacts/pool-comparison.json
```

`configs/catalogs/a.json`、`b.json`、`c.json` 保存三个嵌套候选池的已测单副本参数。K=1 的结果只覆盖声明的预测器和有限域，不代表任意连续路由或真实 GPU 上的全局最优。

## Python API

```python
from gsdeploy import DeploymentSystem, SystemConfig

system = DeploymentSystem(SystemConfig.load("configs/low18.json"))
plan = system.plan()
winner = plan["winner"]
prediction = system.predict(winner["deployment"], winner["routing"])
```

原生 Qwen、Phi、MiniCPM、Whisper、分类器和 PARSeq 适配器在 `gsdeploy/backends/`，模型依赖及检查点由独立环境配置。见[原生后端与执行](docs/runtime.md)。

`experiments/native_acceptance.py` 提供六任务验收入口，默认只准备计划和预检报告。原生运行自动检查 GPU，正式效用还需通过检查点和质量记录核验；未提供证据时保留诊断结果。

## 验证与历史

```bash
python -m unittest discover -s tests -v
```

测试用数据验证三个候选池 × 三个历史负载及独立 holdout 的 U 和决策一致性；历史实现不会进入运行时依赖。原阶段目录、旧 V3 等基线和精选材料完整保留在 [archive/research-history](https://github.com/DuskTillDawn111/generalist-specialist/tree/archive/research-history)。原研究工作区的权重、数据和原始观测保持原样。

[系统架构](docs/architecture.md) · [迁移与版本边界](docs/history.md) · [源码来源](docs/source_provenance.json)
