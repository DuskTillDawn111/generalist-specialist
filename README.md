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
tests/          独立穷举、约束与运行时测试
docs/           架构、真实后端与历史版本说明
```

核心包不导入编号实验目录，不从历史 `results/` 动态加载实现，也不依赖固定研究服务器路径。新增算法与功能在 `gsdeploy/` 中开发；实验调用公开接口。

## 规划与路由

联合搜索同时选择模型、副本数和任务分流，无须预先固定部署。K=1 为单路径，K>=2 使用包含旧网格点的分流域，并补充容量边界见证。搜索、预测与运行使用同一请求轨迹和路由器，保留质量、实测副本域、显存及 `rho<=0.8` 约束。搜索预算耗尽会明确报告，有限域最优不代表真机或连续路由最优。

```bash
gsdeploy plan --config configs/low18.json --output artifacts/low18.json
gsdeploy plan --config configs/static_replicas.json --output artifacts/static-replicas.json
gsdeploy plan --config configs/asr32b.json --output artifacts/asr.json
python experiments/candidate_pool.py --config configs/holdout.json \
  --output artifacts/pool-comparison.json
```

`configs/catalogs/a.json`、`b.json`、`c.json` 保存三个嵌套候选池的单副本冻结参数。`c_replicas.json` 接入同模型已有双/三副本实测参数，供当前静态实验使用；代表任务容量转移仍是规划代理。K=1 的结果只覆盖声明的预测器和有限域，不代表任意连续路由或真实 GPU 上的全局最优。

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

当前静态配置使用 T/L/P/M/E 各 0.2，详见[效用规范](docs/utility_specification.md)。缺少整方案能耗时完整 U 为空，搜索只提供明确标记的诊断候选；不补零、不重新分配权重。demo 的能耗是 CPU synthetic fixture，历史配置显式保留四项权重。

计划保存时长、种子与到达过程。运行默认继承，显式改变时重新预测并标明原最优结论不再适用。可通过 `plan --duration 120 --seed 42 --route-grid 4 --max-states 100000 --search-policy nominal` 冻结搜索条件。`robust` 则比较同一请求轨迹的 1/1.25/2 倍服务时间。

## 验证与历史

```bash
python -m unittest discover -s tests -v
```

CPU 测试验证独立完整枚举一致性、请求守恒、任务顺序、网格嵌套、剪枝、预算状态、能耗缺失规则与真实 mock 子进程。历史输入继续验证合法性；旧分数保留为历史记录，修复后不再复现旧到达流错误。历史实现不会进入运行时依赖。原阶段目录、旧 V3 等基线和精选材料完整保留在 [archive/research-history](https://github.com/DuskTillDawn111/generalist-specialist/tree/archive/research-history)。原研究工作区的权重、数据和原始观测保持原样。

[系统架构](docs/architecture.md) · [迁移与版本边界](docs/history.md) · [源码来源](docs/source_provenance.json)
