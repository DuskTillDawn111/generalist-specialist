# 旧版 PerformanceModel / Optimizer 原型（已暂停）

当前正式初始化入口为 [`deployment_core.seed`](../deployment_core/README.md)：通过一次 MILP
联合返回一个 minimum-memory feasible deployment/routing seed。旧 exhaustive enumeration 仅作测试 oracle、
小模型空间 baseline 和 correctness validator，不用于正式研究路径。
本目录 `framework/`、合成性能表及下面的说明是此前原型，保留供参考，不属于当前阶段，
新入口不导入它们。旧实验脚本与数据保持原样。

以下是旧原型说明：

当前研究入口是 `framework/`：给定六任务到达率，枚举合法部署，通过统一
`PerformanceModel.predict(deployment, workload)` 获取性能，再选出满足约束的组合。
本阶段输出部署建议，不启动模型。现有实验脚本保持独立。

## 仓库检查与复用边界

| 已有资源 | 现状与用途 |
| --- | --- |
| `joint_deploy/*_worker.py`、`worker_protocol.py` | 保留现有进程协议；以后 profiling/runtime 可继续使用 |
| `joint_deploy/run_static_hybrid.py` | 已有 `Worker`、`read_jsonl`、worker 环境配置；其 specialist routing 是四任务版本 |
| `joint_deploy/run_active_single_profile.py` | 已有 Qwen 六任务和 Whisper ASR/LID 的服务路径定义 |
| `joint_deploy/run_placement_plan.py` | 已有按 routing fraction 分配请求的实现，但 `TASK_ORDER` 仍为四任务；不能直接执行新六任务方案 |
| `benchmark/task_pools/mixed_formal_1200_6task.jsonl`、`mixed_smoke_30_6task.jsonl` | 后续组合 profiling 复用的样本池；样本数和顺序不能直接解释成到达率 |
| `benchmark/profiles/model_service_profile_6task.json` | 本版直接读取 supported_tasks 和 peak_process_vram_mib，仅选择 Qwen 和四种 specialists |
| `benchmark/results/placement/active_single_v3_current/` | 单路径 worker latency，可用于后续模型特征；不是组合 workload 下的排队响应时间 |
| `placement/exhaustive_placement.py`、`interference_aware_placement_v2.py` | 旧四任务 LP、请求数/makespan 与 gamma 假设耦合；保留为历史实验，不作为新 backend |
| `benchmark/profiles/placement_scenario.json` | workload 表示请求数；不能直接转换为本版 req/s |

六任务 service profile 中旧单任务 service_ms 与 active profiling 的测量口径不同；
本框架没有将它们混合成组合性能。没有修改模型环境、驱动、worker 或已有 benchmark。

## 最小模块

- `framework/schema.py`：六种任务标识，以及 Model、Workload、Deployment、Hardware、PerformanceResult。
  第一版用模型的 `supported_tasks` 表达 capability，不额外引入 Task/Capability 对象层。
- `framework/performance.py`：统一抽象及精确查表 backend。
- `framework/solver.py`：单副本子集枚举、task coverage、显存过滤、性能约束和选择。
- `framework/__main__.py`：复用 profile 的 catalog adapter 与 JSON CLI。
- `examples/`：两组 workload 和明确标记为 synthetic 的性能表，以及其生成脚本。
- `tests/test_framework.py`：纯标准库 CPU smoke/regression tests。

仅依赖 Python 3.10+ 标准库。没有假设解析性能函数已经成立；后续函数模型只需实现相同接口。

## 可运行 smoke

在项目目录执行：

```bash
cd /root/autodl-tmp/generalist-specialist
/root/miniconda3/envs/gsdeploy/bin/python -m unittest discover -s placement/tests -v
/root/miniconda3/envs/gsdeploy/bin/python -m placement.framework \
  --workload placement/examples/workload_mixed.json \
  --table placement/examples/performance_table.synthetic.json \
  --output results/deployment_framework_smoke_mixed.json
/root/miniconda3/envs/gsdeploy/bin/python -m placement.framework \
  --workload placement/examples/workload_text_heavy.json \
  --table placement/examples/performance_table.synthetic.json \
  --output results/deployment_framework_smoke_text.json
```

两个 fixture 总负载均为 29 req/s，各有 16 个覆盖六任务的合法候选。
预设合成数值使 mixed 选择 Qwen + Whisper，text-heavy 选择 Qwen + DistilBERT。
这只验证完整接口与 workload 精确匹配，**不是实验发现，也不是实际部署推荐**。
需要重建 fixture 时运行 `python placement/examples/build_demo_table.py`。

默认最小化 mean latency，同时要求 throughput >= 总到达率。可通过
`--min-throughput-rps` 指定其他阈值；例如观察过载时应明确选择阈值。
最大化吞吐并限制 P95 可使用：

```bash
/root/miniconda3/envs/gsdeploy/bin/python -m placement.framework \
  --workload placement/examples/workload_mixed.json \
  --table placement/examples/performance_table.synthetic.json \
  --objective throughput --max-p95-latency-ms 300
```

同分时按模型数量较少、模型名排序确定性选择。原始性能始终保留，throughput 目标的
score 只是取负数用于排序。未提供 completed_requests/SLO violation 的 backend 保留 null，
不会把“未知”写成 0。空 workload 返回空部署和零服务指标。

状态为 `ok` 时表示已评估全部合法候选；`infeasible` 表示无合法候选或全部已评估候选不满足性能约束。
`incomplete_table` 表示至少一个候选没有精确测量点，默认不返回最优解。
显式传 `--allow-partial` 可选择已覆盖候选中的最优解，输出 `partial` 和缺失组合清单；
这不构成全候选空间最优结论。无解时 CLI 返回退出码 2，有建议时返回 0。
backend 的其他异常不会被当作缺失数据吞掉。

## 数据口径与研究假设

`Workload.arrival_rates` 的单位是 req/s，省略的任务按 0 处理；负数、NaN、Inf、
未知任务都拒绝。精确查表保留完整到达率向量，同 composition 不同总负载是不同点。
当前 fixture 只含两组负载，任意新负载需要补表或接入经过验证的函数 backend。

只给 x 和平均到达率 λ 并不能唯一决定性能：routing、到达过程（包括 burstiness）、
请求样本/长度分布、硬件、batch/concurrency、队列策略和测量窗口都会影响结果。
第一版采用最小假设：**一个 backend/table 固定一套实验上下文**。
表必须记录 `data_kind`、`hardware_id`、`routing_policy`、`protocol`、`latency_definition`。
真实表的 protocol 应注明上述条件及 warmup/drain 处理，建议增加源文件、重复次数、
窗口秒数、SLO 阈值等 provenance。不同条件用不同表；以后研究 routing 时再扩展决策变量。

表 JSON 结构：

```json
{
  "schema_version": 1,
  "context": {
    "data_kind": "synthetic",
    "hardware_id": "demo-only",
    "routing_policy": "specialist-first; qwen fallback",
    "protocol": "synthetic 60-second example",
    "latency_definition": "arrival-to-completion including queue time"
  },
  "rows": [{
    "model_counts": {"qwen": 1},
    "arrival_rates": {"text": 1},
    "performance": {
      "mean_latency_ms": 10,
      "p95_latency_ms": 20,
      "throughput_rps": 1,
      "completed_requests": 60
    }
  }]
}
```

重复 (deployment, workload) 点会拒绝导入，应先按明确口径聚合重复实验。
整体 P95 应从请求级 latency 分布计算，不能平均各任务 P95；均值需按完成请求数加权。
响应时间应包含排队时间，不能直接用旧 `worker_latency_ms` 替代。
真实 throughput 和 completed_requests 必须基于同一明确窗口，记录窗口末未完成/失败请求，
避免只统计成功完成的快请求而忽略积压。有限窗口的 throughput 约束不证明长期队列稳定。

显存过滤暂用现有 profile 峰值之和，默认预算 49140 MiB；可用
`--gpu-memory-mib` 和 `--reserved-memory-mib` 调整。它只是候选筛选，尚不保证共置可加载。
改变预算只过滤候选，不会把表中的性能迁移到另一种 GPU；使用者必须选择对应硬件的表。
每模型利用率、quality、动态副本数、加载开销与在线切换留待后续验证，不声称已实现。

## 后续数据接入

下一阶段沿用现有 worker/task pool，固定 routing 和实验协议，采集 combination × workload：
balanced、六种 task-heavy 与 low/medium/high 总负载。组合需先满足当前 workload 覆盖。
保存请求级 arrival/submit/start/end、完成数、错误数与窗口信息，再生成真实性能表。
已有 standalone/NVML/pairwise 数据可用于函数模型特征和校验；是否选择函数或查表由验证误差决定。
Workload forecasting 在这个映射可靠之后再接入。
