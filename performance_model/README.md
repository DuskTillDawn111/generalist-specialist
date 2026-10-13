# Module 2: Performance model foundation

正式接口 `PerformanceModel.predict(deployment, workload, routing)` 表达 planning predictor
`P_hat(x,r,lambda)`，返回 `PerformancePrediction`：逐任务 mean/P95 latency（ms）、throughput（req/s）
及 system throughput。缺乏数据时为 `None`，并给出 status。接口不接收 deployment ID、在线队列或实时 telemetry。

本版只实现特征、数据审计及未填充曲线的 piecewise baseline。没有训练、accuracy 报告、guided search、
routing optimization、forecasting 或新的 profiling。原 Stage 1/1.5/MILP/measurement 代码不变。

```python
from deployment_core.models import load_model_catalog
from performance_model import FeatureExtractor, PiecewiseLoadLatencyModel
from performance_model.resources import load_resource_profiles

catalog = load_model_catalog()
extractor = FeatureExtractor(catalog, load_resource_profiles(catalog))
model = PiecewiseLoadLatencyModel(extractor)  # no measured controlled load points yet
prediction = model.predict(
    deployment={"qwen": 1},
    workload={"text": 1.0},
    routing={"text": {"qwen": 1.0}},
)
assert prediction.per_task["text"].mean_latency_ms is None
assert prediction.status == "insufficient_profile_data"
```

`deployment` 为非负整数 count 字典；workload 可为 rate 字典或现有 `Workload`，routing 可为嵌套
`task -> model -> req/s` 或 MILP 的 `(task,model) -> req/s`。输入验证包含 capability、deployment linkage、
非负有限 rate 和 workload conservation。不使用 Stage 1 rho 上限拒绝性能模型输入，以便将来描述过载。
特征可记录整数 count；当前 baseline 对 count>1 明确返回 `unsupported_replica_semantics`，不猜测副本负载分配。

## Planning features

- `standalone_service_demand[m] = sum_k r[k,m] * s0[k,m]`。s0 用 Stage 1 三轮 standalone profile 的
  90 样本聚合值（ms 转秒），不是 NVML 或 Nsight 测得的时间，也不是实测 GPU utilization。
- `task_demand_mix[k] = r[k,m]*s0[k,m]/u[m]`，按固定 `TASKS` 顺序。unsupported task 为零，
  idle 模型向量为零且 `task_demand_mix_defined=False`，避免将其视为概率分布。
- 每模型 memory、instance count、standalone service times、deployment 总驻留显存和实例数。
- 离线 resource fingerprint：读取既有 C1 service-path CSV；交叉核对原 NVML summary 和 request identity，
  保留原 Nsight report、CSV、派生脚本及可读小文件 SHA256。旧构建脚本有副作用，因此不 import/执行它。
- `build_colocation_pressure(target_model, deployment, routing, resource_profiles)` 按 fingerprint 字段动态计算
  `sum_{j != m,k} r[k,j]*p[k,j]`。该 helper 接收 normalized nested routing；FeatureExtractor 负责输入格式转换。
  字段数量不写死。任一正负载贡献路径缺失该字段时 `value=None`，另存 `known_partial_sum` 和 `missing_paths`。
  零负载路径不触发 missing；没有竞争负载时 pressure=0。无 resource schema 时返回空字典，不猜测字段。

NVML 是离线设备级短窗指标。`offline_memory_util_pct` 是 NVML memory utilization，不能当作显存容量占比；
`offline_device_vram_used_mib` 是设备整体 VRAM，不能替代 catalog 的模型驻留显存。
pressure 的单位为 fingerprint 单位乘 req/s，例如 W*req/s，不应解释为实际设备功率。
当前11路径上所选14项 resource 值可用，但仍缺进程归因、硬件带宽/occupancy、输入长度分布等维度。

## Piecewise baseline 的边界

`LoadPoint` 保存 model/task、u、feature context、可空 measured metrics 和绝对 source paths。
它仅作为未来人工审计后的实测曲线输入。当前 dataset builder **不自动生成 LoadPoint**。
同路径至少两个 distinct u、context 相容时才线性插值；超范围不外推，重复 load 点要求显式聚合。
context 包括 task mix、workload composition、memory/count features 与 pressure/u，避免把不同混合流量随意连线。
这不是 deployment-ID lookup：不要求该部署组合已有完整 profile，且跨部署泛化能力尚未评估。

多路 task mean 按 routing fraction 聚合；多路 P95 不能从各路径 P95 加权求出，因此返回 null。
未提供 throughput labels 时不使用 arrival rate 冒充 achieved throughput。
单元测试中的手工数值仅为数学 fixture，从不写入真实 dataset/profile points。

## Dataset 与审计

```bash
cd /root/autodl-tmp/generalist-specialist
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m performance_model.dataset
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python -m unittest discover -s performance_model/tests -v
```

输出 `results/performance_model_foundation/`：

- `dataset_rows.jsonl`：一条 run × positive-routing task-model path 一行。
- `dataset_manifest.json`：原结果绝对路径/SHA256、run/seed/协议、请求数、catalog 来源和数据使用规则。
- `feature_coverage.json`：观测 deployment、u/mix 范围、resource 缺失、pressure 完整度和 curve/context 审计。
- `dataset_summary.md`：五个研究问题、近负载的描述性对比及下一轮最小采样建议。

逐路径 labels 从正式 `request_records` 筛选后重新计算，排除 priming/warmup。
原 task/system aggregates 另存为重复 metadata，不可跨 path 行相加。
queue wait、runtime、recent outcome 一律是 label/分析对象，不进入 planning features。
成功 latency 需同时看失败/incomplete/观测数；smoke 保留独立 role，不当作完整 characterization。
未来 split 应至少按 run 分组，跨 deployment 能力需完整 deployment holdout；148 rows 不是148独立实验。

Runtime correction 是未来独立模块，可使用在线 queue/telemetry；本版只提供 planning contract。
