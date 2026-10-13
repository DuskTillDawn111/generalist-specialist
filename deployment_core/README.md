# Module 7C-2：经验 replica capacity 与 memory

CPU planning 支持 `capacity_mode="linear"` 和 `capacity_mode="empirical"`。
`ReplicaPlanningPolicy` 从冻结 `results/replica_scaling_real/` 读取完整精度
`g_m(2)` 和双副本 idle memory，验证 aggregate、plan 和 raw repetitions 的 SHA256。
`g(0)=0, g(1)=1`；仅 Qwen、Whisper-small、ResNet50 有 g(2) 证据。
其余 types 保持 scaling pending，real empirical planning 的 evidence bound 为 1。
最终 R=min(memory physical bound, evidence bound, 实际 worker/process bound)。

Empirical MILP 引入逐模型逐 count 的 binary one-hot `z[m,n]`：
`sum_n z[m,n]=1, x_m=sum_n n*z[m,n], y_m=1-z[m,0]`。
Capacity 为 `D_m<=rho_max*sum_n g_m(n)*z[m,n]`，memory 为
`sum_mn M_m(n)*z[m,n]<=V_GPU`；routing linkage 和 coverage 继续使用 presence。
所有约束仍是线性。Seeder 的 memory/count lexicographic 目标不变。
返回 witness 包括 count selection、effective capacity 和每模型 memory。

`M(1)` 锚定 frozen candidate-pool 实测 single idle memory，保留 binary 行为；
已测 `M(2)` 来自 7C-1 实测双副本 post-warm idle mean。未测 count 不外推。
两种 memory mode（`additive`、`empirical`）独立于 capacity mode，便于控制变量对照。
所有 R=1 时，empirical 分支复用旧 binary seed / canonical LP / traces。

```python
from deployment_core.replica_calibration import load_replica_calibration
calibration = load_replica_calibration(catalog)
seed = MilpDeploymentSeeder().solve(
    catalog, workload, hardware, capacity_mode="empirical", calibration=calibration,
)
```

这是 **planning memory surrogate** 和经验 family-capacity surrogate。
Qwen/Whisper 的 g(2) 证据来自 ASR；跨 task/mix 的使用是 transfer surrogate，
未验证异构共驻留 interference、peak memory 或任意 workload scaling。
Linear 模式继续用于 synthetic / baseline，不被删除。
本阶段没有实现/启动新的真实 worker 或真实 integer search。
完整 CPU 产物见 `results/replica_capacity_calibration/`。

# Module 7B：整数 replica-count seed

正式整数入口为 `MilpDeploymentSeeder.solve(..., binary_mode=False)`，或显式传入
`replica_bounds`。部署字典 `dict[str,int]` 保存非负整数实例数；默认上界为
`floor(V_GPU/V_m)`。candidate pool v2 的 `V_m` 读取实测
`measured_idle_vram_mib`，不使用 weight size，也不沿用下文历史 peak-memory 口径。

整数 MILP 增加 binary presence `y_m`，以 `y_m<=x_m<=R_m*y_m` 连接。
coverage 使用 `sum c_km*y_m>=1`，routing linkage 使用
`r_km<=lambda_k*c_km*y_m`；memory 使用 `sum V_m*x_m<=V_GPU`，capacity 使用
`sum r_km*s0_km<=rho_max*x_m`。先最小化 memory，再固定其最优值最小化 replica count。
`IntegerSeedResult` 序列化 presence、bounds、secondary objective 和两次 MILP 调用计数。
整数 feedback 的 `utilization[m]` 是 active family 的 `D_m/x_m`，slack 是
`rho_max*x_m-D_m`；task/model demand 保持 aggregate family 语义。

Memory 是 **planning memory surrogate**；capacity 是 analytical planning
surrogate, **not a claim of linear real-GPU scaling**。没有实现真实多副本 worker。
完整 search/API/CPU 验证见 [Module 7B 文档](../deployment_search/INTEGER_REPLICAS.md)。
Module 7B 当时缺失的 14 条 Stage-1 paths 已由 Module 7A-2 真实 standalone measurement 补齐；历史 blocked 报告保留，当前完整 v2 dry-run 已解锁。

不传整数参数的旧调用，以及所有 `R_m=1` 的兼容分支，保留原 binary solve 和
单 memory objective，以复现 Module 3/5 行为。下文为这些历史接口与 Stage 1 数据说明。

# Module 3 binary feasible deployment seed（兼容接口）

当前正式初始化接口为 `MilpDeploymentSeeder.solve(catalog, workload, hardware, rho_max=0.8)`。
从模型集合、workload 与硬件预算，一次 MILP 联合求出一个 `(x0, r0)`，不生成 `X_feasible`。
这是 **minimum-memory feasible seed**，仅用于可行初始化，不是最终性能最优 deployment。
Stage 2A / 2B measurement 不变；本步骤不运行性能 sweep，也不实现 predictor 或 guided search。

```bash
cd /root/autodl-tmp/generalist-specialist
/root/miniconda3/envs/gsdeploy/bin/python -m deployment_core.seed \
  --workload text=10,image=5,ocr=3,asr=2,visual_qa=1,lid=1 \
  --rho-max 0.8 --output results/deployment_seed/example_seed.json
```

```python
from deployment_core import MilpDeploymentSeeder, Workload, load_model_catalog
from deployment_core.models import load_hardware

seed = MilpDeploymentSeeder().solve(
    load_model_catalog(),
    Workload({"text": 10, "image": 5, "ocr": 3, "asr": 2, "visual_qa": 1, "lid": 1}),
    load_hardware(),
    rho_max=0.8,
)
# seed.deployment: model -> integer count, including zero counts
# seed.routing_rps: (task, model) -> req/s, including all zero pairs
```

## 数学与实现

一般变量 `x_m ∈ Z_≥0` 表示实例数；当前 prototype 的整数变量上下界为 `[0,1]`。
`r_{k,m} ≥ 0` 的单位为 req/s。返回 count 字典，不使用旧 `Deployment` 子集对象；
将来扩展 replica 时需要另行验证实例聚合语义和 runtime，本版不实现多副本。

| 条件 | MILP |
| --- | --- |
| Coverage | 对正负载 task，Σ c_km x_m ≥ 1 |
| Memory | Σ V_m x_m ≤ V_GPU |
| Conservation | Σ r_km = λ_k |
| Capability / linkage | 0 ≤ r_km ≤ λ_k c_km x_m |
| Capacity | Σ r_km s0_km ≤ rho_max x_m |
| Objective | min Σ V_m x_m，单位 MiB |

显存和 s0 直接复用下文 Stage 1 catalog 的原始测量及来源哈希，没有硬编码近似值。
MILP 内仅为 active/supported pairs 建 r 变量；返回所有 task×model pair，缺省路径显式为零。
守恒和 linkage 约束按 λ 缩放以改善小负载数值精度，变量本身仍是 req/s。
矩阵以 sparse COO/CSC 构建，大小随模型与有效 task/model pair 数量增长，不创建 deployment 组合。

复用现有 SciPy 1.17.1 的 `scipy.optimize.milp` / HiGHS，`mip_rel_gap=0`。
不使用 epsilon；同显存目标允许任一可行 seed。排序变量保证稳定构建，不承诺跨 solver 版本
同值解完全相同。没有 latency、throughput、queue delay 目标项。

## 返回与 bottleneck feedback

`SeedResult` 保存 `feasible`、`deployment`、`routing_rps`、`objective_value`、
`solver_status`、`solver_time_ms`，以及：

- `service_demand_by_task_model[(k,m)] = r_km * s0_km`（s0 从 ms 转成秒）；
- `utilization[m] = Σ_k D_km`；
- `capacity_slack[m] = rho_max*x_m - utilization[m]`。

以上都是 standalone service-demand surrogate，**不是 measured GPU utilization**。
零实例模型的 utilization/slack/contributions 都为零。
返回前独立复算 coverage/memory/conservation/linkage/capacity、目标值和 feedback，
沿用 Stage 1 的 conservation 相对容差 `1e-7`、capacity 绝对容差 `1e-9`。
浮点误差量级（例如 `-1e-16`）的 slack 原样保留，不掩盖边界。

只有 solver 明确报告 infeasible 才返回 `feasible=False`、deployment/routing/objective 为 `None`。
solver limit、数值失败、无效 witness 抛出 `SeedSolverError`，不会伪装为不可行。
CLI 还支持 `--gpu-memory-mib`、`--catalog-sources`；有效的不可行结果退出码为 0，错误为 2。

全零 workload 时新 formulation 允许全零 deployment，目标为零；旧 baseline 为历史兼容仅枚举非空组合。
本次 19 个已有场景均为正负载，这个边界差异另有测试覆盖。

## 验证与产物

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python \
  -m deployment_core.experiments.validate_seeder
OMP_NUM_THREADS=1 /root/miniconda3/envs/gsdeploy/bin/python \
  -m unittest discover -s deployment_core/tests -v
```

`results/deployment_seed/` 保存 `stage1_5_milp_validation.json`、`synthetic_scalability.json`、
`example_seed.json`、`summary.md`。19 个原场景均分别运行 MILP 和 exhaustive oracle；
核对 feasible/infeasible、集合归属、最小显存目标，以及复用 Stage 1.5 的 routing 校验。
20/30 synthetic models 使用会抛错的 enumeration/旧 Deployment 构造 guard，确认没有枚举调用，
记录求解时间和 MILP 大小。这只是合成 correctness/scalability 测试，不是真实 serving 性能数据。

## 保留的 exhaustive baseline

`generate_feasible_deployments` / `screen_deployments` / `python -m deployment_core.solve`：

```text
Exhaustive feasibility enumeration.

Purpose:
- unit-test oracle
- small-model-space baseline
- validation of MILP seeder

Not used by the production/research search path.
```

旧接口及枚举实现保留；包级旧接口采用惰性导入，正式 seed 入口不导入 enumeration 模块。
下文是已验证 Stage 1 baseline 的定义、来源和历史使用说明，不能视为正式 search path。

---

# Stage 1 baseline：Deployment 与 Routing 可行性筛选

给定模型池、六任务到达率和单 GPU 显存预算，返回所有满足 Coverage、Memory、
Standalone Capacity 的 `(deployment, feasible routing)`。当前五个模型完整枚举
31 个非空组合；不选优、不排名，不加载模型或启动 benchmark。

## 运行

依赖当前主环境已有的 NumPy、SciPy（已验证 SciPy 1.17.1 / HiGHS），无需安装或修改环境。

```bash
cd /root/autodl-tmp/generalist-specialist
/root/miniconda3/envs/gsdeploy/bin/python -m deployment_core.solve \
  --workload text=10,image=5,ocr=3,asr=2,visual_qa=1,lid=1 \
  --output results/feasibility_stage1_smoke.json
```

`--rho-max` 默认 0.8，允许 `(0, 1]`；`--gpu-memory-mib` 可覆盖默认实测显存预算。
省略任务的 rate 按 0 处理，单位为 req/s。重复/未知任务、负值和非有限值会报错。
`--catalog-sources` 可选择另一份明确指定数据来源的 manifest。

CLI 输出所有可行组合、非零 routing、各已部署模型 utilization，并列出每个被淘汰组合的原因。
JSON 额外保存零值的有效 routing pair、完整 catalog、原始数据来源与哈希。
所有未列出的 `(task, model)` routing 为零。按模型数量、模型名称排序，仅用于稳定展示。
LP 可能存在多个 witness，具体 routing 不保证跨 SciPy/HiGHS 版本相同。

退出码：完成枚举返回 0（包括可行集合为空）；输入、数据或求解器错误返回 2。
零 workload 仍枚举全部非空组合，routing/utilization 为零；不额外引入空部署。

## 实测数据复用与来源

`catalog_sources.json` 仅记录文件位置，不复制提示词中的近似数值。
所有相对路径相对于该 manifest 所在目录解析。

**Capability** 从 `benchmark/profiles/model_service_profile_6task.json` 的
`models.*.supported_tasks` 读取，与现有 `joint_deploy/run_active_single_profile.py` 的
`MODEL_TASKS` 一致。选择 Qwen、DistilBERT、ResNet50、PaddleOCR、Whisper。
旧 LP/routing 脚本多数只有四任务，且目标是 makespan 或性能选择，所以新求解器独立实现。

**Memory** 从 `benchmark/results/system_snapshot/` 原始成功 NVML 记录的
`peak_target_process_memory_mib` 读取，交叉核对六任务 metadata 中的
`peak_process_vram_mib`，不一致即报错：

| Model | 实测 MiB | NVML 文件 |
| --- | ---: | --- |
| qwen | 17574 | nvml_qwen_sst2.json |
| distilbert | 750 | nvml_distilbert_sst2_pynvml.json |
| resnet50 | 604 | nvml_resnet50_imagenette.json |
| paddleocr | 582 | nvml_paddleocr_iiit5k.json |
| whisper | 1626 | nvml_whisper_small_librispeech.json |

总计 21136 MiB。硬件预算从 Qwen snapshot 的 `gpu_baseline.total_mib` 读取，为 49140 MiB。
这是已存 snapshot，不代表查询了当前空闲显存；CLI override 表示本次分配的预算。

**Standalone service time** 使用 `benchmark/results/placement/` 下三轮完整矩阵：

- `active_single_v3_current/`
- `active_single_v3_r2/`
- `active_single_v3_r3/`

每轮每条路径有 30 个样本，总计 90。要求 summary 的 protocol 为
`active_single_profile_v1`，用 `target_worker`、`target_task` 定位唯一记录。
采用 `sum(mean_worker_latency_ms * target_count) / sum(target_count)`，聚合值如下
（本表仅展示，程序每次读取源文件计算）：

| Service path | 平均 ms/request |
| --- | ---: |
| qwen/text | 44.638714 |
| qwen/image | 139.613190 |
| qwen/ocr | 94.648199 |
| qwen/asr | 654.078457 |
| qwen/visual_qa | 144.507448 |
| qwen/lid | 139.909434 |
| distilbert/text | 3.769309 |
| resnet50/image | 5.497822 |
| paddleocr/ocr | 6.067525 |
| whisper/asr | 190.165871 |
| whisper/lid | 27.084135 |

旧六任务 metadata 中的 `service_ms` 来自另一种单任务测量口径，这里仅复用其 capability、
显存交叉核对信息，不混用那组延迟。没有读取 pair/interference 系数或 Nsight 下的耗时。
缺少路径、重复 summary、协议不符、样本数无效会明确报错，不猜测缺失值。

## 数学约束与代码

| 数学条件 | 实现 |
| --- | --- |
| `x_m ∈ {0,1}`，非空 | `Deployment` 拒绝重复模型，`enumerate_deployments` 使用全部非空子集 |
| `λ_k > 0 ⇒ Σ c_km x_m ≥ 1` | `candidate_generator.missing_coverage` 仅检查正负载任务 |
| `Σ V_m x_m ≤ V_GPU` | `deployment_memory_mib` 求和后与硬件 MiB 预算比较 |
| `Σ r_km = λ_k` | `routing_solver.solve_capacity_lp` 为每个 active task 建立等式行 |
| `0 ≤ r_km ≤ λ_k c_km x_m` | 只创建已部署且支持任务的 pair，bounds 为 `(0, λ_k)`，其他 pair 隐式为零 |
| `Σ r_km s⁰_km ≤ ρ_max` | 每个已部署模型建立不等式行，系数为 `service_time_ms / 1000` |
| LP feasibility | `linprog(method="highs")`，全零 objective，无延迟目标 |

HiGHS infeasible（status 2）才记为 Capacity 淘汰。求解器数值故障等其他失败单独报错。
成功解去除浮点边界噪声后重新校验 routing 守恒及 utilization。LP 容差为 `1e-9`，
守恒复验使用 `1e-7` 相对容差；极端数量级下若解不能通过复验会报错，不输出不可信 witness。

筛选记录采用**顺序淘汰**：先 Coverage，再 Memory，最后 Capacity。一个组合只记录首次
淘汰原因；`coverage_rejected + memory_rejected + capacity_rejected + capacity_feasible = 31`。
内存通过数只统计 Coverage 已通过的组合，Capacity 通过数只统计前两层都通过的组合。

## Python API

```python
from deployment_core import Workload, generate_feasible_deployments, load_model_catalog
from deployment_core.models import load_hardware

feasible = generate_feasible_deployments(
    load_model_catalog(),
    Workload({"text": 10, "visual_qa": 1}),
    load_hardware(),
    rho_max=0.8,
)
for row in feasible:
    print(row.deployment.models, row.routing.routing_rps, row.routing.utilization)
```

需要逐阶段计数和淘汰原因时使用同参数的 `screen_deployments(...)`，返回
`ScreeningReport`，包含 `counts`、`feasible_deployments`、`rejected_deployments`。
`generate_feasible_deployments(...)` 返回普通 `list[FeasibleDeployment]`。

## 验证结果

```bash
cd /root/autodl-tmp/generalist-specialist
/root/miniconda3/envs/gsdeploy/bin/python -m unittest discover -s deployment_core/tests -v
```

16 项测试通过，包括：31 非空组合、text-only、VQA 必须有 Qwen、ASR 过载、显存边界、
拆分 routing、不同 task 的不同 service demand、可配置 rho、零 workload、数据来源、
非法输入、求解器异常、零目标及完整 CLI 输出。测试对返回的 routing 独立复算请求守恒和利用率。

真实 profile + `rho_max=0.8` 的 smoke 结果：

| Workload | 全部 | Coverage 后 | Memory 后 | Capacity 后 |
| --- | ---: | ---: | ---: | ---: |
| text=10 | 31 | 24 | 24 | 24 |
| visual_qa=1 | 31 | 16 | 16 | 16 |
| asr=2 | 31 | 24 | 24 | 16 |
| asr=100 | 31 | 24 | 24 | 0 |
| text=10，预算 750 MiB | 31 | 24 | 1 | 1 |
| 六任务示例：10,5,3,2,1,1 | 31 | 16 | 16 | 3 |

六任务示例返回三个组合，均保存一组可行 routing：

1. distilbert + qwen + resnet50 + whisper
2. paddleocr + qwen + resnet50 + whisper
3. distilbert + paddleocr + qwen + resnet50 + whisper

该 workload 下 `qwen + whisper` 没有通过 Capacity，CLI 示例的排版不作为可行性假设。

## 当前边界与文件

新模块只做 standalone-demand feasibility screening。均值与线性 capacity 假设未验证真实共置
承载能力，不涵盖 interference、排队响应时间、batching、调度开销、功率或传输效应。
现有 NVML 显存值来自特定独立运行样本，也不是所有任务/输入长度的显存上界。
这些都是当前输入 profile 的适用边界，不把输出宣称为真机服务保证。

新增文件分工：`models.py`/`catalog_sources.json` 管数据与来源；`workload.py` 管到达率；
`deployment.py` 管部署和 witness；`routing_solver.py` 管 Capacity LP；
`candidate_generator.py` 管枚举和三层过滤；`solve.py` 管 CLI/JSON；`tests/` 管验证。

旧 `placement/framework/` 是此前包含性能模型与选择的暂停原型，新模块没有对它的依赖。
仅更新 `placement/README.md` 的入口说明，未修改旧 worker、benchmark、profiling 数据、
输出格式或 CUDA/PyTorch/驱动环境。
