# Current status after Module 7A-2

14 个 pending Stage-1 demands 已由真实三轮 standalone measurement 补齐，25/25 verified paths 完整。W1/W2/W3 12-model integer dry-run 已解除 blocked，使用 frozen idle VRAM + new frozen real s0；验收第 14 项已通过。详细报告见 `../candidate_pool_v2_service_demand/summary.md`。Module 7B 算法和此前 synthetic campaigns 未修改或重跑；以下保留当时的历史状态。

---

# Module 7B Integer Replica-count Optimization Foundation

整数 formulation/search 与 synthetic validation 已完成。**W1/W2/W3 的完整 12-model dry-run 被缺失的 frozen Stage-1 service demand 阻塞**；按用户选择保留 pending，不补值，不把 sanity timing 或不同口径的 formal E2E timing 当作 s0。

## Formulation

`x_m` 是非负整数 replica count；`y_m` 是 binary presence。MILP 使用 `y<=x<=R*y`，coverage 与 routing linkage 均使用 y；memory 使用 `sum V*x<=49140`，capacity 使用 `sum r*s0<=rho_max*x`。integer seeder 先最小化 memory，再在相同 memory optimum 下最小化 replica count，两个 MILP solve 均计入计数。

固定 x 的 canonical LP 最小化 `z`，约束 `D_m<=z*x_m`；反馈使用 active families 的 `D_m/x_m`。`D[k,m]=r[k,m]*s0[k,m]` 原样保留。routing 保持 family-level，future replica IDs 仅是 schema，没有启动 workers。

Memory 是 **planning memory surrogate**；capacity 是 **analytical planning surrogate, not a claim of linear real-GPU scaling**。Module 7C 尚未进行。

每轮最多 2M 个 x±1 mutation，bottleneck 的 Generalist SCALE_UP 与承担 pressure tasks 的 Specialist/alternative SCALE_UP 一起按 delta_z 排序。允许 SCALE_DOWN；runtime acceptance 继续使用既有 RuntimeObjective（stability→mean→p95→memory tie-break），没有 latency+alpha*memory 加权和。

## Synthetic validation

- A：MILP 返回 G=0/S=2；search 接受 S:1→2。单个 specialist 在该 synthetic load 下不足，两个满足 surrogate。
- B：分别存在 Specialist 的 delta_z 更大、Generalist 的 delta_z 更大，以及 Generalist 增副本被 memory 阻塞的情况；无固定 Specialist 优先规则。
- C：G=1/S1=2/S2=1 可降为 S1=1，并记录 2 MiB synthetic resource saving；mean/p95 相同，既有 memory tie-break 接受。
- D：G replica、S replica 与 alternative type 均产生候选，后续 memory 拒绝证明不能无限增加。

| Model types | Theoretical vectors | Per-round mutations | LP solves | Runtime evaluations | Visited states |
|---|---:|---|---:|---:|---:|
| 12 | 6049137024 | [13, 14, 14, 14, 14] (≤24) | 53 | 11 / 17 | 11 |
| 20 | 13829198312595456 | [21, 22, 22, 22, 22] (≤40) | 85 | 11 / 17 | 11 |
| 30 | 1549161398610361081921536 | [31, 32, 32, 32, 32] (≤60) | 125 | 11 / 17 | 11 |

仅小型 oracle 枚举 27 个 3-type/R≤2 vectors；J*=21.2308，J_search=22.8333，gap=0.0754831。正式 IntegerGuidedDeploymentSearch 不导入或调用 oracle，12/20/30 search 带 enumeration tripwire 运行。

## Binary compatibility

5 个 captured pre-change traces（3/5-type synthetic 与 W1/W2/W3 原 pool）均与修改后 legacy path、integer R=1 path 完整相同，仅忽略 solver wall time。R=1 模式明确复用原 binary seed/neighborhood/key/routing/search，以保留 Module 3/5 行为与旧序列化。该兼容模式保留旧单目标 seed；genuine integer mode 使用两阶段 lexicographic seed。

## Frozen pool bounds and missing service matrix

12 个 model 的 R 均来自 `floor(49140/measured_idle_vram_mib)`，不读 weight_size_gb，没有任意 max_replicas=2。synthetic small oracle 的 R≤2 是显式实验设置。实际未确立额外物理 process limit；旧 serving implementation 的 count>1 guard 保持不变，不冒充物理 planning limit。

当前 pool 的 25 个 verified task/model capability pairs 中，11 个有原 Stage-1 standalone worker s0，14 个 pending，涉及 7 个 model types：新增 5 个 specialists，以及只有不同口径 formal benchmark 的 Phi/MiniCPM。capability_verified 与 service_demand_available 是两个独立状态；没有降低 capability 或排除 frozen candidates 来伪造完整 dry-run。

W1/W2/W3 文件已生成 blocked record：seed、memory usage、pressure、bottleneck、candidates、Top-B、N_G/N_S 均为 null，而不是虚构结果或把 5-model diagnostic 冒充 12-model planning。MILP/LP/runtime/GPU calls 均为 0。

三个 workload 的 visual_qa rate 都大于零，唯一 capable model 为 qwen，故 **mandatory model induced by capability coverage: y_qwen=1**。Generalist composition 的自由度受 VQA capability asymmetry 限制；这不是算法自由选择了 Qwen。

## Acceptance and stopping

验收 1–13、15–17 在代码与 synthetic validation 中实现/验证；验收 14（完整 W1/W2/W3 12-model optimization dry-run）为 **blocked**，不能声明全部验收通过。N_G/N_S 对每个可计算 synthetic state 自动输出；blocked workload state 没有计数向量，因此计数为 null。

冻结 pool/workload 文件 hash 全部保持不变；未安装/更改环境，未运行 GPU measurement、real multi-replica serving、final benchmark、forecasting、routing reoptimization、dynamic campaign 或 predictor retraining。

源码入口：`deployment_core.replicas`、`MilpDeploymentSeeder.solve(..., binary_mode=False)` 或显式 replica_bounds；`IntegerGuidedDeploymentSearch`。validation 命令：`envs/data_env/bin/python -m deployment_search.integer_foundation`。

## CPU regression verification

Core 42 项、search 71 项（包含 14 项新增整数测试）全部通过，共 113 项。详见 `cpu_regression_validation.json` 和对应日志；源码 hash 记录于 `source_hashes.json`。
