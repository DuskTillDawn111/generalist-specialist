# Module 10B — Specialist exhaustion / Generalist overflow

**完成并停止。Gate：CURRENT_POOL_OPERATIONAL_OVERFLOW_NOT_VALIDATED。** H0 有可重复的 routing latency 收益，但 H1 已出现真实 instability，H2/H3 没有可重复 steady joint extension。

Primary 固定为 ASR / mix_03 / I1 / B1。旧 H0–H3 作废归档内容未变；Module 9 源码、历史与 gate、worker/runtime、模型池、quality、g(x)、memory 和四个 operational H 点均未改变。

C_S^op=3.742602099238，C_A^op=4.347876366682，W^op=0.605274267444 req/s。safe budget=17007.174054952 MiB。

## 执行与完整结果

7 个独立执行 state × 3 independent repetitions = 21 次物理 GPU 测量。RC-SG/v2 在全部四点严格身份相同，审计 deployment、完整 float routing、workload、context/source hashes、协议和三个 seeds 后复用；共享标签不计为独立方法证据。没有复用作废 sweep 的标签，没有结果重跑、额外 profiling 或 fallback。

原协议：warmup 10s、measurement 210s、bounded drain 60s，seeds 7101/7202/7303。各 repetition 重新启动并释放 worker。相同 seed 差值是编号配对，不保证不同 routing 的随机请求轨迹完全相同。

所有进入 GPU 的 worker 均严格满足 rho<0.8，最高预测 rho=0.777181648637；未使用正 capacity tolerance 放行任何测量。原 J optimality certificate 阈值保持 10ns。capacity residual 的单位说明见 certificate_units_note.json。

| 点 | 方法 | ASR req/s | f_S | f_G | max rho | repetitions 的 tier | mean E2E ms | task-weighted P95 ms |
|---|---|---:|---:|---:|---:|---|---:|---:|
| H0 | RC-SG / v2 | 2.994081679 | 86.299521% | 13.700479% | 0.579650252 | S/S/S | 143.25 | 471.62 |
| H0 | V3_MINIMAL | 2.994081679 | 98.965413% | 1.034587% | 0.651766313 | S/S/S | 77.13 | 187.02 |
| H1 | RC-SG / v2 | 3.667750057 | 86.169351% | 13.830649% | 0.689299565 | R/U/U | 2212.82 | 4596.03 |
| H1 | V3_MINIMAL | 3.667750057 | 93.811769% | 6.188231% | 0.742603965 | S/U/U | 1305.48 | 2525.65 |
| H2 | RC-SG / v2 | 3.924184379 | 86.131546% | 13.868454% | 0.731037975 | U/U/U | 1787.14 | 6232.69 |
| H2 | V3_MINIMAL | 3.924184379 | 92.314986% | 7.685014% | 0.777181649 | R/U/U | 1535.23 | 3029.04 |
| H3 | RC-SG / v2 | 4.166294086 | 86.100123% | 13.899877% | 0.770444847 | U/U/U | 6015.04 | 11302.83 |
| H3 | V3_MINIMAL | 4.166294086 | 91.070917% | 8.929083% | 0.809827797 | OPERATIONAL_ROUTE_INFEASIBLE | — | — |

S=steady，R=recoverable_congested，U=unstable。state 汇总沿用三次中最差 tier。mean/P95 为三次等权汇总；P95 是 task-weighted P95，非 pooled quantile。unstable 状态的 latency 仅描述已完成请求与拥塞，不证明 steady 性能优势。

所有方法得到同一 deployment：Phi + distilbert + resnet50 + paddleocr + whisper-small，各 1 replica，即 1G+4S。完整 deployment vector、routing、replicas、每 worker rho、ASR 计划/观测 S/G fraction、每 task mean/P95、completion、backlog、safe/actual memory 与 raw SHA 在 planning_results.csv、gpu_results.csv 和 method_comparison.csv 中。

## 七个重点回答

1. **H0/H1 保持 Specialist-dominant。** V3 f_S 分别 98.9654% / 93.8118%；RC-SG/v2 为 86.2995% / 86.1694%。V3 没有无理由的大量 premature G routing：H0 只给 G 1.0346%，CPU J 略低于全 S 路由；H1 的 6.1882% G 分流同样降低 J。canonical 则因 minmax rho 均衡提前分配约 13.7–13.8% 给更慢 G，其 J 高于全 S 对照，因此不能将 canonical 的早期分流解释为必须的 exhaustion overflow。机制检查保留该区别，未修改路由。

2. **H2 不是首次合法 S+G split。** 两种 routing 在 H0 已有合法 split；lambda_split 的首个已测点为 H0=2.994081679391 req/s。H2 是首次进入已认证的 operational post-exhaustion 区间：lambda_ASR>C_S^op、S-only planning 不可行、ALL planning 可行、primary replica audit PASS。H2 的真实结果仍为 mixed/repeated instability，未达到 steady extension。

3. **H3 canonical 的 G fraction 进一步小幅增加**：H2 13.868454% → H3 13.899877%。V3 H3 的 tentative 8.929083% 不足以满足 operational envelope：该点至少需要约 10.17% 的 G 流量，Whisper rho=0.809827796536。记录 OPERATIONAL_ROUTE_INFEASIBLE，不送入 GPU、不移动 H 点、不切 oracle route。不存在 H3 V3 的真实 overflow/latency 结果。

4. **没有第二 Generalist。** Primary 的安全预算仅允许一个 background-covering deployment；第二 G 无法 fit，frozen v2/V3 的 feasible unvisited candidates 为空。这是该 budget/deployment 空间的结构限制，不是 DeltaJ ranking 将可用 2G candidate 排出 Top-B。此 context 无法验证 multi-G 排序优势。

5. **三种方法的已验证 steady operating range 并列。** 都只有 H0 三次均 steady；都在 H1=3.667750057254 req/s 首次出现 unstable。不存在更晚进入 unstable 的方法。此处是离散冻结 grid 的 first observed breakpoint，不是精确连续 capacity。

6. **未真正验证可重复的 exhausted → steady G overflow extension。** Operational planning 的 exhaustion 条件在 H2/H3 成立，但真实 joint runtime 不能稳定承载。原 rho1 规划 C_Sonly=4.794315526152，所有新点还低于它；没有额外 S-only GPU control，不能声称 physical Specialist saturation 已测得或 G overflow 的独立因果收益已证明。

7. **失败归因：runtime instability + H3 V3 routing 超限；2G 受 deployment/memory 结构限制。** Planning admissibility 未保证真实 steady；ASR/LID/VQA 的 queue 与尾段 backlog 是主要诊断证据。没有证据证明当前六任务池缺少合法 G/S overlap 或正窗口，也不能因此认定必须增加 task。候选排序在唯一 deployment 的 context 未被实际检验。

## Routing 收益与运行失败证据

H0 两组均 3/3 steady，V3 mean 从 143.2485 降到 77.1349ms，task-weighted P95 从 471.6205 降到 187.0207ms；ASR P95 从 2546.4766 降到 697.6490ms。三次同 seed mean/P95 均改善。Deployment 完全相同，因此方法设计差异来自 routing；不能归功于 deployment search 或 candidate ranking。

H1/H2 的 V3 平均 mean/P95 比 canonical 小，但两组汇总都 unstable，不能据此宣称稳定区间扩展。H1 V3 两次、H1 canonical 两次、H2 canonical 三次和 H2 V3 两次均触发 sustained positive tail growth。H3 canonical 三次同样触发尾段增长，第三次另有 incomplete_requests / drain_timeout；completion ratio 为 0.9990387696。

例如 H1 V3 第二次 completion_ratio=1，仍因尾段增长判 unstable：窗口末 90 个 backlog、tail slope=0.3828759 requests/s、ASR/LID P95 约 14.3s。最终全部 drain 完成不能覆盖测量窗口中的不稳定证据。无 OOM 或资源失败被伪装为 routing infeasibility。

Frozen v2 的 accept/rollback、Top-B=2、iteration/evaluation budgets 均沿用原 stack；实际候选集为空，cached exact final state 执行身份通过核验。H3 V3 保留原 frozen J solver 给出的超限路由作为诊断，不改 Module9 或 tolerance。

## 审计与停止

validation_results.json 为 PASS：21 次 valid timed executions、每 state 三次、零 outcome retries、原 41 个作废文件 SHA 未变、Module9/worker/runtime hash 未变、GPU workers 全部释放、MPS 未开启。

gate 与“无 overflow window”不同：W^op>0 且 replica audit PASS，窗口在冻结规划模型中合法；缺失的是稳态 GPU realization。没有启动 task-rich、增加 task、预算、hardware、interference predictor 或修改任何 frozen component。完成后停止。

明细入口：method_comparison.csv/json；gpu_results.csv/json；phase_transitions.csv/json；attribution.json；minimum_overflow_audit.json；state_identity_audit.json；gpu_manifest.json；module10_gate.json。完整 60-context operational envelopes 在 operational_envelope_audit/；旧记录在 capacity_definition_mismatch_audit/。
