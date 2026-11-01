# 正式确认的最终只读审计清单

本清单审计正式队列 `confirmation-v2-v5-K8/experiment-packet.json`、冻结 v5 报告生成器、独立冻结绘图器和总报告接口的衔接。这里只读已完成证据；本次没有启动 GPU、改变队列或修改任何冻结文件。**正式全量确认尚未完成。**

## 当前已核验的四组

对 seed43 已完成的前四组，实际调用冻结 v5 的 `report_static_confirmation.report_confirmation`，逐组执行 `verify_confirmation_run` 和 `evaluate_run`，结果如下：

| 冻结角色 | seed43 实测 U | 能量与完整六项效用 |
|---|---:|---|
| 自由搜索赢家 / 不分流（同一物理方案） | -0.3105442739182913 | 已核验 |
| 最强其他部署 | -0.32136837807678437 | 已核验 |
| 同部署最强其他路由 | -0.3391766479641359 | 已核验 |
| 纯通预算候选 | -6.064992597753187 | 已核验 |

这是单种子的描述性观测，不能作为完整独立配对结论。纯通角色的冻结搜索状态仍为 `MILP_BUDGET_LIMIT`、`optimality_certified=false`，必须显示为预算候选，不能把实测完成误写成搜索最优证书。

审计副本位于 `artifacts/static-study-gpu-20261022/confirmation-audit-seed43-four/`：含原报告格式的 `report.json`、`report.zh.md`、`audit.json` 和哈希 manifest。该副本明确为 `INCOMPLETE_CONFIRMATION`、完成 4/15、缺失 11 组；没有生成图或替换最终报告。冻结绘图器 `verify_report(..., allow_incomplete=True)` 及总报告 `confirmation_summary` 都已接受该副本。未完成配对没有 Student 区间、符号翻转 p 或 Holm 字段。

## 最终 15 组完成时的必查项

1. **冻结包和候选名册。** 用 v5 `validate_confirmation_packet` 检查 812 个文件绑定、包的 `confirmation_sha256`、源码、任务输入/参考答案、硬件、检查点 receipt、校准版本及冻结搜索。当前包 seal 为 `8ae7689dc0c90f4ebebc6f012e1f4646ad09ee3192095fe411ae1ebcf78eee82`。六个角色映射到五个唯一物理方案，自由搜索与不分流共用一个方案；五方案 × seeds `[43,44,45]` 正好 15 个唯一 job，不能把两个角色重复计为实测或统计重复。
2. **每组身份与完整性。** `job.id == SHA256(job.identity)`，run 的 `experiment_job_id`、`confirmation_packet_sha256`、`executed_measurement_id`、plan、seed、phase=`confirmation`、purpose=`independent_confirmation_not_training` 完整对应。`backend=native`、`status=completed`。执行身份与预声明身份一致；仅允许运行时派生的 worker 环境字段 `GS_CHECKPOINT_FILES_JSON` 在比较时剔除。检查 worker 集合、设备 UUID、设备号、显存容量、GPU preflight、运行环境和驻留模型检查点，不因结果好坏放宽身份。
3. **共同计时与 cohort。** horizon=1800 秒、warmup=30 秒、drain=300 秒、Poisson、各自预声明 seed；同种子各方案 trace 哈希一致。请求记录数等于重建 trace 和窗口 `offered_requests`；逐请求任务、模型 alias、`measurement:i`、scheduled timestamp 与 trace/Router 一致。priming policy 和 size metric 匹配。raw 收集配置只允许门禁声明的 transfer proxy 模式与驻留检查点上下文差异，其余权重、负载、参考值和计划与冻结包相同。
4. **实测门禁。** 逐组再运行 v5 `verify_confirmation_run`，由保存的 request_records 重算延迟、吞吐和质量，不能信任缓存 summary。要求 memory measured/resource_valid、完整 cohort、constraints verified、完整质量身份、measured whole-board energy、`completion_ratio=1`，稳定状态是 steady 或 recoverable_congested。所有组 U 有限且 eligible；质量是六项效用中的连续指标，没有绝对精度阈值。错误列表和 drain 完成情况同时保留。
5. **六项数学。** 记 task 权重 `a_t=lambda_t/sum(lambda)`，检查 `L=sum(a_t*mean_ms_t/ref_t)`、`P=sum(a_t*p95_ms_t/ref_t)`；T=`1-completed_within_window/offered`，drain 内完成仍可使 completion_ratio=1，同时 T>0；M=`safe_memory_mib/gpu_memory_mib`；E=`(run_joules/horizon + startup_joules/amortization)/energy_reference_watts`，当前 amortization=3600 秒、reference=450 W。Q 从实际输出按任务原生错误率重算，再按任务负载权重聚合。六个 normalized 值和 frozen weights 都有限，逐项 `weighted_cost[k]=weight[k]*k_norm`，`U=-fsum(weighted_cost.values())`；不得重新归一化权重或将缺失值当零。
6. **全量报告。** v5 `report_static_confirmation.py` 必须再次原生导入全部 15 份 run，任一重复 job 要显式选定尝试，不能静默覆盖。`completed_jobs=planned_jobs=15`、`missing_job_ids=[]`、`status=ALL_FROZEN_JOBS_VERIFIED`、`paired.all_planned_blocks_complete=true`。所有 run 和 packet 的绝对路径/SHA receipts 仍有效。队列状态只是运行进度，不能替代严格报告。
7. **独立配对重算。** 四个不同竞争方案各有 seeds43/44/45 三个配对差 `d_s=U_selected,s-U_rival,s`，保留原始差、均值、standard_error=`stdev(d)/sqrt(3)`、95% Student 区间 `mean ± t_0.975,2*SE`。枚举全部 8 种 ± 符号计算精确双侧 p，再按四个 rival 做 Holm。请求数不是重复数；三个种子最小双侧符号翻转 p=0.25，不能声称 p<0.05。最终审计应独立核算 Student/Holm；绘图器只核算 receipts/效用，总报告核算差值和精确 p，二者均不能代替该额外统计核算。
8. **绘图与总汇总。** 只将完整严格 report 传给 `confirmation-plot-source-v1/scripts/plot_static_confirmation.py`，不加 `--allow-incomplete`。输出 utility、六项 costs、selected deployment/routing 三组 PNG/SVG/PDF、candidate-roster 和 receipt manifest。图中六角色完整保留，五物理方案不重复作统计单位，纯通仍显示其未认证搜索状态。总报告使用同一 K8 `--search-dir`、同一 calibration/validated 报告和效率结果，传入 `--confirmation-report`；不能用后续新候选替换已冻结确认名册。最后视觉检查图表和复核输入/输出 manifest。

## 已核验的衔接边界

- 正式推进脚本在队列成功后调用冻结 v5 `scripts/report_static_confirmation.py` 生成 `confirmation-report-v1`，然后调用独立冻结绘图快照生成 `confirmation-figures-v1`；均不会修改既有输出目录。
- 绘图器验证 packet seal、候选/job 名册、原始 run receipts 与执行身份、六项效用算术；不重新原生导入、不重算 Student/Holm。因此正式上游报告必须先通过原生导入。
- `scripts/report_static_study.py` 的 `confirmation_summary` 已用本次四组真实 partial 副本验证通过，可把最终报告接入综合汇总；该汇总也保留纯通预算状态和预测/实测的区别。
- 即使最终 15/15 确认成功，也只支持已冻结五方案、固定输入池和三个种子的实测比较。有限预测域搜索证书、实际候选间效用排序、物理全组合最优三者必须分别陈述。
