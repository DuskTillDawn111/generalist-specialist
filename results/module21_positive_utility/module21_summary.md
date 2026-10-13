# Module 21：Positive-Utility Three-Phase Deployment Validation

**Gate：THREE_PHASE_WITHOUT_UTILITY_GAIN**。Stage A=通过；正式 GPU 运行 6 次。未修改冻结 QPS、预算、模型池、质量准入、Module20 效用权重或归一化参考。

## Stage A 全池复核

枚举全部 432 个测得副本组合；69 个满足预算及六任务覆盖，均参与竞争。全部背景任务路由自由，PARSeq 因 Qwen 无 OCR 准入而由覆盖约束自然必选；Qwen 因 ASR/VQA 覆盖自然必选。没有固定 Text/Image/LID Specialist 驻留。

连续路由求解采用两组预冻结 differential-evolution 种子、全部边界角点与 Powell，再逐部署细化等待分布网格；不同 count 全部保留。数值求解没有全局连续最优证书，不能把“全部部署检查”描述为路由全局最优证明。HIGH 包含 LID S-only 和连续 G/S 竞争，也另外求解允许背景重新路由的 LID S-only。

预测假设为独立 Poisson 到达/随机副本 thinning、确定性 standalone path service×n/g(n)。mean 采用 M/G/1 PK 公式，P95 由 stationary 等待分布数值 CDF 计算。粗/细等待网格 1/0.25ms；这是未经共置校准的 CPU 模型，不能代替 GPU 效用证据。Module17 的局部容量包络未推广。

|阶段|自主部署|LID G 份额|预测 U|配对预测 ΔU|
|---|---|---:|---:|---:|
|LOW|`{"qwen": 1, "roberta_sst2": 1, "convnext_tiny": 1, "baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt": 1}`|1.00000000|-0.46183345|0.015811745473079508|
|MID|`{"qwen": 1, "roberta_sst2": 1, "convnext_tiny": 1, "baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt": 1, "facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a": 1}`|0.00000005|-0.54256507|0.5951394359256217|
|HIGH|`{"qwen": 1, "roberta_sst2": 1, "convnext_tiny": 1, "baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt": 1, "facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a": 1}`|0.08456165|-0.95894977|0.04300361645452233|

Stage A 检查：`{"LOW_no_target_S": true, "LOW_G_only": true, "MID_autonomous_target_S": true, "HIGH_autonomous_mixed": true, "all_predicted_paired_gains_gt_001": true, "coarse_and_fine_phase_same": true, "FFT_truncation_small_at_winners": true}`。CPU 总耗时 526.72s。

效用严格继承：U=−0.25 T_loss−0.30 L_norm−0.30 P_norm−0.15 M_norm。逐任务 mean/P95 用原 offered QPS 和原任务参考归一化；系统 pooled P95 从全部真实请求单独计算。缺失质量不填值；原质量准入为硬约束。持续增长、worker/system unstable 或未完成请求均不允许完成子集延迟构造有效 U。

## GPU 证据

每阶段最多 3 个原定独立 repetition，每个配对共享预生成 Poisson task/sample/offset 轨迹和路由 uniform；运行次序 P/C、C/P、P/C。全部输入与实现 hash 在正式运行前冻结。没有重试、改点或筛选种子。

|阶段|真实 ΔU 三次|mean ΔU|配对 nominal 95% CI|通过|
|---|---|---:|---|---|
|LOW|[-0.11786810859921082, 0.001872671793073044, 0.05954423208537296]|-0.018817068240254937|[-0.24362581707566808, 0.20599168059515818]|False|
|MID|未执行|—|—|因 LOW 未通过而跳过|
|HIGH|未执行|—|—|因 LOW 未通过而跳过|

每组 n=3，置信区间为 df=2 nominal paired t interval，未进行多重比较修正；小样本不支持普遍化。冻结判据要求三次有效 ΔU 均>0.01、方向一致；CI 单独报告小样本不确定性，不新增过关门槛。任何阶段不通过就不继续后续正式阶段。对照 unstable 的 U=null，只计容量可行性差异，不作为合法效用提升。

停止原因：LOW 三次 ΔU 未全部 >0.01，且收益方向不一致；按预冻结规则跳过 MID/HIGH 共 12 次计划运行。

gpu_paired_runs.csv 逐 run 保存各 norm/效用贡献、真实系统 mean/pooled P95、固定/实际 offered QPS、窗口/含 drain 吞吐、完成比例、safe/peak memory、积压及系统/worker 状态。per_task_metrics.csv 严格区分 E2E、worker service 与 queue；worker_stability.csv 保存各 PID 的 busy、队列斜率、积压和稳定性。worker busy 为 handler wall fraction，不冒充 GPU SM；有未完成时它只为完成 handler 的下界。

阶段没有真实观测时不产生收益结论；失败和跳过状态完整保留。

## LOW 真实指标与效用分项

|run|实际 U|offered/completed|系统 mean ms|真实 pooled P95 ms|safe MiB|末尾 backlog|
|---|---:|---:|---:|---:|---:|---:|
|low_proposed_rep1|-1.06673026|1016/1016|33.601|36.978|22750.289|0|
|low_control_rep1|-0.94886215|1016/1016|28.456|36.066|27855.774|0|
|low_control_rep2|-1.06688384|1008/1008|26.452|35.211|28628.928|0|
|low_proposed_rep2|-1.06501116|1008/1008|28.819|36.050|22930.247|0|
|low_proposed_rep3|-1.03694518|1009/1009|28.111|32.467|22743.624|0|
|low_control_rep3|-1.09648942|1009/1009|26.176|33.155|28677.806|0|

|repetition|ΔT contribution|ΔL contribution|ΔP contribution|ΔM contribution|ΔU|
|---|---:|---:|---:|---:|---:|
|1|0.00000000|-0.06086520|-0.08092048|0.02391757|-0.11786811|
|2|0.00000000|-0.01307947|-0.01174436|0.02669650|0.00187267|
|3|0.00000000|0.00490415|0.02684033|0.02779975|0.05954423|

平均效用贡献差：T=0，L=−0.02301351，P=−0.02194150，M=+0.02613794。平均显存收益被延迟成本抵消。

六次系统及全部 27 个 fresh worker PID 均 steady，6,066 个正式请求全部完成；配对 task/sample/arrival offset 一致，所有 E2E 和 service 时间方程以及分项 U 求和核验通过。GPU 已释放，最终 owned/全设备 process list 为空，设备 used=1 MiB。

CI 跨零且 n=3，不支持普遍负收益结论；本轮结论是未通过预定一致正收益判据。MID/HIGH 没有真实 GPU 收益证据。

历史 15281 个文件 hash 未变；6 项 CPU 不变量测试通过。结果与完整部署/连续路由竞争保存在本目录。

完成后停止；未进入 Module22，未搜索其他模型。
