# Module 32：Heavy Specialist Positive-Split Feasibility

Gate：**HEAVY_S_INCONCLUSIVE**。正式窗口 0/3；三臂合法稳定=False；匹配轨迹=False；历史保护=True。

候选 1：Qwen2.5-Omni + Whisper-large-v3。Text→RoBERTa SST-2，Image→ConvNeXt tiny，OCR→PARSeq，LID/VQA→Qwen；五个单副本覆盖六任务。Whisper-large-v3 使用已有 Module13B FP32、batch=1、native attention 和生成设置，不新增推理等待或低效参数。所有臂同一运行版本、B3=34567.5534 MiB、效用、模型池和输入。

Whisper-large-v3 WER=0.03948844514247252，目标门槛=0.042181，原始200条质量输出已复算。Module14B 的 FAIL_QUALITY 是相对 Distil baseline 的严格高质量筛选，不能误当目标 ASR 准入失败。候选2仅保留 CPU 配对审查；候选1预检通过后不扩展 GPU 搜索。

|臂|实际G/S ASR|合法U|ΔU vs S-only|状态|
|---|---|---:|---:|---|

效用完整沿用 Module20A：Q/T/L/P/M=0/.25/.30/.30/.15，六任务 offered QPS 加权和原参考时延；原 stability_v2、完整cohort、资源和恢复判据不变。ΔU 只在对应双方合法稳定时计算。原+0.01、G样本≥20、各背景≥72、温度差≤5°C、频率差≤15%探索 Gate 单独审查。


Module31A 在ASR=1QPS、Whisper-small且LID→S时，S执行227.57ms、等待36.06ms；5%/10%分流的同输入S等待减少1.472/4.816ms，G新增执行724.551/736.142ms。本轮更重S、不同ASR QPS及LID→G，因此只作机制对照，不能把跨模块差异归因于模型重量单一因素。


split05 ASR L/P效用变化=0，其余五任务L/P变化=0；T/M差和环境可比性见 pair_comparisons.json。

split10 ASR L/P效用变化=0，其余五任务L/P变化=0；T/M差和环境可比性见 pair_comparisons.json。

Text/Image/OCR/LID/VQA逐任务mean/P95变化完整保存在 per_task_latency_comparison.csv 和 task_utility_changes.csv；共置同输入S服务膨胀、G新增成本见 worker_interference_analysis.csv。温度、频率、功率、显存、CPU和异常逐样本见 GPU_CPU_environment.csv；有共置差异不等于已经唯一识别GPU因果。

Stage B 的40秒真实闭环容量、S/G原生服务mean/P95、背景任务延迟、队列稳定性、观察到的G余量保存在 colocation_profile.json。闭环吞吐是短窗共置观测，G剩余QPS是观测需求的推算，不能宣称长窗容量认证。预检负载选择完全不使用ΔU。

停止原因：AssertionError(('G_SAFE_NOT_VALID', {'valid_session': True, 'full_cohort': True, 'stable': False, 'memory': True, 'journal': True, 'nonblocking': True}))

未形成可接受探索性正例，不提出或实施正式优化器修改；保留所有负结果与不合法窗口。

统计仅为单模型组合、单次匹配轨迹探索。窗内30秒配对区块bootstrap不包含独立跨运行不确定性，不满足原独立holdout支持规则。未宣称独立复现，未执行独立长窗口，完成后停止。

产物：model_pair_review、quality_admission_provenance、colocation_profile、frozen_load、raw_runs、request_journals、system_utility_comparison、native_latency_decomposition、module32_gate，以及冻结manifest和保护审计。
