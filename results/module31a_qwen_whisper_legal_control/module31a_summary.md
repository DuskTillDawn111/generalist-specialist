# Module31A Qwen2.5-Omni + Whisper-small 合法同版本对照

最终 Gate：**NO_POSITIVE_FEASIBILITY_CANDIDATE**。正式窗口 3/3；三臂合法稳定=True；同负载=True。

|组|目标/实际分流|G/S ASR|合法 U|ΔU vs A0|状态|
|---|---|---|---:|---:|---|
|A0|0%/0.0000%|0/595|-0.9342740370866206|0.0|VALID / steady|
|A1|5%/4.7059%|28/567|-1.0152123144496383|-0.08093827736301773|VALID / steady|
|A2|10%/8.9076%|53/542|-1.1156757533216823|-0.18140171623506174|VALID / steady|

ASR=1 QPS，其余五任务=.2 QPS，同一 Module30 C1 seed=301001 冻结轨迹；各组完整 600 秒，30 秒预热另计，最长 60 秒排空，五副本预算 34567.5534 MiB。SafeMonitorV2 全臂同版本；初版 NULL 不补算。

历史保护=True。Git commit 不可用（目录无 Git 元数据），源码快照、模型全文件 SHA256、环境/依赖及轨迹均冻结于 experiment_manifest.json；这是明确记录的版本管理限制。

不建议基于此负收益点进入 31B/31C；如将来有新科学问题需另行授权设计。

统计：单次对照不能证明可重复收益；原独立统计支持规则未满足。窗内 30 秒配对区块 bootstrap（2000次）见 statistical_diagnostics.json（仅三臂全合法时生成）。

运行证据：campaign.log、campaign_status.json、smoke_acceptance.json、raw_runs/{SMOKE_A0,SMOKE_A1,SMOKE_A2,A0,A1,A2}.json、request_journals/*.jsonl、run_attempts/*.json、thermal_prelaunch/*.json。完整计量见 per_task_latency_comparison.csv、worker_service_wait.csv、resource_comparison.csv、worker_interference_analysis.csv。

未执行 Stage D 或 31B–31E。

## 主要结果与解释

三组各 600 秒，共 3552 个正式请求，全部完成、失败数 0；每组完整 cohort 吞吐 1.973333 QPS。三臂 journal、时间戳、资源、线程退出、GPU 释放及原始效用复算均通过；全部检查点最终完整 SHA256 再核验通过。

|组|S-ASR 执行/等待均值 ms|G-ASR 执行/等待均值 ms|VQA mean/P95 ms|
|---|---:|---:|---:|
|A0|227.57/36.06|无 G-ASR 请求|200.40/396.57|
|A1|222.49/35.68|947.63/56.11|220.14/486.78|
|A2|223.76/30.53|957.28/66.04|247.14/753.03|

A1 同输入保留 S 等待减少 1.472 ms/请求（负值表示增加），总变化 -0.834 秒；转交 G 相对同输入原 S 增加执行 724.551 ms/请求，共 20.287 秒。五项严格加总重建 ASR mean Δ=29.635 ms，见 native_latency_decomposition.csv。

A2 同输入保留 S 等待减少 4.816 ms/请求（负值表示增加），总变化 -2.610 秒；转交 G 相对同输入原 S 增加执行 736.142 ms/请求，共 39.016 秒。五项严格加总重建 ASR mean Δ=59.498 ms，见 native_latency_decomposition.csv。

基线 S-ASR 服务需求/600 秒=0.2257（服务需求比，非 GPU utilization），排队红利有限。执行差和共享 G 队列的背景尾部代价按数据分开报告；单次序列实验不能唯一隔离 GPU 因果，也不能推广到其他负载点。

完整六任务完成/失败、cohort 与窗口内吞吐、mean/P50/P95/P99/max 见 per_task_latency_comparison.csv；资源均值/峰值见 resource_comparison.csv；配对干扰见 worker_interference_analysis.csv。

## 统计证据

- A1 ΔU 的窗内配对 30 秒区块 bootstrap 95% 诊断区间 [-0.153408, -0.023547]；不含独立跨运行不确定性，不能作为已证实可重复收益的依据。
- A2 ΔU 的窗内配对 30 秒区块 bootstrap 95% 诊断区间 [-0.268083, -0.077432]；不含独立跨运行不确定性，不能作为已证实可重复收益的依据。
