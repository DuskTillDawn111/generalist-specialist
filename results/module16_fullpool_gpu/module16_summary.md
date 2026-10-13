# Module 16：Full-Pool Three-Phase GPU Validation

Gate：**FULLPOOL_GPU_REJECTED**。完成后停止。

固定预算27980.1894 MiB；text=1.1664,image=0.3729,ocr=0.5501,lid=12.0,visual_qa=0.3603 QPS；ASR LOW=0.12,MID=1.00,HIGH=3.80。Qwen/Phi/MiniCPM和14F完整Heavy pool全部参与CPU竞争，无强制模型；quality、真实实测副本表、safe-factor、rho≤0.8与J目标不变，无人为延迟SLO。

资源优先：全池MILP最小化safe residency，并枚举全部同等最小资源state后按原J选择。HIGH原Module9数值结果未通过1e-8秒证书；独立同目标CPU Newton精化通过同一严格凸LP最优性证书。未修改Module9或原V3，原V3独立CPU dry-run与数值失败完整保留；不把独立精化称为原V3成功。CPU复核与竞争见cpu_witness_revalidation.json。

LOW：最优部署 {'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt': 1, 'convnext_tiny': 1, 'facebook/mms-lid-256@edc73fd00996e671dfc59d16436a29b12b10588a': 1, 'phi4mm': 1, 'textattack/xlnet-base-cased-SST-2@9ceeb077dcd5cf5ae790572b2bd6aec755a263be': 1}；safe=22215.59796649061 MiB；max rho=0.6271892074666666；J=0.1333161713024968秒。

MID：最优部署 {'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt': 1, 'convnext_tiny': 1, 'openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e': 1, 'phi4mm': 1, 'textattack/xlnet-base-cased-SST-2@9ceeb077dcd5cf5ae790572b2bd6aec755a263be': 1, 'whisper_turbo': 1}；safe=24309.187387494312 MiB；max rho=0.5507499831927358；J=0.15145449124714733秒。

HIGH：最优部署 {'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt': 1, 'convnext_tiny': 1, 'openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e': 1, 'phi4mm': 1, 'textattack/xlnet-base-cased-SST-2@9ceeb077dcd5cf5ae790572b2bd6aec755a263be': 1, 'whisper_turbo': 1}；safe=24309.187387494312 MiB；max rho=0.7442381203546895；J=0.3225825843972564秒。

冻结后顺序LOW→MID→HIGH；每点最多三次独立进程，10秒warmup+120秒测量，seeds16101/16202/16303，原完整六任务样本流、native handlers、scheduled arrival-to-result E2E及stability_v2阈值。首个unstable或运行/显存不可行立即停止，不改点、不重试。

|run|steady状态|ASR G实际份额|ASR G/S完成数|六任务完成|safe预算|通过|
|---|---|---:|---|---|---|---|
|low_rep1|steady|1.0|14/0|True|True|True|
|low_rep2|steady|1.0|14/0|True|True|True|
|low_rep3|steady|1.0|14/0|True|True|True|
|mid_rep1|recoverable_congested|0.0|0/120|True|True|False|
|mid_rep2|recoverable_congested|0.0|0/120|True|True|False|
|mid_rep3|unstable|0.0|0/120|True|True|False|

per_task_routing.csv来自实际request记录，e2e_summary.csv逐任务报告E2E mean/P95；memory_summary.csv单列factor×两个idle窗口与sampled owned peak。per_model_metrics.csv包含throughput、真实worker handler busy wall fraction、基于实测s0/g的offered rho、host CPU和NVML process GPU utilization；如设备不支持process SM利用率明确留空，handler busy不是GPU SM占用率。device utilization单列，不能归因到个别模型。raw保留队列/尾部增长、完整请求、actual PIDs、READY和遥测，所有进程独立且释放核验。

首个未通过阶段：MID / mid_rep1。CPU预测可行而GPU观察=recoverable_congested，错误=[]，显存通过=True。具体队列/运行原因见gate.first_failure和raw，不把数值预测当真实稳定性。未移动冻结工作负载或路由。

这只验证三个离散120秒development serving点与独立重复，不宣称连续稳定边界、未定义延迟SLO或未测副本能力。全部历史保护文件hash核验。

## 首个 unstable 与停止证据

MID rep3 触发原 hard reason `sustained_positive_tail_growth_with_nontrivial_end_backlog`：尾部 slope=0.509483 requests/s，growth=12.295744 requests，increasing fraction=1.0，窗口末 backlog=62、peak=64。bounded drain 后 backlog=0，所有请求最终完成，但不改变 measurement 期间持续队列增长的 unstable 判定。

首个成功条件失败是MID rep1（recoverable_congested）；强制停止点是MID rep3（unstable）。HIGH三次全部SKIPPED，CPU约5.1556%分流预测未获GPU验证，不能声称HIGH支持或被直接反证。

六任务同时 serving 下，语音 worker 实测 service time 高于冻结 standalone s0，turbo worker 接近持续忙碌，队列尾部增长；疑似共享 GPU 竞争及单任务校准向完整 serving/样本流的迁移偏差。本实验未做因果隔离，不能确定唯一根因。不是显存超预算或运行错误；全部请求最终完成也不能消除原 stability_v2 的持续尾部增长判据。

LOW各3/3 steady；MID 0/3 steady（2 recoverable_congested + 1 unstable）；所有6个实际session均无request/worker错误、显存合规、六任务均完成且worker clean release。详见failure_analysis.json和serving_identity_and_request_audit.json。

逐进程SM利用率为NVML实际返回样本的均值，缺失/不活跃样本不补0，不把它称为均匀时间窗duty fraction；完整口径见utilization_reporting_note.json。
