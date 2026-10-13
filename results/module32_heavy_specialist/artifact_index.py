import sys,time,json,hashlib,csv
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import read,save,ref,csvsave
f=read(OUT/'experiment_manifest.json');gate=read(OUT/'module32_gate.json')
# Raw source label was inherited; preserve raw logs and separately document actual frozen trace identity.
save('provenance_annotations.json',dict(recorded_unix_ns=time.time_ns(),historical_transport_label='Module32 frozen matched Module30 C1 trajectory',actual_formal_candidate=f.get('formal_candidate'),actual_formal_seed=f['slots'][0]['seed'] if f['slots'] else None,actual_trajectories=[dict(run_id=s['run_id'],trajectories=s['trajectories']) for s in f['slots']],scope='The inherited routing_source text describes the source adapter. Actual matched input/arrival traces are the frozen trajectory files and SHA, not that descriptive string. Raw logs were preserved.'))
profiles=read(OUT/'colocation_profile.json');sensors=[]
for o in profiles:
 p=OUT/'raw_runs'/(o['run_id']+'.json');r=read(p)
 sensors.extend(dict(run_id=o['run_id'],stage='B',**v) for v in r['telemetry']['samples'])
csvsave('stage_B_GPU_CPU_environment.csv',sensors,['run_id','stage'])
index='''# Module 32 数据索引

最终报告：module32_summary.md。最终Gate：module32_gate.json / gate_decision.md。所有状态与ΔU以这两份最终产物为准；归档目录中的报告是当时的中间状态。

|阶段|入口|内容|
|---|---|---|
|A|model_pair_review.md / .json / .csv|两配对、目标质量、六任务路径、单副本显存|
|A|quality_admission_provenance.csv|原始质量准入来源；heavy S 200条WER复算|
|A|historical_review_14F_31A.json|历史画像与Module31A机制对照|
|B|colocation_profile.json / .csv|全部共置容量与压力预检；不合法窗口U为空|
|B|stage_B_GPU_CPU_environment.csv|所有预检的GPU、CPU、显存、温度、频率传感样本|
|B|G_service_drift_audit.json / S_service_expansion_by_arrival_block.csv|服务需求漂移审计|
|冻结|frozen_load.json / experiment_manifest.json|固定负载、池、副本、原始输入/源码/checkpoint SHA与运行协议|
|冻结|module32_gate_definition.json|正式窗口开始前冻结的用户Gate|
|C|three_arm_gpu_measurement.json / raw_runs/A*.json|真实600秒三臂测量与原始日志|
|C|request_journals / trajectories|完整请求journal及匹配轨迹|
|C|system_utility_comparison.csv|原Module20A系统U、T/L/P/M分项与合法ΔU|
|C|per_task_latency_comparison.csv / worker_service_wait.csv|六任务mean/P95变化、各worker执行/等待|
|C|native_latency_decomposition.csv|同输入S等待释放、服务变化、G新增执行/等待及host残差|
|C|GPU_CPU_environment.csv / resource_comparison.csv / backlog_timeseries.csv|传感器、异常、队列及资源|
|D|heavy_specialist_mechanism_analysis.md / .json|31A排队收益、G成本、背景抵消、共置膨胀判断|
|D|pair_comparisons_user_gate.json|两合法对照的最终Gate计算|
|审计|final_data_acceptance.json / historical_integrity_after.json|完整权重SHA、源码/输入、历史保护、600秒、完整cohort、GPU释放|
|审计|provenance_annotations.json|继承transport文字标签与实际新冻结轨迹的说明|
|可视化|utility_and_queue_analysis.png / .pdf|六任务效用及排队图；仅完整合法三臂生成|
|完整索引|artifact_manifest.json|全部本轮产物的递归SHA/大小清单|

所有初始失败、过载诊断与审计修正保留在各归档目录；未通过窗口不可用于ΔU。原始数据不因报告修正而重写。研究只完成一次三臂探索，无独立长窗口复现，正式优化器未修改。容量测量使用有效真实样本及原native推理参数；未加入推理延迟或无意义计算。
'''
(OUT/'README.md').write_text(index)
# Avoid self-reference and logs still being written by this invocation.
artifacts=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and p.name!='artifact_index.log']
save('artifact_manifest.json',dict(finished_unix_ns=time.time_ns(),artifacts=artifacts,recursive=True,gate=gate['gate'],history_protected=gate['historical_integrity'],excluded_self=True))
print('ARTIFACTS_INDEXED',len(artifacts))
