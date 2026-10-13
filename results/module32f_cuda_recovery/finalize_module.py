"""Summarize engineering evidence only, after all GPU children exit."""
from m32f.isolation import install
install('supervisor')
from m32f.common import *
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    f=read(OUT/'frozen_validation_manifest_v2.json')
    immutable_errors=[]
    for x in f['sources']+f['frozen_inputs']:
        if ref(x['path'])['sha256']!=x['sha256']:immutable_errors.append(x['path'])
    hist=read(OUT/'history_hashes/before_diagnosis_vs_after_preflight.json')
    cpu=read(OUT/'cpu_regression/CPU_runtime_acceptance.json')
    eq=read(OUT/'cpu_regression/routing_equivalence_report.json')
    co=read(OUT/'co_residency_acceptance_v2.json')
    init=read(OUT/'initialization_campaign_status_v2.json')
    rows=[];arms={}
    for rid in ['E0_S','E0_D']:
        p=OUT/'gpu_acceptance'/f'{rid}.json'
        if not p.exists():continue
        a=read(p);arms[rid]=a
        raw=read(OUT/'raw_runs'/f'{rid}.json');d=a['details'];tele=raw['telemetry']['samples']
        control=a['control'];ph=raw.get('phases',{}).get('measurement',{})
        win=ph.get('window',{});measurement=[s for s in tele if win.get('start_time_ns',0)<=s['time_ns']<=win.get('end_time_ns',0)]
        cmax=max(control.get('check',[])+control.get('phase_publish',[]),default=0)*1000
        # Exact raw names are retained separately; no absent probe is accepted.
        probes=raw['control_probes'];check_max=max((s['wall_ms'] for s in probes),default=None)
        rows.append(dict(run_id=rid,passed=a['passed'],measurement_seconds=win.get('duration_seconds'),offered=d.get('offered'),completed=d.get('completed'),G_ASR=d.get('G_ASR'),S_ASR=d.get('S_ASR'),router_P95_ms=d.get('router',{}).get('p95_ms'),router_P99_ms=d.get('router',{}).get('p99_ms'),control_check_max_ms=check_max,control_and_publish_max_ms=cmax,control_latency_original=control,sampled_peak_mib=a['sampled_peak_mib'],temperature_mean_c=float(np.mean([s['temperature_gpu_c'] for s in measurement])) if measurement else None,temperature_max_c=max((s['temperature_gpu_c'] for s in measurement),default=None),sm_clock_min_mhz=min((s['sm_clock_mhz'] for s in measurement),default=None),sm_clock_max_mhz=max((s['sm_clock_mhz'] for s in measurement),default=None),stability=d.get('stability'),U=None,utility_computed=False))
    # Full samples remain in immutable raw_runs and gpu_measurements CSVs;
    # the gate/comparison retain maxima and explicit raw references.
    for row in rows:
        row.pop('control_latency_original')
        row['raw_run']=str(OUT/'raw_runs'/f"{row['run_id']}.json")
    cpu_ok=cpu['passed'] and eq['passed'] and not eq['semantics_changed']
    initialization_ok=init['status']=='PASSED' and co['passed']
    gpu_ok=len(arms)==2 and all(a['passed'] for a in arms.values())
    protection=read(OUT/'history_write_protection_tests.json')
    proc=read(OUT/'proc_scope_protection.json')
    minimal=[read(p) for p in (OUT/'minimal_cuda').glob('repaired_*.json')]
    minimal_ok=bool(minimal) and all(r['passed'] and r['synchronized_tensor_sum']==357389824 for r in minimal)
    checks=dict(minimal_repaired_CUDA=minimal_ok,CPU_runtime=cpu_ok,historical_semantics=eq['passed'],history_write_protection=protection['passed'],proc_scope_protection=proc['passed'],cold_cache_atomic_publication=read(OUT/'triton_cache_CPU_repaired.json')['passed'],six_model_initialization=initialization_ok,two_short_GPU_preflights=gpu_ok,historical_hashes=hist['passed'],frozen_source_hashes=not immutable_errors)
    gate='GPU_PREFLIGHT_RECOVERED' if all(checks.values()) else ('CUDA_WORKER_INITIALIZATION_FAILED' if not initialization_ok else 'GPU_PREFLIGHT_FAILED')
    result=dict(module='32F',gate=gate,at_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),timezone='Asia/Shanghai',scope='CUDA startup and short engineering preflight only',checks=checks,immutable_source_errors=immutable_errors,root_cause='Original Landlock denied CUDA native /proc/self/task/<tid>/comm thread naming; clean base CUDA works, old isolation reproducibly returns Error304.',secondary_infrastructure_cause='Cold Triton native cache publication used cross-directory rename blocked by Landlock ABI1 (EXDEV). Same-directory atomic provider preserves native compilation/inference.',repair_version='infrastructure_v2',router_semantics_changed=False,model_weights_or_inference_parameters_changed=False,driver_reinstalled=False,formal_utility_computed=False,U=None,long_window_started=False,historical_gates_modified=False,prior32D_irrecoverable_difference_preserved=True,preflight=rows,history_manifest=ref(OUT/'history_hashes/before_diagnosis_vs_after_preflight.json'),freeze=ref(OUT/'frozen_validation_manifest_v2.json'),no_automatic_next_stage=True)
    csvsave('gpu_preflight_comparison.csv',rows)
    csvsave('regression_results.csv',[
        dict(stage='1_minimal_CUDA',passed=minimal_ok,evidence='minimal_cuda/repaired_*.json',note=f'{len(minimal)} successful repaired isolated fresh processes; exact synchronized tensor result'),
        dict(stage='2_PARSeq_single',passed=read(OUT/'single_model_runs_v2/0.json')['passed'],evidence='single_model_runs_v2/0.json',note='Native transform/weights/float32; original 3 OCR probes'),
        dict(stage='3_six_independent_models',passed=initialization_ok,evidence='initialization_campaign_status_v2.json',note='Six fresh launches sequentially, supported native paths and clean release'),
        dict(stage='4_six_model_co_residency',passed=co['passed'],evidence='co_residency_raw_v2.json',note=f"safe resident {co['safe_memory_mib']}MiB; original B3"),
        *[dict(stage='5_S_only_preflight' if r['run_id']=='E0_S' else '6_Dynamic_preflight',passed=r['passed'],evidence=r['raw_run'],note=f"{r['completed']}/{r['offered']} complete; G/S ASR {r['G_ASR']}/{r['S_ASR']}; U not computed") for r in rows],
        dict(stage='CPU_semantics',passed=eq['passed'],evidence='cpu_regression/routing_equivalence_report.json',note='1381 exact decisions, no tolerance and no policy changes'),
        dict(stage='CPU_concurrent_scale',passed=cpu_ok,evidence='cpu_regression/CPU_runtime_acceptance.json',note='2400 second request scale accelerated 50x, not wall-time soak; all outliers retained'),
        dict(stage='history_final',passed=hist['passed'],evidence='history_hashes/before_diagnosis_vs_after_preflight.json',note='SHA/size/mtime/symlink equal; old irreversible32D exception remains documented'),
    ])
    save('module32f_gate.json',result)
    table='\n'.join(f"| {r['run_id']} | {r['passed']} | {r['completed']}/{r['offered']} | {r['G_ASR']}/{r['S_ASR']} | {r['router_P95_ms']} | {r['control_check_max_ms']} | {r['sampled_peak_mib']} |" for r in rows)
    summary=f'''# Module32F CUDA恢复结果

Gate：`{gate}`。本轮只进行启动修复与短时工程预检，U=NULL；未运行正式效用验证、2400秒窗口或优化器集成。

## 根因与修复

基础CUDA在干净独立进程中可用。旧Landlock独立及原组合均复现Error304，seccomp独立正常；全线程syscall证据显示CUDA线程命名打开自身`/proc/self/task/<tid>/comm`被EACCES拒绝。不是PARSeq独有、fork后CUDA复用、显存不足或并发context故障的证据。给worker自身task树追加WRITE_FILE，父supervisor只提供继承envelope，worker导入backend前再次收窄。历史文件写入仍拒绝。

冷缓存首次OCR另暴露原生Triton跨目录rename被Landlock ABI1拒绝EXDEV。仅以同目录临时文件+原子replace实现已有cache provider接口；不改变编译器选项、native算子、推理参数或权重。版本1失败完整保留，版本2在任何短时预检之前重新冻结工程源码；没有根据U或预检结果改策略。

## 回归与限制

历史1381次同状态决策逐字段完全相同，数值容差0。CPU2400秒规模3419请求以50倍时间压缩，真实发送/收集/遥测/日志线程；1x/8x历史的路由P95分别{cpu['results'][0]['p95_ms']:.6f}/{cpu['results'][1]['p95_ms']:.6f}ms。保留全部长尾，包括8x单次路由最大{cpu['results'][1]['max_ms']:.6f}ms；验收门槛是P95≤1ms，不能声称每次≤1ms或2400秒墙钟浸泡通过。

PARSeq与原六模型逐个初始化、各支持路径原输入探测、六模型完整共置与清洁释放结果为{initialization_ok}。共置安全驻留{co['safe_memory_mib']:.3f}MiB，原B3预算{f['slots'][0]['safe_memory_budget_mib']:.3f}MiB。质量仅核验原准入及闭集输出合法、原权重/推理/任务映射不变，不是重新评估质量分数。

短时两臂保持原工程seed320490和输入/到达序列、原六任务QPS、单副本驻留、300秒实际六任务调理/30秒warmup/180秒工程测量/60秒有界全队列排空。原control墙钟包含状态锁等待，检查及phase publish最大值均须<10ms；非空真实在线Dynamic P95≤1ms。全请求/等待/服务/E2E/队列/遥测/worker退出及原策略回放证据见raw_runs、gpu_measurements与gpu_acceptance，不按不同温度均值推断降频。

| 臂 | 工程通过 | 完成/计划 | Phi/Whisper ASR | 路由P95 ms | control check max ms | 实测峰值MiB |
|---|---|---|---|---|---|---|
{table}

历史SHA及size/mtime/symlink前后核验：{hist['passed']}（{hist['files']}条输入，差异{len(hist['changed'])}）。全部活动freeze源核验：{not immutable_errors}。32D不可逐字节恢复的旧progress差异仍保留prior_incident_record，本轮不伪称恢复，也未修改历史Gate。SHA重读安排在GPU测量之外，避免人为干扰。

工程有效性与效用结论分开：恢复成功只支持后续验证的工程准备，不证明Dynamic正收益或独立复现。完成本Module后停止。
'''
    (OUT/'module32f_summary.md').write_text(summary)
    save('final_frozen_source_integrity.json',dict(passed=not immutable_errors,mismatches=immutable_errors,manifest=ref(OUT/'frozen_validation_manifest_v2.json')))
    save('evidence_annotations.json',dict(history_write_protection_tests_scope_string='The inherited prose label Module32E in history_write_protection_tests.json is a documentation typo. Actual test source, OUT, Landlock writable directory and test evidence are Module32F, as independently recorded by frozen SHA and worker guard manifests. Original raw result preserved.',raw_frozen_manifest='Legacy raw frozen_manifest field identifies the retained original engineering freeze. active_engineering_manifest is the authoritative version2 source/cache-provider freeze, verified before each launch; workload/router settings unchanged.'))
    envrows='\n'.join(f"| {r['run_id']} | {r['temperature_mean_c']} | {r['temperature_max_c']} | {r['sm_clock_min_mhz']}–{r['sm_clock_max_mhz']} | {r['control_and_publish_max_ms']} |" for r in rows)
    (OUT/'gpu_environment_report.md').write_text(f'''# GPU工程环境记录

同一GPU UUID `GPU-31f2671b-99c5-ef57-be45-5c9a871e11f4`，RTX4090，驱动595.71.05；nvidia-smi CUDA13.2是驱动支持上限，PARSeq PyTorch2.14.0+cu130实际runtime13.0。可见物理设备节点为/dev/nvidia1，CUDA仅暴露一张卡映射cuda:0；完整设备和容器字段见container_environment.json/device_nodes.txt。未重装驱动、安装软件包、设置频率/风扇/功率。原subprocess.Popen+exec顺序READY，控制器不初始化CUDA，worker在导入后端之前安装自身隔离。

每臂前置温度≤38°C连续3次，驻留调理后≤40°C连续3次，原等待上限300/180秒。调理300秒有实际六任务工作，不在人为延时ASR中制造排队。原300/30/180/60秒工程协议与冷/热缓存加载条件详见冻结协议和raw日志。0.5秒实际NVML记录SM/显存频率、Pstate、功率、温度、风扇与hw/sw thermal slowdown；不以温度均值差推断降频。每臂均保持原450W功率限制，GPU_operating_state检查只在真实测量窗口核验全部采样的hw/sw thermal slowdown为Not Active，不宣称没有任何功率/空闲clock变化。峰值是NVML采样高水位，非瞬时allocator最大值，native worker额外保存allocator峰值。

| 臂 | 测量平均°C | 测量最高°C | 实际SM MHz范围 | check/publish联合max ms |
|---|---|---|---|---|
{envrows}

控制探针沿用perf_counter墙钟，覆盖state_lock等待与原异常谓词，检查及phase发布最大值均须<10ms；原始每次wall/thread_cpu/state_wait/state_hold以及全部publish样本保存，不删长尾。路由开销为真实lock+增量projection+policy.decide+enqueue，另存P50/P95/P99/max。等待/服务时间使用同一worker Unix submit/start/end；控制器monotonic到达/完成及跨钟映射原始字段均保留，未把路由CPU开销混为GPU服务时间。

六模型退出BYE、exit code、线程及显存释放逐项由valid_session验收，最终nvidia-smi状态见nvidia_smi_after.txt。历史全量SHA扫描安排在测量前后，未在请求关键阶段占用CPU/磁盘。
''')
    print(gate,checks,flush=True)

if __name__=='__main__':main()
