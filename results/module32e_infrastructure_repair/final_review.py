"""Check frozen source integrity, honest missing measurements and final outputs."""
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
f=read(OUT/'frozen_validation_manifest.json');g=read(OUT/'module32e_gate.json');a=read(OUT/'gpu_preflight_acceptance.json')
source_errors=[r['path'] for r in f['sources']+f['frozen_inputs'] if ref(r['path'])['sha256']!=r['sha256']]
metadata_errors=[p for p,v in f['checkpoint_metadata'].items() if [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
assert not source_errors and not metadata_errors,(source_errors,metadata_errors)
assert g['gate']=='GPU_PREFLIGHT_FAILED' and not g['engineering_valid'] and not a['passed']
assert a['control']['check_samples']==0 and a['control']['max_check'] is None and a['control']['max_publish'] is None
assert a['U'] is None and not a['utility_computed'] and g['formal_GPU_runs']==0
assert g['checks']['router_CPU'] and g['checks']['control_CPU'] and g['checks']['history_protection']
assert len(list((OUT/'run_attempts').glob('*.json')))==1
raw=read(OUT/'raw_runs/E0_DYNAMIC.json');assert raw['gpu_release']['verified']
assert raw['status']=='FAILED' and not raw['phases'] and not raw['all_runtime_records']
assert read(OUT/'routing_equivalence_report.json')['decisions']==1381
assert ref(OUT/'router_optimized.py')['sha256']==ref(ROOT/'results/module32d_router_repair/router_optimized.py')['sha256']
assert ref(OUT/'vqa_aware_dynamic_router.py')['sha256']==ref(OLD/'vqa_aware_dynamic_router.py')['sha256']
summary=OUT/'module32e_summary.md'
text=summary.read_text().replace('全部历史输入14927文件完整SHA','15,250条输入清单对应14,927个不同规范路径，完整SHA')
text=text.replace('真实GPU同状态因果回放结果见gpu_preflight_acceptance.json。','GPU未进入请求窗口，未执行真实在线因果回放，不以CPU等价性代替GPU验收。')
summary.write_text(text)
env=OUT/'gpu_environment_report.md';s=env.read_text();s=s.replace('本轮一臂工程验收，无配对温差/频率可比性结论，无正式U。完整运行前预热、温控、native worker日志、请求/控制时间和逐0.5秒GPU采样见原始数据。','本轮在首个worker CUDA初始化失败后停止；仅有启动阶段0.5秒遥测和预启动温控记录，没有驻留预热、conditioning、请求测量或在线控制计时。无配对温差/频率可比性结论，无正式U。native worker错误、清洁关闭、GPU释放及全部启动样本见原始数据。');env.write_text(s)
save('final_artifact_review.json',dict(passed=True,checks=dict(frozen_sources_unchanged=True,frozen_inputs_unchanged=True,
 checkpoint_metadata_unchanged=True,original_runtime_byte_identical=True,original_policy_byte_identical=True,
 historical_1381_exact_decisions=True,CPU_route_P95_pass=True,CPU_control_pass=True,history_full_SHA_pass=True,
 failed_GPU_not_counted_as_pass=True,zero_control_samples_NOT_zero_latency=True,all_initial_failures_preserved=True,
 no_GPU_retries=True,no_utility_calculations=True,clean_GPU_release=True,no_automatic_next_module=True),
 frozen_source_count=len(f['sources']),frozen_input_count=len(f['frozen_inputs']),historical_baseline_exception_not_restored=ref(OUT/'prior_incident_record.json'),
 GPU_gate=g['gate'],scope='CPU engineering passed; GPU acceptance failed before warmup. Strict current32E history preservation, not retroactive byte-exact32D restore.'))
assert not VIOLATIONS
files=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'work' not in p.relative_to(OUT).parts and p.name!='artifact_manifest.json']
save('artifact_manifest.json',dict(at_unix_ns=time.time_ns(),files=files,history_unchanged_since32E_baseline=True,old32D_byte_restore_not_claimed=True))
print('FINAL_REVIEW_PASS',g['gate'],len(files),flush=True)
