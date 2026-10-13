"""One bounded infrastructure amendment; only four previously unused frozen slots."""
import os,time,fcntl,copy,shutil
from .common import *
from .telemetry_v2 import SafeMonitorV2
from . import gpu as native
from . import report_v2
from replica_scaling.telemetry import snapshot

def archive_v1_outputs():
 dest=OUT/'v1_controller_outputs';dest.mkdir(exist_ok=True)
 for p in list(OUT.glob('*.csv'))+[OUT/'run_observations.json',OUT/'telemetry_fix_gate.json']:
  if p.exists() and not (dest/p.name).exists():shutil.copy2(p,dest/p.name)

def freeze():
 stress=read(OUT/'telemetry_stress_result_v2.json');fast=read(OUT/'nonblocking_control_acceptance.json');harness=read(OUT/'cpu_harness_acceptance_v2.json')
 assert stress['passed'] and stress['elapsed_seconds']>1997.196239099 and stress['max_control_check_seconds']<.01 and fast['passed'] and harness['passed']
 original=read(OUT/'frozen_gpu_manifest.json');f=copy.deepcopy(original);remaining=read(OUT/'campaign_v1_stopped.json')['remaining_frozen_slots']
 f['slots']=[s for s in original['slots'] if s['run_id'] in remaining];assert len(f['slots'])==4
 f['parent_manifest']=ref(OUT/'frozen_gpu_manifest.json');f['amendment']='NONBLOCKING_TELEMETRY_CONTROL_PATH_ONLY';f['frozen_unix_ns']=time.time_ns()
 f['GPU']=snapshot();assert not f['GPU']['processes'] and f['GPU']['used_mib']<=128
 assert all(f['GPU'][k]==original['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 f['implementation_v2']=[ref(ROOT/'module30'/n) for n in ['telemetry_v2.py','stress_v2.py','nonblocking_checks.py','harness_checks_v2.py','gpu_v2.py','report_v2.py','continue_v2.py']]
 f['infrastructure_v2_evidence']=[ref(OUT/n) for n in ['telemetry_stress_result_v2.json','nonblocking_control_acceptance.json','cpu_harness_acceptance_v2.json','telemetry_observer_lock_reproduction.json','campaign_v1_stopped.json']]
 f['global_formal_run_budget']=dict(maximum=6,used_before_amendment=2,remaining=4,retries=0,original_model_points_seeds_ratios_unchanged=True)
 f['invalid_for_physical_U']=['C1_split05','C1_s_only'];f['cross_version_comparison']='FORBIDDEN';f['only_complete_same_version_candidate']='C2'
 archive_v1_outputs();save('frozen_gpu_manifest_v2.json',f)
 save('telemetry_fix_gate.json',dict(passed=True,version='v2_nonblocking_control',original_concurrency_fixed=True,observer_lock_fixed=True,
  long_stress=stress,nonblocking_checks=fast,harness=harness,original_v1_gate=ref(OUT/'v1_controller_outputs/telemetry_fix_gate.json'),
  Module29_gate=ref(ROOT/'results/module29_dynamic_replication/module29_gate.json')))
 audit=OUT/'telemetry_fix_audit.md'
 audit.write_text(audit.read_text()+'''\n## v2：修复初版控制路径锁阻塞\n\n初版check和set_active使用整个采样锁，NVML/host/thermal I/O期间可能阻塞请求提交。这是本模块新增实现问题，不能当成模型的背景损失或分流收益。CPU慢传感器直接重现250ms check等待；完整第一窗口Text/Image/OCR平均schedule delay为34.5/45.3/54.7ms。原始时间戳与效用算式未篡改，但这些窗口不具备物理可行性比较资格。两个已启动窗口完整保留并计入6次预算，不补跑。\n\n新SafeMonitorV2把控制状态和phase指针发布与串行采样/日志锁分离：check只读取短锁错误状态及线程安全Event；coherent_records先取得phase指针，退出状态锁后在runtime.lock下拷贝。所有I/O均在控制锁之外。NVML不能阻塞check或phase发布，request仍完整冻结并按原时间轨迹dispatch。\n\n250ms传感器I/O期间12次check与phase发布均低于10ms，实测为微秒级。慢传感器下原生80请求完整/40完成40中止测试通过，全部80终态保留；完整窗口每请求schedule delay均低于20ms。重新运行2100秒带25ms传感器阻塞的并发压力，dict快照一致、日志无重复遗漏、线程退出、sensor/文件/退出故障注入可见，control check最大延迟低于10ms。精确值见对应v2 JSON。\n\n仅上述检查通过且GPU空闲后，使用frozen_gpu_manifest_v2.json继续4个此前未执行槽。原配置、模型、seed、输入、分流比例及效用均不变；C1跨版本对照禁止，C2三臂可比较。原Module29 Gate和全部历史文件、初版manifest及数据保持不变。\n''')
 return f

def main():
 lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'campaign_completed.json').exists()
 f=freeze();native.SafeMonitor=SafeMonitorV2
 attempted=['C1_split05','C1_s_only'];stop=None
 for slot in f['slots']:
  assert not (OUT/'raw_runs'/(slot['run_id']+'.json')).exists(),'Previously unused slots only'
  for r in f['implementation']+f['reused_implementation']+f['source_refs']+f['implementation_v2']:
   assert ref(r['path'])['sha256']==r['sha256'],r['path']
  attempted.append(slot['run_id']);print('V2_START',slot['run_id'],flush=True)
  try:raw=native.run(slot,f)
  except Exception as exc:
   save('v2_prelaunch_failure.json',dict(run_id=slot['run_id'],error=repr(exc)));stop=dict(run_id=slot['run_id'],error=repr(exc));break
  # Preserve the raw saved by the unchanged native harness, then add version provenance.
  source=OUT/'raw_runs'/(slot['run_id']+'.json');dest=OUT/'v2_native_raw_before_provenance'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
  raw.update(telemetry_version='v2_nonblocking_control',frozen_manifest_v2=ref(OUT/'frozen_gpu_manifest_v2.json'),native_raw_before_provenance=ref(dest))
  save('raw_runs/'+source.name,raw)
  o=report_v2.observation(raw);print('V2_FINISH',slot['run_id'],'eligible',o['eligible'],'U',o['U'],'errors',raw['errors'],flush=True)
  report_v2.interim()
  if not raw['valid_session'] or not raw['gpu_release']['verified']:
   stop=dict(run_id=slot['run_id'],reason='TECHNICAL_FAILURE_NO_MORE_AMENDMENTS',errors=raw['errors']);break
 save('campaign_completed.json',dict(attempted=attempted,stop=stop,skipped=[s['run_id'] for s in read(OUT/'frozen_gpu_manifest.json')['slots'] if s['run_id'] not in attempted],
  maximum_formal_runs=6,no_retries=True,infrastructure_amendments=1,original_points_seeds_routes_unchanged=True))
 report_v2.final()
if __name__=='__main__':main()
