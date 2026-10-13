"""Exclude v1 observer-affected points from feasibility U; preserve original raw arithmetic."""
import copy,shutil
from .common import *
from . import report as original

_native_observation=original.observation
V1={'C1_split05','C1_s_only'}
def observation(raw):
 o=_native_observation(raw);o['telemetry_version']='v1_blocking_check' if o['run_id'] in V1 else 'v2_nonblocking_check'
 if o['run_id'] in V1:
  o['diagnostic_full_cohort_observer_affected_U']=o['U'];o['eligible']=False;o['U']=None
  o['technical_observer_effect']=True;o['unavailable_reason']='Sampling mutex stalls request dispatch; not eligible for physical split feasibility'
  if 'components' in o:o['components']['U']=None
 return o

def install():original.observation=observation

def interim():
 install();return original.interim()

def final():
 install();original.finalize()
 gate=read(OUT/'module30_gate.json');old=[]
 for runid in sorted(V1):
  p=OUT/'raw_runs'/(runid+'.json')
  if p.exists():old.append(_native_observation(read(p)))
 diagnostic=None
 if len(old)==2 and all(o['eligible'] for o in old):
  a=next(o for o in old if o['arm']=='split05');b=next(o for o in old if o['arm']=='s_only');diagnostic=a['U']-b['U']
 save('v1_observer_affected_diagnostics.json',dict(full_cohort_only=True,original_observations=old,
  diagnostic_observed_U_difference=diagnostic,eligible_for_feasibility=False,
  reason='Identical frozen v1 observer can impose variable dispatch jitter; physical effect not established. No truncated utility used.'))
 if not gate['positive_candidates'] and diagnostic is not None and diagnostic>.01:
  gate['gate']='INCONCLUSIVE_POSITIVE_SIGNAL';gate['observer_affected_positive_signal']=diagnostic
 gate.update(infrastructure_final_version='v2_nonblocking_check',v1_observer_fault=ref(OUT/'telemetry_observer_lock_reproduction.json'),
  corrected_manifest=ref(OUT/'frozen_gpu_manifest_v2.json'),observer_affected_runs=sorted(V1),
  C1_controls_comparable=False,C1_status='TECHNICALLY_INCONCLUSIVE_OBSERVER_EFFECT',
  formal_budget_includes_two_v1_attempts=True,formal_retry_count=0)
 save('module30_gate.json',gate)
 summary=OUT/'module30_summary.md'
 summary.write_text(summary.read_text()+f'''\n## 必须阅读的遥测修正\n\n初版SafeMonitor.check与NVML采样共用长锁，CPU替身复现250ms阻塞，第一窗口背景提交延迟均值35–55ms。前两个完整600秒窗口全部原始请求保留，但合法可用于物理可行性筛选的U为NULL；原始完整cohort算式保存在v1_observer_affected_diagnostics.json，禁止当正收益证据。没有截尾效用、没有补跑已执行槽。\n\n当前窗口清洁退出并释放GPU后停止初版；v2错误检查和phase发布使用独立短状态锁，传感器I/O与journal串行锁不进入请求dispatch路径。250ms传感器阻塞下12次控制检查均微秒级；原生80请求/40请求中止测试与2100秒慢传感器并发压力在剩余GPU之前重新通过。完整证据见telemetry_fix_audit.md和frozen_gpu_manifest_v2.json；初版manifest、源码、错误与数据保留。\n\n四个未执行的原冻结槽继续测量：C1_split10及C2三方案。模型、负载、seed、输入和比例均不改，总正式窗口6次。C1跨监测器版本不作合法ΔU；只有C2有完整同版本三臂对照。当前权威Gate为 **{gate['gate']}**；初版表观差值{diagnostic}只作技术诊断。未自动启动独立验证或下一Module。\n''')
 p=OUT/'positive_split_mechanism.md';p.write_text(p.read_text()+'\n遥测版本边界：C1前两窗口受采样锁影响，C1跨版本物理ΔU不合法；C2三臂均使用v2，方可作物理比较。所有v1完整cohort算式另存，不能当可行性收益。\n')
 # Preserve every new source hash too; original code and Module14F–29 hashes remain unchanged.
 f=read(OUT/'frozen_gpu_manifest_v2.json');changed=[r['path'] for r in f['implementation_v2'] if ref(r['path'])['sha256']!=r['sha256']]
 assert not changed,changed
 save('v2_execution_integrity.json',dict(passed=True,implementation_v2=f['implementation_v2'],no_previous_formal_slot_repeated=True,total_formal_slots=6))
 print('MODULE30_FINAL_V2',gate['gate'],flush=True)
