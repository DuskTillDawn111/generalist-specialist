"""Freeze only four OCR next counts and memory correction rule before new GPU."""
import copy,re
from .common import *
from replica_scaling.telemetry import snapshot

def memory_evidence():
 """Reuse independent complete, isolated postwarm-idle observations by state."""
 obs=[]
 roots=[ROOT/'results/module11_replica_and_cost/raw_runs',D/'raw_runs',V/'raw_runs']
 for directory in roots:
  for p in sorted(directory.glob('*.json')):
   raw=read(p)
   if not raw.get('valid_session') or not raw.get('postwarm_idle') or not raw.get('gpu_release',{}).get('verified'):continue
   slot=raw['slot']
   if directory!=V/'raw_runs' and slot.get('kind')!='closed_loop_profile':continue
   if directory==V/'raw_runs':group=slot['context']+'::'+slot['Specialist_regime']
   else:group=re.sub(r'_x\d+_rep\d+$','',slot['run_id'])+'::'+slot.get('primary_task','')
   identity=raw['worker_identities'];idle=raw['postwarm_idle']
   for m,n in slot['deployment'].items():
    pids={v['pid'] for v in identity.values() if v['family']==m}
    if len(pids)!=n:continue
    if any(s['foreign_pids'] or not s['each_owned_pid_has_cuda_context'] for s in idle):continue
    value=statistics.mean(sum(p['used_mib'] for p in s['processes'] if p['pid'] in pids) for s in idle)
    obs.append({'family':m,'count':n,'repetition':slot['repetition'],'seed':slot['seed'],'group':str(directory)+'::'+group,'postwarm_idle_mib':value,'raw':ref(p),'observation_scope':'postwarm idle owned worker-family sum, mean over fixed idle window; completed isolated session; mixed-load instability does not erase a completed pre-timed idle observation'})
 return obs

def main():
 assert not (OUT/'profiling_freeze.json').exists(),'Existing freeze is immutable'
 previous=read(D/'protected_before.json')['files']+read(D/'module14d_artifact_manifest.json')['files']+read(V/'module14e_artifact_manifest.json')['files']+[ref(D/'module14d_artifact_manifest.json'),ref(V/'module14e_artifact_manifest.json')]
 previous=list({r['path']:r for r in previous}.values());verify(previous);protected=[ref(r['path']) for r in previous];save('protected_before.json',{'files':protected,'verified':True,'at_utc':now()})
 from .reconcile import main as reconcile
 reconcile()
 decisions=read(D/'x3_profiling_decision.json')['decisions'];required=[d['planning_id'] for d in decisions if d['decision']=='X3_PROFILING_REQUIRED'];assert len(required)==4 and all(m.startswith('baudm/') for m in required)
 plan=copy.deepcopy(read(D/'measurement_plan.json'));plan['GPU']=snapshot();assert not plan['GPU']['processes'] and plan['GPU']['used_mib']<=128;plan['registry']={m:plan['registry'][m] for m in required}
 protocol=read(D/'closed_loop_protocol.json');verify(protocol['semantics_source_refs']);save('closed_loop_protocol.json',protocol);save('measurement_plan.json',plan)
 h=read(D/'heavy_empirical_replica_table.json');slots=[]
 for index,m in enumerate(required):
  rate=.45*h['capacity_factors'][m]['2']*1000/h['models'][m]['service_time_ms']['ocr']
  for rep,seed in enumerate(protocol['seeds'],1):
   for x in ([2,3] if rep%2 else [3,2]):
    rid=f'ocr{index}_x{x}_rep{rep}';slot={'run_id':rid,'kind':'closed_loop_profile','model':m,'primary_task':'ocr','task_path':'ocr','required_x':3,'repetition':rep,'seed':seed,'deployment':{m:x},'measurement_seconds':30.,'lambda_rps':{'ocr':rate},'routing':{'ocr':{m:rate}},'low_lambda':{'ocr':rate},'low_routing':{'ocr':{m:rate}}};save('slots/'+rid+'.json',slot);slots.append(slot)
 save('resident_set_manifest.json',{'states':slots,'scope':'Four OCR only; x2 auxiliary paired references and x3 measurements; no x>=4'})
 save('closed_loop_source_revision_v2.json',{'source':ref(ROOT/'module11b/closed_loop.py'),'reused_unchanged_runner':True})
 obs=memory_evidence();save('existing_postwarm_memory_observations.json',obs)
 policy={'at_utc':now(),'name':'EMPIRICAL_POSTWARM_IDLE_GROUP_ENVELOPE','definition':'For each already-measured family/count, retain historical M and take max with the arithmetic mean of each complete THREE-independent-repetition postwarm-idle observation group. No sample maxima, peak memory, device overhead, extrapolated next counts or synthetic padding are inserted. All groups use the same owned worker-family NVML sum and frozen safe factor.','why':'Historical one-sample pre-cohort M1 may understate warmed residency; an empirical envelope across existing measured warmup/task scopes reconciles both regimes without shrinking costs to fit outcomes. Same rule applied to E/H shared identities.','group_rule':'Dataset/run-series/task or existing 14E context/regime, exactly repetitions1/2/3 with independent PIDs/seeds; exclude invalid/missing/non-isolated sessions.','single_replica_catalog_policy':'Update local memory attribute to reconciled M1 solely to satisfy existing loader identity M1==profile.memory; identity/capability/role/quality/s0 and original files unchanged.','safe_factor':h['safe_factor'],'new_x3_memory':'Arithmetic mean of three x3 postwarm-idle family observations; same empirical envelope definition; existing counts may include newly measured auxiliary x2 group, g2 frozen.','unmeasured_memory':'UNKNOWN; no linear next-count inference','budgets_unchanged':True,'latency_unchanged':True,'source_observations':ref(OUT/'existing_postwarm_memory_observations.json')};save('memory_semantics_freeze.json',policy)
 implementation=[ref(p) for p in sorted((ROOT/'module14f').glob('*.py'))]+[ref(plan['registry'][m]['script']) for m in required]
 save('input_freeze.json',{'protected_references':protected,'measurement_plan':ref(OUT/'measurement_plan.json'),'resident_set_manifest':ref(OUT/'resident_set_manifest.json'),'samples':plan['sample_pool'],'implementation':implementation})
 save('profiling_freeze.json',{'at_utc':now(),'requested_checkpoints':required,'x3_independent_repetitions':12,'x2_paired_reference_sessions':12,'slots':slots,'protocol':ref(OUT/'closed_loop_protocol.json'),'memory_rule':ref(OUT/'memory_semantics_freeze.json'),'input_freeze':ref(OUT/'input_freeze.json'),'source_x3_active_optimum_witness':ref(D/'x3_profiling_decision.json'),'x4_not_authorized_without_new_witness':True,'no_other_GPU_measurements':True,'frozen_contexts':ref(C/'screen_protocol_freeze.json'),'gate_policy':'Specialist-only structure ends uniform-heavy->Generalist-sharing hypothesis; only multiple G-share or new-split events support shift; no-shift emits two proposal manifests only.'})
 print('14F_FROZEN',len(protected),'protected',len(obs),'memory observations',len(slots),'sessions',flush=True)
if __name__=='__main__':main()
