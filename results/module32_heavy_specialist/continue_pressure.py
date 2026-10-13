import sys,shutil,time,fcntl,traceback,collections
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
from m32.campaign import execute,describe_B,setup
import numpy as np
from module21.trajectory import choose
lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
f=read(OUT/'experiment_manifest.json');assert not f['slots']
assert (OUT/'raw_runs/B_S.json').exists() and (OUT/'raw_runs/B_G.json').exists()
a=OUT/'idle_worker_correction_continuation';a.mkdir(exist_ok=False)
for name in ['campaign.log','campaign_failure.json','campaign_status.json','experiment_manifest.json','module32_gate.json','module32_summary.md']:
 p=OUT/name
 if p.exists():shutil.copy2(p,a/name)
(OUT/'campaign_failure.json').unlink();f['CPU_only']=False
f['candidates']=read(OUT/'model_pair_review.json')
f['capacity_preparation_revision']=dict(at_unix_ns=time.time_ns(),failed_manifest=ref(a/'experiment_manifest.json'),failed_raw=ref(OUT/'raw_runs/B_G.json'),reason='Correct only the empty standby-worker audit to match original Module30 skip rule; reuse complete B_G_SAFE raw, no GPU remeasurement',old_G_probe_qps=.6,new_G_probe_qps=.2,G_probe_seconds=120,formal_windows_started=0,U_used_for_selection=False,settings_models_backgrounds_unchanged=True)
f['sources']=[ref(r['path']) for r in f['sources']];f['sources'].append(ref(OUT/'continue_pressure.py'))
save('capacity_preparation_revision.json',f['capacity_preparation_revision']);save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
stage=[read(OUT/'stage_B/B_S.json'),read(OUT/'stage_B/B_G.json')]
try:
 r=read(OUT/'raw_runs/B_G_SAFE.json');go=describe_B(r,f);stage.append(go);assert go['passed'],('G_SAFE_NOT_VALID',go['checks'])
 so=stage[0];cap=so['capacity']['actual_ASR_capacity_qps'];sd=next(x['service']['mean'] for x in so['paths'] if x['task']=='asr')/1000
 gd=next(x['service']['mean'] for x in go['paths'] if x['task']=='asr')/1000
 # Offered background demand uses complete native service samples; avoids low-throughput rho bias.
 bg=sum(.2*x['service']['mean']/1000 for x in go['paths'] if x['worker']=='qwen#0' and x['task']!='asr')
 q=float(np.floor(min(.9*cap,.9/sd)*100)/100);g10=bg+.1*q*gd
 assert g10<1,('SPLIT10_PROJECTED_CAPACITY_RISK',q,bg,gd)
 save('stage_B_pressure_preselection.json',dict(S_capacity_qps=cap,S_service_seconds=sd,G_service_seconds=gd,G_background_nominal_service_demand_ratio=bg,G_remaining_nominal_qps=(1-bg)/gd,selected_ASR_qps=q,split10_projected_G_demand=g10,rule=f['stage_B_protocol']['pressure_rate_rule'],U_not_used_for_selection=True,G_probe_reduced_once=True))
 pressure=[]
 for rid,p in [('B_PRESSURE_S',0),('B_PRESSURE_10',.1)]:
  s=slot(f,rid,q,p,120,320003,'B_PRESSURE')
  if pressure:s['trajectories']=copy.deepcopy(pressure[0]['trajectories'])
  pressure.append(s)
 save('stage_B_pressure_slots.json',pressure);po=[]
 for s in pressure:
  save('campaign_status.json',dict(status='STAGE_B_PRESSURE',run_id=s['run_id'],pid=os.getpid()))
  r=execute(s,f);o=describe_B(r,f);stage.append(o);po.append(o);assert o['passed'],('PRESSURE_NOT_LEGAL_STABLE',o['run_id'],o['checks'])
 wait=next(x['wait']['mean'] for x in po[0]['paths'] if x['task']=='asr');assert wait>200,('NO_OBSERVED_QUEUE_PRESSURE',wait)
 for rid,p in [('A0',0),('A1',.05),('A2',.1)]:
  s=slot(f,rid,q,p,600,320032,'C_EXPLORATORY')
  if f['slots']:s['trajectories']=copy.deepcopy(f['slots'][0]['trajectories'])
  ev=read(s['trajectories']['measurement']['path'])['events'];counts=collections.Counter(e['task'] for e in ev);g=sum(e['task']=='asr' and choose(e,s['routing'],s['deployment'])=='qwen#0' for e in ev)
  s.update(frozen_offered_counts=dict(counts),actual_planned_G_asr=g,actual_planned_G_fraction=g/counts['asr'])
  assert all(counts[t]>=72 for t in counts if t!='asr') and (p==0 or g>=20)
  f['slots'].append(s)
 f['formal_load_frozen_unix_ns']=time.time_ns();f['stage_B_evidence']=[ref(OUT/'stage_B'/ (x['run_id']+'.json')) for x in stage];f['split_fractions']=[.05,.1];f['split_revision_count']=0
 save('frozen_load.json',dict(at_unix_ns=f['formal_load_frozen_unix_ns'],ASR_qps=q,background_qps=.2,B3_budget_mib=34567.5534,slots=f['slots'],split_revision_count=0,selection_evidence=f['stage_B_evidence'],no_utility_result_used_for_selection=True))
 save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f);save('stage_B_gate.json',dict(passed=True,observed_S_mean_wait_ms=wait,selected_ASR_qps=q,split10_projected_G_demand=g10,formal_candidate='C1',candidate2_GPU_executed=False,G_overload_diagnostic_retained=True))
 save('colocation_profile.json',stage)
 for rid in f['formal_order']:
  s=next(s for s in f['slots'] if s['run_id']==rid);save('campaign_status.json',dict(status='STAGE_C',run_id=rid,pid=os.getpid()))
  r=execute(s,f);_,_,r31=setup();o=r31.assess(r);save('formal_acceptance/'+rid+'.json',o)
  print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['validation'],flush=True)
  assert o['technical_valid'],('FORMAL_TECHNICAL_FAILURE',rid,o['validation'])
 save('campaign_status.json',dict(status='COMPLETED',formal_arms=['A1','A0','A2'],independent_replication_executed=False))
except BaseException as ex:
 save('campaign_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE',error=repr(ex)));print('STOPPED',repr(ex),flush=True)
finally:
 save('colocation_profile.json',stage)
 from m32.report import generate
 generate()
