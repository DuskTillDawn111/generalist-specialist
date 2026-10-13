import sys,shutil,time,fcntl,traceback,collections
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
from m32.campaign import execute,describe_B,setup
import numpy as np
from module21.trajectory import choose
from replica_scaling.telemetry import snapshot
G='phi4mm'
def c2slot(f,rid,q,p,dur,seed,stage):
 s=slot(f,rid,q,p,dur,seed,stage);s.update(candidate='C2',Generalist=G)
 s['deployment'].pop('qwen');s['deployment'].update(phi4mm=1,**{'whisper-small':1})
 s['routing']['lid']={'whisper-small':.2};s['routing']['visual_qa']={G:.2};s['routing']['asr']={S:q*(1-p)}
 if p:s['routing']['asr'][G]=q*p
 return s
lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
f=read(OUT/'experiment_manifest.json');assert not f['slots'] and f['formal_candidate']=='C2'
assert not snapshot()['processes'];(OUT/'campaign_failure.json').unlink()
f['sources']=[ref(r['path']) for r in f['sources']]+[ref(OUT/'candidate2_continue.py'),ref(ROOT/'candidate_pool_v2_1_adapters/vqa_extension.py')]
dependency=read(OUT/'candidate2_dependency_freeze.json');f['environment'].update(dependency['environment'])
f['frozen_inputs'].append(ref(OUT/'candidate2_dependency_freeze.json'))
f['positive_gate_original_module30_diagnostics']=f['positive_gate']
f['positive_gate']=dict(delta_U_min=.01,all_three_arms_legal_stable=True,matched_trace=True,utility='Module20A unchanged',stability='Original stability_v2 unchanged',nonzero_actual_G_and_S_execution=True,scope='User Module32 exploratory criterion; original Module30 auxiliary sample/thermal diagnostics retained but not substituted for the requested gate')
f['candidate2_upper_probe_reuse']=dict(at_unix_ns=time.time_ns(),raw=ref(OUT/'raw_runs/B_C2_S.json'),reason='Upper-load capacity diagnostic is not a comparator; original growth label and NULL U retained. Select the first pressure point using the already-frozen 80% measured-capacity rule.',no_GPU_repeat=True,no_new_rate_rule=True)
save('candidate2_pressure_preparation_revision.json',f['candidate2_upper_probe_reuse']);save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
stage=read(OUT/'colocation_profile.json')
try:
 r=read(OUT/'raw_runs/B_C2_S.json');so=describe_B(r,f)
 assert r['valid_session'] and r['capacity_probe']['all_completed'],('C2_CAPACITY_DATA_INVALID',r['errors'])
 # The initial upper-load diagnostic may be unstable; it never receives utility.
 cap=r['capacity_probe']['actual_ASR_capacity_qps'];sd=next(x['service']['mean'] for x in so['paths'] if x['task']=='asr')/1000
 s=c2slot(f,'B_C2_G',.15,1,120,320202,'B_C2_G');save('candidate2_initial_G_slot.json',s)
 save('campaign_status.json',dict(status='STAGE_B_C2',run_id=s['run_id'],pid=os.getpid()))
 r=execute(s,f);go=describe_B(r,f);stage.append(go);assert go['passed'],('C2_G_NOT_VALID',go.get('checks'))
 gd=next(x['service']['mean'] for x in go['paths'] if x['task']=='asr')/1000;bg=sum(.2*x['service']['mean']/1000 for x in go['paths'] if x['worker']==G+'#0' and x['task']!='asr')
 q=float(np.floor(min(.8*cap,.8/sd)*100)/100);fractions=[.05,.1];reason=None
 if bg+.1*q*gd>=.85:
  upper=(.75-bg)/(q*gd);fractions=[min(.025,upper/2),min(.05,upper)];reason='Measured C2 G demand has clear capacity risk at 10%; one pre-formal smaller split freeze'
  assert min(fractions)>0
 save('candidate2_pressure_preselection.json',dict(S_capacity_qps=cap,S_native_service_seconds=sd,G_native_service_seconds=gd,G_background_demand_ratio=bg,G_nominal_remaining_ASR_qps=(1-bg)/gd,selected_ASR_qps=q,split_fractions=fractions,split_change_reason=reason,U_used=False,rule=f['candidate2_stage_B_protocol']['pressure_rule']))
 pressure=[]
 for rid,p in [('B_C2_PRESSURE_S',0),('B_C2_PRESSURE_G',fractions[1])]:
  s=c2slot(f,rid,q,p,120,320203,'B_C2_PRESSURE')
  if pressure:s['trajectories']=copy.deepcopy(pressure[0]['trajectories'])
  pressure.append(s)
 save('candidate2_pressure_slots.json',pressure);po=[]
 for s in pressure:
  save('campaign_status.json',dict(status='STAGE_B_C2_PRESSURE',run_id=s['run_id'],pid=os.getpid()))
  r=execute(s,f);o=describe_B(r,f);stage.append(o);po.append(o);assert o['passed'],('C2_PRESSURE_NOT_VALID',o['run_id'],o.get('checks'))
 wait=next(x['wait']['mean'] for x in po[0]['paths'] if x['task']=='asr');assert wait>200,('C2_NO_OBSERVED_QUEUE_PRESSURE',wait)
 f['slots']=[]
 for rid,p,arm in [('A0',0,'s_only'),('A1',fractions[0],'split05'),('A2',fractions[1],'split10')]:
  s=c2slot(f,rid,q,p,600,320232,'C_EXPLORATORY');s['arm']=arm
  if f['slots']:s['trajectories']=copy.deepcopy(f['slots'][0]['trajectories'])
  ev=read(s['trajectories']['measurement']['path'])['events'];counts=collections.Counter(e['task'] for e in ev);g=sum(e['task']=='asr' and choose(e,s['routing'],s['deployment'])==G+'#0' for e in ev)
  s.update(frozen_offered_counts=dict(counts),actual_planned_G_asr=g,actual_planned_G_fraction=g/counts['asr'])
  assert all(counts[t]>=72 for t in counts if t!='asr') and (p==0 or g>0)
  f['slots'].append(s)
 f['formal_load_frozen_unix_ns']=time.time_ns();f['split_fractions']=fractions;f['split_revision_count']=int(reason is not None);f['stage_B_evidence']=[ref(OUT/'stage_B'/ (x['run_id']+'.json')) for x in stage]
 save('frozen_load.json',dict(at_unix_ns=f['formal_load_frozen_unix_ns'],ASR_qps=q,background_qps=.2,B3_budget_mib=34567.5534,slots=f['slots'],split_revision_count=f['split_revision_count'],split_change_reason=reason,selection_evidence=f['stage_B_evidence'],no_utility_result_used_for_selection=True))
 save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f);save('stage_B_gate.json',dict(passed=True,observed_S_mean_wait_ms=wait,selected_ASR_qps=q,formal_candidate='C2',candidate1_GPU_formal_executed=False,candidate2_GPU_executed=True))
 save('colocation_profile.json',stage)
 for rid in f['formal_order']:
  s=next(s for s in f['slots'] if s['run_id']==rid);save('campaign_status.json',dict(status='STAGE_C',run_id=rid,pid=os.getpid()))
  r=execute(s,f);_,_,r31=setup();o=r31.assess(r);save('formal_acceptance/'+rid+'.json',o)
  print('FORMAL_ACCEPT',rid,o['eligible'],o['U'],o['validation'],flush=True)
  assert o['technical_valid'],('FORMAL_TECHNICAL_FAILURE',rid,o['validation'])
 save('campaign_status.json',dict(status='COMPLETED',formal_candidate='C2',formal_arms=['A1','A0','A2'],independent_replication_executed=False))
except BaseException as ex:
 save('campaign_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));save('campaign_status.json',dict(status='STOPPED_INCONCLUSIVE',error=repr(ex)));print('STOPPED',repr(ex),flush=True)
finally:
 save('colocation_profile.json',stage)
 from m32.report import generate
 generate()
