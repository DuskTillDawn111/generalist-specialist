import sys,math,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
from m32b.campaign import assess
from module26.common import cohort
from replica_scaling.telemetry import snapshot
from vqa_aware_dynamic_router import VQAAwareRouter
f=read(OUT/'experiment_manifest.json');checks={};bad=[]
environments={}
for py,prior in f['environment'].items():
 probe=subprocess.run([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True,capture_output=True,check=True)
 current=__import__('json').loads(probe.stdout);environments[py]=dict(unchanged=current==prior,current=current)
checks['worker_environments_unchanged']=all(x['unchanged'] for x in environments.values());save('final_environment_identity.json',environments)

for x in f['checkpoints']:
 if ref(x['path'])['sha256']!=x['sha256']:bad.append(x['path'])
checks['all_checkpoint_full_SHA']=not bad;checks['frozen_sources_inputs_SHA']=all(ref(x['path'])['sha256']==x['sha256'] for x in f['sources']+f['frozen_inputs']);before=read(OUT/'protected_before.json');changed=[p for p,v in before.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v];checks['historical_and_optimizer_metadata']=not changed
oldsha=read(OUT/'historical_module32_file_hashes.json');checks['all_Module32_full_SHA']=all(ref(x['path'])['sha256']==x['sha256'] for x in oldsha)
raws=[];assessed=[];replay=[]
for s in f['slots']:
 p=OUT/'raw_runs'/(s['run_id']+'.json')
 if not p.exists():continue
 r=read(p);raws.append(r);o=assess(r);assessed.append(dict(run_id=o['run_id'],eligible=o['eligible'],technical_valid=o['technical_valid'],U=o['U']));checks[s['run_id']+'_technical']=o['technical_valid'];checks[s['run_id']+'_journal']=o['journal_audit']['passed'];checks[s['run_id']+'_freeze_before_run']=f['frozen_unix_ns']<read(OUT/'run_attempts'/(s['run_id']+'.json'))['started_unix_ns']
 if s['arm']=='dynamic':
  for label,ph in [('warmup',r.get('warmup'))]+list(r.get('phases',{}).items()):
   if not ph:continue
   router=VQAAwareRouter(read(OUT/'frozen_routing_parameters.json'),read(OUT/'audio_metadata_cache.json'),ph['window']['duration_seconds']);trace=read(s['trajectories'][label]['path'])['events'];ev={f'{label}:{i}':e for i,e in enumerate(trace)}
   for d in ph['routing_decisions']:
    v=router.decide(d['decision_perf_ns'],ev[d['request_id']]['item']['sample']['input'],d['past_state_snapshot']);eq=all(native(v[k])==d[k] for k in v);past=all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']);replay.append(dict(phase=label,request_id=d['request_id'],exact_decision_replay=eq,only_past_host_times=past))
checks['three_formal']=len(raws)==3;checks['matched_inputs_arrivals']=len(raws)==3 and all(cohort(r)==cohort(raws[0]) for r in raws);checks['identical_pool_rates_budget']=len(raws)==3 and all(all(r['slot'][k]==raws[0]['slot'][k] for k in ['deployment','lambda_rps','safe_memory_budget_mib']) for r in raws);checks['causal_router_exact_replay']=bool(replay) and all(x['exact_decision_replay'] and x['only_past_host_times'] for x in replay)
v=snapshot();checks['GPU_released']=not v['processes'] and v['used_mib']<=128
save('causal_router_replay_audit.json',replay);save('final_data_acceptance.json',dict(passed=all(checks.values()),checks=checks,checkpoint_count=len(f['checkpoints']),bad_checkpoint_SHA=bad,protected_files=len(before),changed_history=changed,formal=assessed,GPU=v,at_unix_ns=time.time_ns(),independent_replication=False))
print('FINAL_AUDIT',checks,flush=True)
