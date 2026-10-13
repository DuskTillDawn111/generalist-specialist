import sys, subprocess, ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
from m32c.campaign import assess
from module26.common import cohort
from replica_scaling.telemetry import snapshot
from vqa_aware_dynamic_router import VQAAwareRouter
f=read(OUT/'experiment_manifest.json');checks={};bad=[];env={};replay=[];formal=[];raws={}
for py,prior in f['environment'].items():
 p=subprocess.run([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True,capture_output=True,check=True)
 current=json.loads(p.stdout);env[py]=dict(unchanged=current==prior,current=current)
checks['worker_environments_unchanged']=all(x['unchanged'] for x in env.values());save('final_environment_identity.json',env)
for r in f['checkpoints']:
 if ref(r['path'])['sha256']!=r['sha256']:bad.append(r['path'])
checks['all_checkpoints_full_SHA']=not bad;checks['frozen_sources_inputs_SHA']=all(ref(r['path'])['sha256']==r['sha256'] for r in f['sources']+f['frozen_inputs'])
protected=read(OUT/'protected_before.json');changed=[p for p,v in protected.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v];checks['history_and_optimizer_metadata']=not changed
checks['all_Module32B_full_SHA']=all(ref(r['path'])['sha256']==r['sha256'] for r in read(OUT/'module32b_full_file_hashes.json'))
checks['all_Module32_full_SHA']=all(ref(r['path'])['sha256']==r['sha256'] for r in read(OLD/'historical_module32_file_hashes.json'))
checks['original_router_byte_identical']=ref(OUT/'vqa_aware_dynamic_router.py')['sha256']==ref(OLD/'vqa_aware_dynamic_router.py')['sha256'];checks['original_parameters_byte_identical']=ref(OUT/'frozen_routing_parameters.json')['sha256']==ref(OLD/'frozen_routing_parameters.json')['sha256']
for s in f['slots']:
 p=OUT/'raw_runs'/(s['run_id']+'.json')
 if not p.exists():continue
 r=read(p);raws[s['run_id']]=r;o=assess(r);formal.append(dict(run_id=o['run_id'],eligible=o['eligible'],technical_valid=o['technical_valid'],U=o['U'],stability=o.get('stability')))
 checks[s['run_id']+'_technical']=o['technical_valid'];checks[s['run_id']+'_journal']=o['journal_audit']['passed'];checks[s['run_id']+'_freeze_before_run']=f['frozen_unix_ns']<read(OUT/'run_attempts'/(s['run_id']+'.json'))['started_unix_ns']
 if s['arm']=='dynamic':
  for label,ph in [('warmup',r.get('warmup')),*r.get('phases',{}).items()]:
   if not ph:continue
   router=VQAAwareRouter(read(OUT/'frozen_routing_parameters.json'),read(OUT/'audio_metadata_cache.json'),ph['window']['duration_seconds']);events={f'{label}:{i}':e for i,e in enumerate(read(s['trajectories'][label]['path'])['events'])}
   for d in ph['routing_decisions']:
    v=router.decide(d['decision_perf_ns'],events[d['request_id']]['item']['sample']['input'],d['past_state_snapshot']);eq=all(native(v[k])==d[k] for k in v);past=all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']);replay.append(dict(run_id=s['run_id'],phase=label,request_id=d['request_id'],exact_replay=eq,past_only=past))
checks['six_formal_runs']=len(raws)==6
for pair in range(1,4):
 a=raws.get(f'P{pair}_S');b=raws.get(f'P{pair}_D');checks[f'P{pair}_same_cohort']=bool(a and b and cohort(a)==cohort(b));checks[f'P{pair}_full_inputs']=bool(a and b and read(a['slot']['trajectories']['measurement']['path'])['events']==read(b['slot']['trajectories']['measurement']['path'])['events'])
checks['pool_rates_budget_identical']=bool(raws) and all(all(r['slot'][k]==f['slots'][0][k] for k in ['deployment','lambda_rps','safe_memory_budget_mib']) for r in raws.values())
checks['causal_router_exact_replay']=bool(replay) and all(x['exact_replay'] and x['past_only'] for x in replay)
checks['no_more_than_six_attempts']=len(list((OUT/'run_attempts').glob('*.json')))<=6
checks['six_frozen_trajectory_hashes']=len({s['trajectories'][label]['sha256'] for s in f['slots'] for label in ['warmup','measurement']})==6
v=snapshot();checks['GPU_released']=not v['processes'] and v['used_mib']<=128
integrity_keys=['worker_environments_unchanged','all_checkpoints_full_SHA','frozen_sources_inputs_SHA','history_and_optimizer_metadata','all_Module32B_full_SHA','all_Module32_full_SHA','original_router_byte_identical','original_parameters_byte_identical','pool_rates_budget_identical','causal_router_exact_replay','no_more_than_six_attempts','six_frozen_trajectory_hashes','GPU_released']
integrity_keys += [k for k in checks if k.endswith('_journal') or k.endswith('_freeze_before_run')]
integrity_checks={k:checks[k] for k in integrity_keys}
save('causal_router_replay_audit.json',replay);save('final_data_acceptance.json',dict(passed=all(checks.values()),checks=checks,integrity_passed=all(integrity_checks.values()),integrity_checks=integrity_checks,history_preserved=not changed and checks['all_Module32B_full_SHA'] and checks['all_Module32_full_SHA'],bad_checkpoint_SHA=bad,checkpoint_count=len(f['checkpoints']),protected_files=len(protected),changed_history=changed,formal=formal,GPU=v,at_unix_ns=time.time_ns(),independent_replication=True))
print('FINAL_AUDIT',checks,flush=True)
