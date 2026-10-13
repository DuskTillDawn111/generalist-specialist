import sys,collections,time
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
from m32.campaign import setup
from module26.common import arithmetic,cohort
from module25.common import utility_inputs
import numpy as np
from replica_scaling.telemetry import snapshot
f=read(OUT/'experiment_manifest.json');checks={};badcp=[]
for r in f['checkpoints']:
 if ref(r['path'])['sha256']!=r['sha256']:badcp.append(r['path'])
checks['full_checkpoint_hashes']=not badcp
checks['frozen_source_hashes']=all(ref(r['path'])['sha256']==r['sha256'] for r in f['sources']+f['frozen_inputs'])
before=read(OUT/'protected_before.json');changed=[p for p,v in before.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
checks['historical_metadata']=not changed
_,rep,r31=setup();raws=[];aud=[];inputfiles={}
for s in f['slots']:
 p=OUT/'raw_runs'/(s['run_id']+'.json')
 if not p.exists():continue
 r=read(p);raws.append(r);o=r31.assess(r);checks[s['run_id']+'_technical']=o['technical_valid'];checks[s['run_id']+'_journal']=rep.journal_audit(r)['passed']
 for phase in ['warmup','measurement']:
  tr=read(s['trajectories'][phase]['path'])
  for e in tr['events']:
   val=e['item']['sample'].get('input')
   if isinstance(val,str) and val.startswith('/'):inputfiles[val]=ref(val)
 checks[s['run_id']+'_600s']=r.get('phases',{}).get('measurement',{}).get('window',{}).get('duration_seconds')==600
 checks[s['run_id']+'_no_errors']=not r['errors']
 aud.append(dict(run_id=s['run_id'],eligible=o['eligible'],U=o['U'],validation=o['validation']))
gate=read(OUT/'module32_gate_definition.json')
checks['gate_frozen_before_formal']=all(gate['frozen_unix_ns']<read(OUT/'run_attempts'/(r['slot']['run_id']+'.json'))['started_unix_ns'] for r in raws)
checks['three_formal']=len(raws)==3
checks['matched_cohorts']=len(raws)==3 and all(r.get('phases',{}).get('measurement') for r in raws) and all(cohort(r)==cohort(raws[0]) for r in raws)
checks['same_models_rates_budget']=len(raws)==3 and all(all(r['slot'][k]==raws[0]['slot'][k] for k in ['deployment','lambda_rps','safe_memory_budget_mib']) for r in raws)
# Compare actual input files with prior frozen hashes where available.
old={r['path']:r for r in f.get('request_input_files',[])}
checks['known_input_hashes_unchanged']=all(v['sha256']==old[p]['sha256'] for p,v in inputfiles.items() if p in old)
checks['all_input_files_previously_frozen']=all(p in old for p in inputfiles)
gpu=snapshot();checks['GPU_released']=not gpu['processes'] and gpu['used_mib']<=128
save('final_request_input_files.json',list(inputfiles.values()))
save('final_data_acceptance.json',dict(passed=all(checks.values()),checks=checks,checkpoint_count=len(f['checkpoints']),bad_checkpoint_hashes=badcp,changed_history=changed,formal=aud,GPU=gpu,finished_unix_ns=time.time_ns(),independent_replication_executed=False))
print('FINAL_AUDIT',checks,flush=True)
