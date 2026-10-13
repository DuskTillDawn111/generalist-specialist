"""CPU-only invariant checks, admission provenance and sample accessibility."""
import ast,collections
from .common import *
from joint_deploy.measurement.load_generator import SamplePool
from module21.trajectory import choose

def main():
 f=read(OUT/'cpu_screening.json');base=read(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json');heavy=read(ROOT/'results/module14_high_quality_heavy/quality_nondegraded_heavy_pool.json');quality=read(f['quality_policy']['path']);thresholds={r['task']:r['historical_threshold'] for r in quality['tasks']}
 rows=[]
 for slot in f['slots']:
  assert len(slot['lambda_rps'])==6 and all(q>0 for q in slot['lambda_rps'].values())
  assert slot['measurement_seconds']==600 and sum(slot['deployment'].values())==5
  for t,routing in slot['routing'].items():
   assert abs(sum(routing.values())-slot['lambda_rps'][t])<1e-12
   for m,q in routing.items():
    assert t in f['plan']['models'][m]['supported_tasks']
    if m.startswith('baudm/parseq:'):
     measured=list(csv.DictReader(open(ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv')))
     r=next(r for r in measured if r['model_id']+'@'+r['revision']==m);value=float(r['quality']);passed=value>=thresholds[t] and r['measurement_complete']=='True'
     evidence=dict(model=m,task=t,quality=value,threshold=thresholds[t],passed=passed,quality_raw=r['quality_raw'],source='Module14B OCR gap-fill exact checkpoint measurement')
    elif m in heavy['models']:
     value=heavy['models'][m]['tasks'][t]['quality'];passed=value>=thresholds[t]
     evidence=dict(model=m,task=t,quality=value,threshold=thresholds[t],passed=passed,source='Module14B measured quality nondegraded pool')
    else:
     r=next(r for r in base['rows'] if r['model']==m and r['task']==t);passed=r['planning_admissible'];evidence={**r,'passed':passed}
    assert passed,(slot['run_id'],m,t);rows.append(dict(run_id=slot['run_id'],**evidence))
  tr=read(slot['trajectories']['measurement']['path']);counts=collections.Counter();worker_counts=collections.Counter()
  for e in tr['events']:
   assert 0<=e['offset_seconds']<600
   counts[e['task']]+=1;worker_counts[choose(e,slot['routing'],slot['deployment'])]+=1
   value=e['item']['sample'].get('input')
   if isinstance(value,str) and value.startswith('/'):assert Path(value).exists(),value
   assert e['item']['task_type']==e['task']
  assert all(counts[t]>=72 for t in slot['lambda_rps'] if t!='asr'),counts
  if slot['arm']!='s_only':assert worker_counts[slot['Generalist']+'#0']-counts['visual_qa']>=20
 for candidate in f['candidates']:
  group=[s for s in f['slots'] if s['candidate']==candidate['candidate']]
  assert len(group)==3
  assert len({s['trajectories']['measurement']['sha256'] for s in group})==1
  assert all(s['deployment']==group[0]['deployment'] and s['lambda_rps']==group[0]['lambda_rps'] for s in group)
 for v in f['plan']['registry'].values():assert Path(v['python']).exists() and Path(v['script']).exists()
 for p in (ROOT/'module30').glob('*.py'):ast.parse(p.read_text())
 csvsave('quality_admission_provenance.csv',rows)
 refs=[ref(ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json'),ref(ROOT/'results/module14_high_quality_heavy/quality_nondegraded_heavy_pool.json'),ref(ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv')]
 save('cpu_preflight.json',dict(passed=True,admission_paths=len(rows),sample_files_accessible=True,matched_six_task_trajectories=True,routing_conserved=True,
  minimum_G_requests=20,minimum_background_requests=72,model_weights_or_inference_started=False,quality_evidence=refs))
 print('CPU_PREFLIGHT_PASSED',len(rows),flush=True)
if __name__=='__main__':main()
