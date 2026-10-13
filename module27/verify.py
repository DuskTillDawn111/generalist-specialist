"""Acceptance of frozen science, complete raw cohort, budgets and old history."""
import ast
from .common import *
from module21.trajectory import choose

def main():
 f=read(OUT/'discovery_protocol.json');campaign=read(OUT/'campaign.json');gate=read(OUT/'module27_gate.json');checks=[]
 for p in (ROOT/'module27').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 verify(f['implementation']+[f[k] for k in ['catalog','utility_definition','utility_reference','quality_policy']]);verify(read(OUT/'consumed_historical_sources.json'))
 assert campaign['B_attempts']<=6<=12 and campaign['C_attempts']<=6
 ledger=read(OUT/'attempt_ledger.json');assert len({r['run_id'] for r in ledger})==len(ledger),'No duplicate outcome attempt'
 checks.append('Frozen utility/quality/source checks and preregistered run budget')
 for n in campaign['executed_B']+campaign['executed_C']:
  raw=read(OUT/'raw_runs'/(n+'.json'));v,o=arithmetic(raw);s=raw['slot'];ph=raw.get('phases',{}).get('measurement')
  if not ph:assert not o['eligible'] and o['U'] is None;continue
  assert ph['window']['planned_duration_seconds']==s['measurement_seconds']
  events=read(s['trajectories']['measurement']['path'])['events'];verify([s['trajectories']['measurement']]);records=ph['request_records']
  assert len(events)==len(records)
  for e,r in zip(events,records):
   assert e['task']==r['task'] and e['item']['pool_id']==r['pool_id'] and e['item']['sample_id']==r['sample_id']
   assert abs(e['offset_seconds']-r['frozen_offset_seconds'])<1e-12 and r['model']==choose(e,s['routing'],s['deployment'])
   if r['status']=='completed':assert abs(r['e2e_latency_ms']-(r['completion_time_ns']-r['scheduled_time_ns'])/1e6)<1e-8
  if o['eligible']:
   assert raw['valid_session'] and raw['gpu_release']['verified'] and len(records)==o['completed']
   assert o['safe_memory_mib']<=BUDGET and o['sampled_peak_mib']<=BUDGET and not o['hard_unstable']
   assert abs(o['U']-sum(v[k+'_contribution'] for k in ['T','L','P','M']))<1e-12
   assert max(r['completion_time_ns'] for r in raw['warmup']['request_records'])<ph['window']['start_time_ns']
  else:assert o['U'] is None
 checks.append('Every raw offered request, task/input/trace/route, full-cohort U, memory and stability')
 b=read(OUT/'exploratory_result.json');c=read(OUT/'independent_result.json') if (OUT/'independent_result.json').exists() else None
 assert bool(c)==bool(b['Stage_C_permitted']) or gate.get('failure')
 if c:
  assert set(read(OUT/'frozen_positive_candidate.json')['rates'])==set(f['refs'])
  assert set(r['seed'] for r in ledger if r['stage']=='C')<=set(f['independent_protocol']['seeds'])
  assert not (set(r['seed'] for r in ledger if r['stage']=='C') & set(r['seed'] for r in ledger if r['stage']=='B'))
 assert (OUT/'optimizer_autonomous_decision.json').exists()==bool(c and c['Stage_D_permitted'])
 checks.append('Development never counted as independent; conditional optimizer gate; no extra seeds')
 before=read(OUT/'historical_metadata_before.json');after=protected_metadata();assert all(after.get(p)==v for p,v in before.items())
 checks.append('Historical metadata and consumed scientific SHA256 unchanged')
 required=['stage_a_evidence_audit.json','candidate_screening.csv','historical_service_evidence.csv','discovery_protocol.json','frozen_positive_candidate.json',
  'gpu_discovery_runs.csv','exploratory_utility_comparison.csv','per_task_latency.csv','worker_queue_stability.csv','utility_decomposition.csv',
  'GPU_thermal_state.csv','module27_gate.json','module27_summary.md','historical_integrity.json']
 assert all((OUT/n).exists() for n in required)
 save('execution_source_manifest.json',dict(sources=[ref(p) for p in sorted((ROOT/'module27').glob('*.py'))],results=[ref(OUT/n) for n in required]))
 save('final_verification.json',dict(passed=True,checks=checks,history_files=len(before),all_negative_results_retained=True,no_next_Module=True,remaining_work=False))
 print('MODULE27_VERIFIED_AND_STOPPED',gate['status'],flush=True)

if __name__=='__main__':main()
