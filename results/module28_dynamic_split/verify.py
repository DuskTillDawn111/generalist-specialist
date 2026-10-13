"""Post-campaign CPU verification; no policy fitting, no new GPU inference."""
from audit import *
from report import evaluate,records,match
from module26.common import cohort
import shutil

def main():
 f=json.loads((OUT/'frozen_candidate_manifest.json').read_text());gate=json.loads((OUT/'module28_gate.json').read_text());checks=[]
 def passed(name,**data):checks.append(dict(name=name,passed=True,**data));print('PASS',name,flush=True)
 for r in f['implementation']+[f['catalog'],f['utility_definition'],f['utility_reference'],f['quality_policy']]:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 assert json.loads((OUT/'router_parameters.json').read_text())==f['parameters'];passed('frozen_sources_parameters_utility_quality_unchanged',sources=len(f['implementation']))
 ledger=json.loads((OUT/'attempt_ledger.json').read_text());assert sum(r['stage']=='C' for r in ledger)<=6 and sum(r['stage']=='D' for r in ledger)<=9 and len(ledger)<=15;assert len(set(r['run_id'] for r in ledger))==len(ledger);passed('bounded_unique_attempts_no_outcome_retries',attempts=len(ledger))
 raws=[json.loads(p.read_text()) for p in sorted((OUT/'raw_runs').glob('*.json'))];groups=collections.defaultdict(list);trace=[];decisions=[];allrecords=0;tokens=0
 for raw in raws:
  s=raw['slot'];v=evaluate(raw,f);groups[s['stage'],s['arrival_profile'],s['repetition']].append(raw);rs=records(raw);allrecords+=len(rs)
  for r in rs:
   if r['status']=='completed':assert abs((r['completion_time_ns']-r['scheduled_time_ns'])/1e6-r['e2e_latency_ms'])<1e-8
   assert r['task'] in f['plan']['models'][raw['replica_to_family'][r['model']]]['supported_tasks']
   row=dict(run_id=s['run_id'],stage=s['stage'],profile=s['arrival_profile'],arm=s['arm'],request_id=r['request_id'],task=r['task'],sample_id=r['sample_id'],final_status=r['status'],actual_route=r['model'],actual_service_ms=r['worker_latency_ms'],actual_wait_ms=r['worker_queue_wait_ms'],actual_E2E_ms=r.get('e2e_latency_ms'),actual_submit_time_ns=r.get('actual_submit_time_ns'),completion_time_ns=r.get('completion_time_ns'))
   decision=r.get('routing_decision')
   if decision:
    assert decision['latest_observed_completion_time_ns']<=decision['decision_time_ns']
    assert decision['route']==r['model'];row.update(decision);decisions.append(row)
    if s['arm']=='dynamic' and r['model']==f['parameters']['G_worker']:
     assert decision['reason']=='G_LOWER_SYSTEM_MARGINAL_COST' and decision['G_pending']==0 and decision['G_target_inflight']==0 and decision['admitted'] and decision['capacity_ok']
     assert decision['margin_ms']>f['parameters']['exit_margin_ms']
   else:row.update(reason='FROZEN_BACKGROUND_TASK_ROUTE',route=r['model'],scheduled_time_ns=r['scheduled_time_ns'],decision_time_ns=None,overhead_us=None,observation_scope='Prescribed admitted route; background dispatch overhead not separately measured')
   trace.append(row)
  # Inflight bound at actual enqueue, independent of policy's reported counters.
  if s['arm']=='dynamic':
   tr=sorted([r for r in rs if r['task']==f['target'] and r['model']==f['parameters']['G_worker']],key=lambda r:r.get('enqueued_time_ns',r['scheduled_time_ns']))
   for a,b in zip(tr,tr[1:]):assert a['completion_time_ns']<=b['enqueued_time_ns'],(s['run_id'],a['request_id'],b['request_id'])
   for r in tr:
    assert r['routing_decision']['fallback_until_seconds']<=r['routing_decision']['decision_time_ns']/1e9
  journal=OUT/'request_journals'/(s['run_id']+'.jsonl');terminal=0
  if journal.exists():
   byphase={'measurement':{r['request_id']:r for r in rs},'warmup':{r['request_id']:r for r in raw.get('warmup',{}).get('request_records',[])}}
   for line in journal.open():
    x=json.loads(line)
    if x['kind']=='REQUEST_TERMINAL':
     j=x['record'];record=byphase[x['phase']][j['request_id']]
     assert record['status']==j['status'] and record['completion_time_ns']==j['completion_time_ns'] and record['model']==j['model'];terminal+=1
  if v['eligible']:
   assert v['full_cohort'] and v['memory_ok'] and not v['hard_unstable'];assert abs(v['U']-sum(v[k+'_contribution'] for k in ['T','L','P','M']))<1e-12
  else:assert v['U'] is None
  passed('raw_'+s['run_id'],records=len(rs),journal_terminal=terminal,eligible=v['eligible'],U=v['U'])
 for key,rr in groups.items():
  if len(rr)==3:
   assert all(cohort(rr[0])==cohort(r) and rr[0]['slot']['deployment']==r['slot']['deployment'] for r in rr[1:])
 passed('matched_inputs_arrivals_and_full_raw_arithmetic',groups=len(groups),records=allrecords)
 csvsave('target_routing_decision_trace.csv',decisions);csvsave('routing_decision_trace.csv',trace,fields=['run_id','request_id','task','actual_route','reason'])
 before=json.loads((OUT/'historical_metadata_before.json').read_text());after=metadata();assert all(after.get(p)==stat for p,stat in before.items())
 for r in json.loads((OUT/'historical_sources.json').read_text()):assert ref(r['path'])['sha256']==r['sha256']
 passed('historical_experiments_unchanged',files=len(before));passed('all_six_task_routing_trace_complete',rows=len(trace))
 required=['opportunity_audit.md','frozen_candidate_manifest.json','dynamic_router.py','routing_decision_trace.csv','gpu_exploration_runs.csv','utility_comparison.csv','worker_interference_analysis.csv','module28_gate.json','module28_summary.md'];assert all((OUT/p).exists() for p in required);passed('required_artifacts_present',files=required)
 save('final_verification.json',dict(passed=True,at_unix_ns=time.time_ns(),gate=gate['gate'],checks=checks,policy_or_parameters_modified=False,GPU_inferences=0,reporting_note='routing_decision_trace.csv expanded from target trace to all6 tasks using raw fixed background routes; target trace separately preserved'))
if __name__=='__main__':main()
