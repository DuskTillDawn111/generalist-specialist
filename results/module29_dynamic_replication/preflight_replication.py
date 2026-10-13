"""CPU checks of safety transitions, observation boundary, and source syntax."""
from common_replication import *
from dynamic_router import DynamicRouter
from gpu_replication import VisibleRuntime
from types import SimpleNamespace

def router_checks():
 for p in OUT.glob('*.py'):compile(p.read_text(),str(p),'exec')
 p=dict(G_worker='g',S_worker='s',target='lid',S_lower_by_task_ms={'lid':65.,'asr':300.},G_upper_by_task_ms={'lid':520.,'visual_qa':650.},G_service_ms=520.,S_inflation_per_pending_ms=10.,background_cost_ms=100.,G_max_target_inflight=1,G_token_rate_rps=.575,cooldown_seconds=.52,entry_margin_ms=65.,exit_margin_ms=32.5,ewma_alpha=.1,prediction_error_ratio=1.5,fallback_seconds=30.,S_service_alarm_ms=118.)
 pending={'s':[dict(task='lid',request_id=str(i)) for i in range(20)]}
 router=DynamicRouter(p);w,d=router.decide('a',100.,pending);assert w=='g' and d['reason']=='G_LOWER_SYSTEM_MARGINAL_COST'
 pending_busy={**pending,'g':[dict(task='lid',request_id='a')]};assert router.decide('b',100.1,pending_busy)[1]['reason']=='G_INFLIGHT_BACKPRESSURE'
 assert DynamicRouter(p).decide('c',100.,pending,False,True)[0]=='s'
 assert DynamicRouter(p).decide('d',100.,pending,True,False)[0]=='s'
 assert DynamicRouter(p).decide('e',100.,{'s':[dict(task='lid')]})[0]=='s'
 bg={**pending,'g':[dict(task='visual_qa')]};assert DynamicRouter(p).decide('f',100.,bg)[1]['reason']=='G_BACKGROUND_OR_OTHER_WORK_PENDING'
 assert router.decide('t',100.2,pending)[1]['reason']=='G_TOKEN_BACKPRESSURE'
 router.observe_completion(dict(request_id='a',model='g',task='lid',worker_latency_ms=1000.),101.);assert router.decide('h',103.,pending)[1]['reason']=='PREDICTION_ERROR_FALLBACK'
 # Service/result fields on a submitted row must not enter an arrival snapshot.
 rt=VisibleRuntime(SimpleNamespace());rt.visible={'future':dict(status='submitted',request_id='future',model='s',task='lid',enqueued_time_ns=1,actual_submit_time_ns=2,completion_time_ns=99999,worker_response={'prediction':'secret'},worker_latency_ms=1234.),'received':dict(status='completed',request_id='received',model='s',task='lid',enqueued_time_ns=1,completion_time_ns=9,worker_latency_ms=65.),'not_received_yet':dict(status='completed',request_id='not_received_yet',model='g',task='lid',completion_time_ns=200)}
 snap=rt.arrival_snapshot(10);assert list(snap)==['s'] and len(snap['s'])==1;assert not any(k in snap['s'][0] for k in ['completion_time_ns','worker_response','worker_latency_ms']);assert rt.latest_received_ns==9 and 'not_received_yet' in rt.visible
 save('cpu_preflight.json',dict(passed=True,checks=['all_source_syntax','positive_proxy_routes_G','one_inflight_backpressure','token_backpressure','background_protection','admission_and_capacity_fallback','low_queue_S','large_service_error_fallback','future_fields_excluded_from_live_snapshot','unreceived_completion_not_observed'],new_gpu_inferences=0))
 print('CPU_ROUTER_PREFLIGHT_PASS')

def main():
 import tempfile,copy
 import common_replication as common
 import report_replication as reports
 from module21.trajectory import arrivals,choose
 f=json.loads((OUT/'frozen_replication_manifest.json').read_text())
 router_checks()
 dev=ROOT/'results/module28r_dynamic_followup'
 parity=[]
 for rid in ['R2_LOW_dynamic','R4_LOW_s_only','R6_LOW_fixed10']:
  raw=json.loads((dev/'raw_runs'/(rid+'.json')).read_text())
  v=reports.original_report.evaluate(raw,f)
  expected=next(r for r in csv.DictReader((dev/'system_utility_comparison.csv').open()) if r['run_id']==rid)
  assert v['eligible'] and abs(v['U']-float(expected['U']))<1e-12
  parity.append(dict(development_run=rid,arithmetic_match=True,excluded_from_replication=True))
 for slot in f['independent_slots']:
  for label,r in slot['trajectories'].items():
   tr=json.loads(Path(r['path']).read_text());events=arrivals(f['rates'],tr['duration'],slot['seed'],label)
   assert len(events)==len(tr['events'])
   assert all(all(e[k]==q[k] for k in e) for e,q in zip(events,tr['events']))
   for e in tr['events']:
    worker=choose(e,slot['routing'],slot['deployment']);m=worker.rsplit('#',1)[0]
    assert e['task'] in f['plan']['models'][m]['supported_tasks']
    for k in ['audio_path','image_path']:
     if e['item']['sample'].get(k):assert Path(e['item']['sample'][k]).is_file()
 cases=[(False,False,False,[],[], 'REPLICATION_INVALID'),
        (True,True,True,[.02]*5+[-.02],[dict(lower=.001)]*2,'DYNAMIC_NOT_REPRODUCED'),
        (True,True,False,[.02]*6,[dict(lower=.001)]*2,'DYNAMIC_NOT_REPRODUCED'),
        (True,True,True,[.02]*6,[dict(lower=-.001),dict(lower=.001)],'DYNAMIC_POSITIVE_INCONCLUSIVE'),
        (True,True,True,[.02]*6,[dict(lower=.001)]*2,'DYNAMIC_POSITIVE_REPLICATED')]
 for complete,valid,both,deltas,cis,expected in cases:assert reports.decide_gate(complete,valid,both,deltas,cis)==expected
 assert reports.paired_stats([1.,2.,3.])['n']==3
 # End-to-end serialization/branch test uses development raw data ONLY in a
 # temporary CPU fixture directory. It does not enter the science output.
 actual=common.OUT
 with tempfile.TemporaryDirectory(prefix='module29_cpu_fixture_') as directory:
  common.OUT=Path(directory);reports.OUT=Path(directory)
  fixtures={a:json.loads((dev/'raw_runs'/(rid+'.json')).read_text()) for a,rid in [('dynamic','R2_LOW_dynamic'),('s_only','R4_LOW_s_only'),('fixed10','R6_LOW_fixed10')]}
  try:
   for slot in f['independent_slots']:
    raw=copy.deepcopy(fixtures[slot['arm']]);raw['slot'].update(run_id=slot['run_id'],group=slot['group'],run_order=slot['run_order'],seed=slot['seed'])
    common.save('raw_runs/'+slot['run_id']+'.json',raw);reports.describe_run(raw,f)
   smoke=reports.write_report(f,final=True)
   assert len(smoke['pairs'])==6 and smoke['gate']=='DYNAMIC_POSITIVE_REPLICATED'
   reports.write_report(f,fatal=dict(reason='CPU_ONLY_FAILURE_FIXTURE'),final=True)
  finally:common.OUT=actual;reports.OUT=actual
 common.save('cpu_preflight.json',dict(passed=True,new_gpu_inferences=0,
  checks=['source_compile','deterministic_matched_generator','six_task_coverage_and_admitted_routes','original_router_safety_and_observation_boundary','20A_raw_arithmetic_parity','all_four_gate_branches','full_nine_run_report_serialization_in_temporary_CPU_fixture_only'],
  parity=parity,development_in_science_tables=False))
 print('CPU_PREFLIGHT_PASS')

if __name__=='__main__':main()

