"""Synthetic arithmetic fixtures only; never registered GPU measurements."""
import copy,time
from unittest.mock import patch
from types import SimpleNamespace
from .common import *
from .report import observation
from module21.trajectory import choose
from joint_deploy.measurement.run_measurement import new_record
from joint_deploy.measurement.metrics import annotate_records
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def fixture(congested=False):
 f=read(OUT/'cpu_screening.json');slot=copy.deepcopy(next(s for s in f['slots'] if s['run_id']=='C1_s_only'));slot['run_id']='CPU_SYNTHETIC_CONGESTED' if congested else 'CPU_SYNTHETIC_STEADY'
 mapping={m+'#0':m for m in slot['deployment']};start=100000000000000;unix=200000000000000;end=start+600000000000;records=[];finish={}
 events=read(slot['trajectories']['measurement']['path'])['events']
 for i,e in enumerate(events):
  at=start+round(e['offset_seconds']*1e9);worker=choose(e,slot['routing'],slot['deployment']);r=new_record('measurement',i,e['task'],worker,e['item'],at)
  svc=2000. if congested and e['task']=='asr' else 2.
  begin=max(at+10000,finish.get(worker,0));finish[worker]=begin+round(svc*1e6);wait=(begin-at)/1e6
  r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform'],enqueued_time_ns=at,actual_submit_time_ns=at,actual_submit_unix_ns=unix+at-start,status='completed',worker_latency_ms=svc,worker_queue_wait_ms=wait,completion_time_ns=finish[worker],worker_response=dict(task_type=e['task'],submit_ns=unix+at-start,start_ns=unix+begin-start,end_ns=unix+finish[worker]-start))
  records.append(r)
 annotate_records(records);drain=max(end,max(finish.values()))
 window=dict(start_time_ns=start,end_time_ns=end,drain_end_time_ns=drain,planned_duration_seconds=600,duration_seconds=600,drain_seconds=(drain-end)/1e9)
 identities={w:{'pid':0} for w in mapping};temp=dict(request_records=records,measurement_window=window,status='completed',errors=[],drain_timeout_seconds=60,worker_identities=identities)
 ev=reconstruct(temp,{w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for w,m in mapping.items()},DEFAULT_THRESHOLDS)
 path=OUT/'request_journals'/(slot['run_id']+'.jsonl')
 with path.open('w') as s:
  s.write(json.dumps(dict(kind='PHASE_WINDOW',label='measurement'))+'\n')
  for r in records:s.write(json.dumps(dict(kind='REQUEST_TERMINAL',phase='measurement',record=r))+'\n')
 sample=dict(time_ns=start+300000000000,phase='measurement',owned_used_mib=1000,temperature_gpu_c=35,sm_clock_mhz=2000,memory_clock_mhz=10000)
 raw=dict(slot=slot,status='COMPLETED',valid_session=True,errors=[],phases={'measurement':dict(request_records=records,window=window,stability=classify(ev,DEFAULT_THRESHOLDS))},replica_to_family=mapping,worker_identities=identities,
  postwarm_idle=[sample],postmeasurement_idle=[sample],telemetry={'samples':[sample]},baseline_GPU={'uuid':'CPU_SYNTHETIC_NO_GPU'},warmup=None)
 return f,raw

def main():
 original_read=read
 results={}
 for congested in [False,True]:
  f,r=fixture(congested)
  def patched(path):return f if Path(path)==OUT/'frozen_gpu_manifest.json' else original_read(path)
  with patch('module30.report.read',side_effect=patched):o=observation(r)
  if congested:assert o['hard_unstable'] and o['U'] is None and not o['eligible']
  else:
   assert o['eligible'] and o['U'] is not None and o['journal_audit']['passed']
   with patch('module30.report.read',side_effect=patched),patch('module30.report.journal_audit',return_value={'passed':False,'mismatch':['injected']}):bad=observation(r)
   assert bad['U'] is None and not bad['eligible']
   results['journal_mismatch_blocks_U']=True
  results['unstable_U_NULL' if congested else 'independent_utility_arithmetic_matches']=True
 save('cpu_analysis_acceptance.json',dict(passed=True,synthetic_fixtures_only=True,GPU_measurements=0,checks=results))
 print('CPU_ANALYSIS_ACCEPTANCE_PASSED',flush=True)
if __name__=='__main__':main()
