from .common import *
import numpy as np
from module30.gpu import new_record,annotate_records
from module21.trajectory import arrivals,choose
from joint_deploy.measurement.load_generator import SamplePool

def capacity(runtime,config,slot,plan,mapping,monitor):
 duration=40.;start=time.perf_counter_ns();end=start+int(duration*1e9)
 # Background Poisson arrivals and one useful ASR request outstanding.
 rates={t:q for t,q in slot['lambda_rps'].items() if t!='asr'}
 pool=SamplePool(plan['sample_pool']['path'],list(slot['lambda_rps']),slot['seed'])
 ev=arrivals(rates,duration,slot['seed'],'capacity_background');rs=[];cursor=0;active=None
 monitor.phase='capacity_probe';monitor.set_active(dict(label='capacity_probe',planned_start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=duration,request_records=rs))
 while time.perf_counter_ns()<end:
  now=time.perf_counter_ns();monitor.check()
  with runtime.lock:
   if runtime.errors:raise RuntimeError(runtime.errors)
   while cursor<len(ev) and start+round(ev[cursor]['offset_seconds']*1e9)<=now:
    e=ev[cursor];item=pool.next(e['task']);r=new_record('capacity_probe',len(rs),e['task'],choose(e,slot['routing'],slot['deployment']),item,start+round(e['offset_seconds']*1e9));rs.append(r);runtime.enqueue(r,item['sample']);cursor+=1
   if active is None or active['status']=='completed':
    item=pool.next('asr');active=new_record('capacity_probe',len(rs),'asr',S+'#0',item,now);rs.append(active);runtime.enqueue(active,item['sample'])
   elif active['status']=='error':raise RuntimeError('CAPACITY_REQUEST_ERROR')
  # Host scheduler polling, not an inference delay; no fixed pauses in worker.
  runtime.stopping.wait(.001)
 runtime.drain(rs,60)
 with runtime.lock:annotate_records(rs)
 monitor.flush_active();monitor.set_active(None)
 aa=[r for r in rs if r['task']=='asr' and r['status']=='completed'];within=[r for r in aa if r['completion_time_ns']<=end]
 return dict(duration_seconds=duration,start_time_ns=start,end_time_ns=end,request_records=rs,completed_in_window=len(within),actual_ASR_capacity_qps=len(within)/duration,service_mean_ms=float(np.mean([r['worker_latency_ms'] for r in aa])),service_p95_ms=float(np.percentile([r['worker_latency_ms'] for r in aa],95)),all_completed=all(r['status']=='completed' for r in rs),scope='Observed 40s one-outstanding S capacity with real concurrent backgrounds; finite-window measurement, not long-window certification')
