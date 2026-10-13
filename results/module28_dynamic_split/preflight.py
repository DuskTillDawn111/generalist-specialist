"""CPU checks of safety transitions, observation boundary, and source syntax."""
from audit import *
from dynamic_router import DynamicRouter
from gpu import VisibleRuntime
from types import SimpleNamespace

def main():
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
 print('CPU_PREFLIGHT_PASS')
if __name__=='__main__':main()
