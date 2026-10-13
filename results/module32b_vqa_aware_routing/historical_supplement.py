import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
probes=[];states=[];inventory=[]
for p in sorted(OLD.rglob('*.json')):
 if p.parent.name!='raw_runs':continue
 r=read(p);pp=r.get('path_probe_records',[]);inventory.append(dict(source=str(p),run_id=r['slot']['run_id'],path_probe_requests=len(pp)))
 for x in pp:
  w=x.get('worker_response',{});probes.append(dict(source=str(p),run_id=r['slot']['run_id'],request_id=x['request_id'],task=x['task'],worker=x['model'],status=x['status'],service_ms=x.get('worker_latency_ms'),queue_ms=x.get('worker_queue_wait_ms'),E2E_ms=x.get('e2e_latency_ms'),worker_start_unix_ns=w.get('start_ns'),worker_end_unix_ns=w.get('end_ns')))
for a in ['A1','A2']:
 rr=read(OLD/'raw_runs'/f'{a}.json')['phases']['measurement']['request_records']
 for x in rr:
  if x['task']!='asr' or x['model']!=G:continue
  for basis,k in [('scheduled_arrival','scheduled_time_ns'),('actual_host_enqueue','enqueued_time_ns'),('actual_host_send','actual_submit_time_ns')]:
   at=x[k];pp=[y for y in rr if y is not x and y['model']==G and y['enqueued_time_ns']<at<y['completion_time_ns']];states.append(dict(arm=a,request_id=x['request_id'],basis=basis,host_perf_time_ns=at,Phi_observable_pending=len(pp),Phi_VQA_pending=sum(y['task']=='visual_qa' for y in pp),Phi_ASR_pending=sum(y['task']=='asr' for y in pp),host_schedule_delay_ms=x['schedule_delay_ms'],scope='Past enqueued requests without received RESULT; fixed historical routing was preassigned and did not consult this state'))
csvsave('historical_path_probe_timings.csv',probes);save('historical_path_probe_inventory.json',inventory);csvsave('Phi_ASR_state_clock_basis.csv',states)
with (OUT/'vqa_interference_attribution.md').open('a') as f:
 f.write('\n## 辅助worker与到达状态口径\n\n跨worker重叠子群：Fixed5/Fixed10保留Whisper与Phi ASR handler重叠n=34/44，同输入服务比1.0093/1.0552；非重叠组为0.9014/0.9644。其他Specialist的重叠组服务变化有正有负：Fixed10 Text/OCR/LID约+69.5%/+27.8%/+18.4%，Image约−8.6%。样本小、输入时长与重叠互相影响，只有关联，不能判定这些变化均由GPU竞争引起。\n\n原fixed路由在cohort构建时预先按uniform指定，并未使用实时状态。Phi_ASR_arrival_state.csv记录的是host send时点；补充Phi_ASR_state_clock_basis.csv分别用scheduled arrival、实际enqueue与实际send重建“已入队而尚未收到RESULT”的因果可观测pending。动态原型在实际到达dispatch时同一runtime锁下快照并接纳，绝不读取native start/end。跨worker ASR/VQA重叠与未来VQA到达计数仅用于事后机制解释。全部path probe请求另存historical_path_probe_timings.csv，排除于正式U。\n')
print('SUPPLEMENT',len(probes),len(states))
