import sys
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
import numpy as np
rows=[]
for rid in ['B_S','B_PRESSURE_S']:
 r=read(OUT/'raw_runs'/(rid+'.json'));ph=r['phases']['measurement'];w=ph['window'];rs=ph['request_records'];ss=r['telemetry']['samples']
 for left in range(0,int(w['duration_seconds']),30):
  rr=[x for x in rs if x['task']=='asr' and left<=x['frozen_offset_seconds']<left+30];done=[x for x in rr if x['status']=='completed'];sn=[x for x in ss if w['start_time_ns']+left*1e9<=x['time_ns']<w['start_time_ns']+(left+30)*1e9]
  rows.append(dict(run_id=rid,arrival_bin_seconds=left,ASR_qps=r['slot']['lambda_rps']['asr'],offered=len(rr),completed=len(done),incomplete=len(rr)-len(done),service_mean_ms=float(np.mean([x['worker_latency_ms'] for x in done])) if done else None,service_p95_ms=float(np.percentile([x['worker_latency_ms'] for x in done],95)) if done else None,queue_wait_mean_ms=float(np.mean([x['worker_queue_wait_ms'] for x in done])) if done else None,**{key+'_mean':float(np.mean([x[key] for x in sn])) if sn else None for key in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','gpu_util_percent','worker_CPU_core_percent']},censored=(len(rr)!=len(done)),U=None))
csvsave('S_service_expansion_by_arrival_block.csv',rows)
save('S_service_expansion_audit.json',dict(rows=rows,scope='Completed native service grouped by scheduled arrival blocks; failed pressure run remains censored with NULL U. Resource averages use corresponding physical window blocks; associations are not an isolated causal experiment.'))
print('S_EXPANSION_AUDIT',len(rows))
