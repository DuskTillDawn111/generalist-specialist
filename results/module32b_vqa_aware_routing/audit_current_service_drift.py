import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
old=read(OLD/'raw_runs/A2.json');new=read(OUT/'raw_runs/A1.json');a=old['phases']['measurement']['request_records'];b=new['phases']['measurement']['request_records'];aa={r['request_id']:r for r in a};rows=[]
for r in b:
 x=aa[r['request_id']];assert all(r[k]==x[k] for k in ['task','sample_id','pool_id','routing_uniform','model','frozen_offset_seconds'])
 rows.append(dict(task=r['task'],worker=r['model'],request_id=r['request_id'],old_service_ms=x['worker_latency_ms'],new_service_ms=r['worker_latency_ms'],old_wait_ms=x['worker_queue_wait_ms'],new_wait_ms=r['worker_queue_wait_ms'],prediction_unchanged=r['worker_response'].get('prediction')==x['worker_response'].get('prediction')))
groups=[]
for m,t in sorted({(r['worker'],r['task']) for r in rows}):
 rr=[r for r in rows if r['worker']==m and r['task']==t];groups.append(dict(worker=m,task=t,n=len(rr),old_service_ms=stats(r['old_service_ms'] for r in rr),new_service_ms=stats(r['new_service_ms'] for r in rr),old_wait_ms=stats(r['old_wait_ms'] for r in rr),new_wait_ms=stats(r['new_wait_ms'] for r in rr),prediction_unchanged=sum(r['prediction_unchanged'] for r in rr)))
csvsave('cross_run_same_input_service_drift.csv',rows);save('cross_run_same_input_service_drift.json',dict(exact_matched_inputs_routes=True,old=ref(OLD/'raw_runs/A2.json'),new=ref(OUT/'raw_runs/A1.json'),groups=groups,explanation='Observed service-demand drift; identical input and route matching plus frozen native sources. No unique cause identified; historical U is never the strict comparator for Module32B. Parameters remain frozen.'))
print('DRIFT',[(r['worker'],r['task'],r['old_service_ms']['mean'],r['new_service_ms']['mean'],r['prediction_unchanged']) for r in groups])
