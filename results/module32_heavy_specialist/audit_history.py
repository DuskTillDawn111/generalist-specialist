import sys
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
import numpy as np
paths=list((ROOT/'results/module31a_qwen_whisper_legal_control/raw_runs').glob('A*.json'))+list((ROOT/'results/module26_positive_probe/raw_runs').glob('high_*.json'))
rows=[]
for p in sorted(paths):
 d=read(p);ph=d.get('phases',{}).get('measurement')
 if not ph:continue
 rr=ph['request_records'];dur=ph['window']['duration_seconds'];row=dict(source=ref(p),duration=dur,valid_session=d.get('valid_session'),slot=d['slot'],paths=[])
 for task,worker in sorted(set((r['task'],r['model']) for r in rr)):
  sub=[r for r in rr if r['task']==task and r['model']==worker and r['status']=='completed']
  if not sub:continue
  row['paths'].append(dict(task=task,worker=worker,n=len(sub),service_mean_ms=float(np.mean([r['worker_latency_ms'] for r in sub])),service_p95_ms=float(np.percentile([r['worker_latency_ms'] for r in sub],95)),wait_mean_ms=float(np.mean([r['worker_queue_wait_ms'] for r in sub])),wait_p95_ms=float(np.percentile([r['worker_queue_wait_ms'] for r in sub],95)),demand_ratio=sum(r['worker_latency_ms'] for r in sub)/1000/dur))
 rows.append(row)
save('historical_native_profile_review.json',rows)
mem=list(csv.DictReader((ROOT/'results/module14f_memory_ocr_x3/memory_semantics_corrections.csv').open()))
save('module14F_memory_provenance_review.json',dict(source=ref(ROOT/'results/module14f_memory_ocr_x3/memory_semantics_corrections.csv'),selected=[r for r in mem if r['count']=='1' and r['family'] in ['qwen','phi4mm','roberta_sst2','convnext_tiny',OCR]]))
print('HISTORICAL_RAW_REVIEW',len(rows))
