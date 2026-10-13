"""Read-only compact campaign snapshot from durable native records."""
import json,collections
from pathlib import Path
p=Path(__file__).resolve().parent
progress=json.loads((p/'live_progress.json').read_text()) if (p/'live_progress.json').exists() else {}
rid=progress.get('run_id');path=p/'request_journals'/f'{rid}.jsonl'
records={};checkpoint=None
if path.exists():
    for line in path.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('phase')!='measurement':continue
        if row['kind']=='REQUEST_TERMINAL':records[row['record']['request_id']]=row['record']
        elif row['kind']=='CHECKPOINT':checkpoint=row
    if checkpoint:
        for r in checkpoint['outstanding']:records[r['request_id']]=r
target=[r for r in records.values() if r['task']=='lid']
out=dict(progress=progress,checkpoint_outstanding=len(checkpoint['outstanding']) if checkpoint else None,
         observed_target_routes=dict(collections.Counter('G' if r['model']=='qwen#0' else 'S' for r in target)),
         dynamic_reasons=dict(collections.Counter(r.get('routing_decision',{}).get('reason') for r in target)),
         final_runs=[{k:r.get(k) for k in ['run_id','status','eligible','U']} for r in json.loads((p/'attempt_ledger.json').read_text())])
print(json.dumps(out,ensure_ascii=False))
