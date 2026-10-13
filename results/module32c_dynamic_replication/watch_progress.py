"""Read-only, cheap host progress view; no routing inputs or GPU work."""
import json
from pathlib import Path
o=Path(__file__).resolve().parent
s=json.loads((o/'campaign_status.json').read_text())
p=json.loads((o/'live_progress.json').read_text()) if (o/'live_progress.json').exists() else {}
v={k:p.get(k) for k in ['run_id','phase','elapsed_seconds','planned_seconds','completed','submitted','planned','temperature_gpu_c','used_mib']}
if isinstance(v['elapsed_seconds'],float):v['elapsed_seconds']=round(v['elapsed_seconds'],1)
v.update(campaign=s['status'],active=s.get('run_id'),attempts=len(s['attempted']))
print(json.dumps(v))
if s['status'] in ['COMPLETED','STOPPED_INCONCLUSIVE']:
 print((o/'campaign.log').read_text()[-4000:])
elif (o/'module32c_gate.json').exists():
 g=json.loads((o/'module32c_gate.json').read_text());print(json.dumps({'complete_pairs':g['statistics']['n_independent_pairs'] if g.get('statistics') else len(list((o/'formal_acceptance').glob('*.json')))//2,'interim_gate':g['gate']}))
