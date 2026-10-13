"""CPU-only archival audit. Never infer counterfactual utility from a replay."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,hashlib,heapq,collections,time
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module28_dynamic_split'
sys.path.insert(0,str(ROOT))
from module25.common import stats,json_native,utility_inputs
from module26.common import arithmetic

def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=json_native)+'\n');tmp.replace(p)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def csvsave(name,rows,fields=()):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False,default=json_native) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def metadata():
 return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in [ROOT/'results',*[p for p in ROOT.glob('module*') if p.is_dir()]] for p in root.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def main():
 assert not (OUT/'opportunity_audit.json').exists()
 save('historical_metadata_before.json',metadata())
 definition,refs=utility_inputs();sources=[];inventory=[];paths=[];systems=[];opps=[];alltarget=[]
 directories=[p for p in (ROOT/'results').glob('module*') if any(p.name.startswith(f'module{i}_') for i in range(19,28))]
 for root in sorted(directories):
  rawfiles=sorted(root.glob('raw_runs/*.json'));inventory.append(dict(directory=str(root),native_raw_count=len(rawfiles),status='NATIVE' if rawfiles else 'NO_OWN_RAW; inspect preserved summaries and linked experiments'))
  for p in sorted(root.glob('*gate.json'))+sorted(root.glob('*summary.md')):
   sources.append(ref(p))
  for p in rawfiles:
   sources.append(ref(p));raw=json.loads(p.read_text());ph=raw.get('phases',{}).get('measurement')
   if not ph:
    systems.append(dict(source=str(p),status=raw.get('status'),measurement=False,U=None,reason='UNSUPPORTED: missing complete measurement'));continue
   rs=ph['request_records'];done=[r for r in rs if r['status']=='completed'];mapping=raw['replica_to_family'];slot=raw['slot'];targets=['lid','asr'] if 'module26_' in root.name or 'module27_' in root.name else [slot.get('target_task',slot.get('target'))]
   # Frozen arithmetic used only diagnostically here; original eligibility retained separately.
   d=dict(safe_factor=1.1)
   d['safe_factor']=json.loads((ROOT/'results/module27_positive_split_discovery/selected_pool_catalog.json').read_text())['safe_factor']
   a=arithmetic(raw,d,refs,definition['weights'])
   systems.append(dict(source=str(p),status=raw.get('status'),measurement=True,valid_session=raw.get('valid_session'),system_stability=ph['stability']['stability_state_v2'],**{k:v for k,v in a.items() if k not in ['U','eligible','per_task']},per_task=a.get('per_task'),utility_scope='RAW_ARITHMETIC_DIAGNOSTIC; use original hard constraints for ranking'))
   byworker=collections.defaultdict(list)
   for r in done:byworker[r['model']].append(r)
   for w,wrs in byworker.items():
    for t in sorted(set(r['task'] for r in wrs)):
     rows=[r for r in wrs if r['task']==t]
     paths.append(dict(source=str(p),run_id=slot['run_id'],worker=w,family=mapping[w],task=t,n=len(rows),service=stats([r['worker_latency_ms'] for r in rows]),queue=stats([r['worker_queue_wait_ms'] for r in rows]),e2e=stats([r['e2e_latency_ms'] for r in rows]),quality_correct=sum(r['worker_response'].get('correct') is True for r in rows),quality_correct_available=sum('correct' in r['worker_response'] for r in rows)))
   if 'module26_' not in root.name and 'module27_' not in root.name:continue
   G=next(w for w,m in mapping.items() if m=='qwen');S=next((w for w,m in mapping.items() if 'whisper' in m),None)
   if not S:continue
   # Retrospective arrival snapshots of already-enqueued, not-yet-observed-complete work.
   # Native handler intervals determine actual historical busy states only; these
   # are never provided to a live policy or used to resimulate an alternate route.
   native={}
   for w,wrs in byworker.items():
    iv=[]
    for r in wrs:
     v=r['worker_response'];delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns'];iv.append((v['start_ns']+delta,v['end_ns']+delta))
    native[w]=sorted(iv)
   active=collections.defaultdict(dict);enq=sorted([r for r in rs if r.get('enqueued_time_ns')],key=lambda r:r['enqueued_time_ns']);ei=0;completions=[];indices=collections.defaultdict(int);counts=collections.Counter();rowsout=[]
   for r in sorted(rs,key=lambda r:r['scheduled_time_ns']):
    at=r['scheduled_time_ns']
    while ei<len(enq) and enq[ei]['enqueued_time_ns']<at:
     q=enq[ei];active[q['model']][q['request_id']]=q
     if q.get('completion_time_ns'):heapq.heappush(completions,(q['completion_time_ns'],q['request_id'],q['model']))
     ei+=1
    while completions and completions[0][0]<=at:
     _,rid,w=heapq.heappop(completions);active[w].pop(rid,None)
    if r['task'] not in targets:continue
    def busy(w):
     iv=native[w];ix=indices[w]
     while ix<len(iv) and iv[ix][1]<=at:ix+=1
     indices[w]=ix
     return ix<len(iv) and iv[ix][0]<=at<iv[ix][1]
    gs=busy(G);ss=busy(S);npendingS=len(active[S]);npendingG=len(active[G]);counts['n']+=1
    counts['S_backlog_G_empty']+=npendingS>=8 and npendingG==0
    counts['S_backlog_G_native_idle']+=npendingS>=8 and not gs
    row=dict(source=str(p),request_id=r['request_id'],task=r['task'],route=r['model'],scheduled_time_ns=at,enqueued_time_ns=r.get('enqueued_time_ns'),submitted_time_ns=r.get('actual_submit_time_ns'),native_start_unix_ns=r.get('worker_response',{}).get('start_ns'),native_end_unix_ns=r.get('worker_response',{}).get('end_ns'),completion_time_ns=r.get('completion_time_ns'),S_observable_pending=npendingS,G_observable_pending=npendingG,S_native_busy=ss,G_native_busy=gs,observed_wait_ms=r.get('worker_queue_wait_ms'),observed_service_ms=r.get('worker_latency_ms'),counterfactual_utility='UNSUPPORTED')
    rowsout.append(row)
   alltarget.extend(rowsout);opps.append(dict(source=str(p),target_tasks=targets,counts=dict(counts),pending_threshold=8,meaning='Descriptive backlog/empty windows only; no causal benefit or unexecuted service claim'))
 sources.extend(ref(ROOT/'results/module20_utility_optimizer'/name) for name in ['utility_definition.json','utility_reference_manifest.json'])
 save('historical_sources.json',sources);save('historical_inventory.json',inventory);save('historical_system_audit.json',systems);save('opportunity_audit.json',dict(stage='A_CPU_ONLY',windows=opps,unsupported=['Native receive ACK unavailable; host send is not worker acceptance','Native busy reconstruction is retrospective; future timestamps never policy input','Unexecuted routes have no per-request counterfactual service or utility','Mean GPU utilization is not model idle evidence','Physical GPU/CPU/thermal mechanisms cannot be uniquely separated from paired service inflation']))
 csvsave('historical_worker_paths.csv',paths);csvsave('historical_target_request_audit.csv',alltarget)
 print(json.dumps(opps,indent=2))
if __name__=='__main__':main()
