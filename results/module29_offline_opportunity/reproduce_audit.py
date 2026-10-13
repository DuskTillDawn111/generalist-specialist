"""Offline, frozen-input audit. No worker imports, inference, optimizer or network.
Writes only beside this script; never emits request bodies or model answers.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,csv,hashlib,collections,bisect,copy,math,ast,datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
OUT=Path(__file__).resolve().parent
FREEZE=json.loads((OUT/'input_freeze.json').read_text())
SRC=Path(FREEZE['read_only_source_directory'])
ROOT=SRC.parents[1]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=OUT/name;assert p.parent==OUT
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def verify_inputs():
 expected={x['path'] for x in FREEZE['inputs'] if x['module28_file']}
 current={str(p) for p in SRC.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.cache','cache','weights'])}
 changes=[x['path'] for x in FREEZE['inputs'] if not Path(x['path']).is_file() or Path(x['path']).stat().st_size!=x['bytes'] or sha(x['path'])!=x['sha256']]
 assert not changes and current==expected,(changes,current^expected)
 return dict(status='UNCHANGED',checked_input_files=len(FREEZE['inputs']),checked_module28_files=len(expected),sha256_changes=[],module28_file_set_changes=[])
def extract_module(path,allowed_imports):
 source=Path(path).read_text();tree=ast.parse(source)
 for node in tree.body:
  assert isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef,ast.Assign,ast.Expr))
  if isinstance(node,ast.Import):assert all(n.name.split('.')[0] in allowed_imports for n in node.names)
  if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in allowed_imports
 ns={'__name__':'module29_offline_source'};exec(compile(source,str(path),'exec'),ns)
 return ns,source

def duration_profile(events,start,end):
 grouped=collections.defaultdict(list)
 for t,kind,r in events:grouped[t].append((kind,r))
 counts=collections.Counter();active=set();last=start;segments=[];rows=[];seq=0
 for t,items in sorted(grouped.items()):
  if t>last:
   left=max(start,last);right=min(end,t)
   if right>left:segments.append(dict(start_s=(left-start)/1e9,end_s=(right-start)/1e9,duration_s=(right-left)/1e9,n=sum(counts.values()),lid=counts['lid'],asr=counts['asr']))
  # Same-time completions processed before enqueue, then arrival/send/query;
  # dwell durations use the aggregate resulting state, not transient tie peaks.
  for kind,r in sorted(items,key=lambda x:({'SUCCESS_RESPONSE':0,'HOST_ENQUEUE':1,'OFFERED_ARRIVAL':2,'HOST_SEND_START':3,'WINDOW_START':4,'WINDOW_END':5,'DRAIN_END':6}[x[0]],x[1].get('request_id',''))):
   before=sum(counts.values());pre=dict(counts)
   if r.get('model')==P['S_worker']:
    if kind=='HOST_ENQUEUE':assert r['request_id'] not in active;active.add(r['request_id']);counts[r['task']]+=1
    elif kind=='SUCCESS_RESPONSE':assert r['request_id'] in active;active.remove(r['request_id']);counts[r['task']]-=1
   seq+=1
   rows.append(dict(event_index=seq,t_s=(t-start)/1e9,time_ns=t,event=kind,request_id=r.get('request_id',''),task=r.get('task',''),worker=r.get('model',''),phase='FORMAL' if t<end else 'DRAIN' if t>end else 'WINDOW_BOUNDARY',S_unfinished_before=before,S_unfinished_after=sum(counts.values()),S_lid=counts['lid'],S_asr=counts['asr'],S_composition={k:v for k,v in counts.items() if v},data_quality='HOST_ENQUEUE_MINUS_SUCCESS_READER_RESPONSE; NOT_WAITING_QUEUE; NO_COLLECTOR_COMMIT_TIME'))
  last=t
 if last<end:segments.append(dict(start_s=(last-start)/1e9,end_s=(end-start)/1e9,duration_s=(end-last)/1e9,n=sum(counts.values()),lid=counts['lid'],asr=counts['asr']))
 assert abs(sum(x['duration_s'] for x in segments)-(end-start)/1e9)<1e-8
 return rows,segments

def merge_intervals(segments,predicate):
 out=[]
 for s in segments:
  if not predicate(s):continue
  if out and abs(out[-1]['end_s']-s['start_s'])<1e-8:out[-1]['end_s']=s['end_s'];out[-1]['duration_s']+=s['duration_s'];out[-1]['peak_n']=max(out[-1]['peak_n'],s['n'])
  else:out.append(dict(start_s=s['start_s'],end_s=s['end_s'],duration_s=s['duration_s'],peak_n=s['n']))
 return out

def time_quantiles(segments):
 durations=collections.defaultdict(float)
 for s in segments:durations[s['n']]+=s['duration_s']
 total=sum(durations.values());result={}
 for q in [.5,.9,.95,.99]:
  acc=0.
  for n,d in sorted(durations.items()):
   acc+=d
   if acc+1e-12>=q*total:result['P'+str(round(q*100))]=n;break
 return result

def make_plot(segments,first):
 # Dependency-free SVG of recorded queue levels; illustrative, not a fitted model.
 W,H=1200,360;L,R,T,B=70,30,25,60;maxn=max(s['n'] for s in segments)+2
 def xy(t,n):return L+t/600*(W-L-R),H-B-n/maxn*(H-T-B)
 line=[]
 for s in segments:
  x,y=xy(s['start_s'],s['n']);xx,_=xy(s['end_s'],s['n']);line.extend([f'{x:.3f},{y:.3f}',f'{xx:.3f},{y:.3f}'])
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><style>text{{font:14px sans-serif;fill:#253047}}</style>']
 for n in range(0,maxn,5):
  x,y=xy(0,n);parts.append(f'<path d="M {L},{y} H {W-R}" stroke="#e3e8ef"/><text x="{L-12}" y="{y+5}" text-anchor="end">{n}</text>')
 for t in range(0,601,60):
  x,y=xy(t,0);parts.append(f'<text x="{x}" y="{H-B+25}" text-anchor="middle">{t}</text>')
 parts.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="#2671b8" stroke-width="1.4"/>')
 if first:
  x,_=xy(first['observation_t_s'],0);parts.append(f'<path d="M {x},{T} V {H-B}" stroke="#c54a24" stroke-dasharray="5 4"/><text x="{x+7}" y="{T+15}">First single-step potential: {first["observation_t_s"]:.3f}s</text>')
 parts.extend([f'<text x="{W/2}" y="{H-8}" text-anchor="middle">Seconds since formal-window start; host unfinished count, not waiting queue</text>','</svg>'])
 (OUT/'s_only_queue_timeline.svg').write_text(''.join(parts))

verify_inputs()
RAW=read(SRC/'raw_runs/C_steady_s_only.json');F=read(SRC/'frozen_candidate_manifest.json');P=read(SRC/'router_parameters.json');assert P==F['parameters']
PH=RAW['phases']['measurement'];RS=PH['request_records'];WIN=PH['window'];START=WIN['start_time_ns'];END=WIN['end_time_ns'];STOP=WIN['drain_end_time_ns'];DURATION=(END-START)/1e9
RouterNS,router_source=extract_module(SRC/'dynamic_router.py',{'collections'});Router=RouterNS['DynamicRouter']
StabilityNS,_=extract_module(ROOT/'performance_model/local_calibration/stability_v2.py',{'collections','math','numpy'})
# Save actual source expressions, including every decision guard in original order.
router_ast=ast.parse(router_source);methods={n.name:n for c in router_ast.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
conditions=[dict(line=n.lineno,expression=ast.unparse(n.test)) for n in ast.walk(methods['decide']) if isinstance(n,ast.If)]
save('frozen_rule_extraction.json',dict(router_sha256=sha(SRC/'dynamic_router.py'),config_sha256=sha(SRC/'frozen_candidate_manifest.json'),parameters=P,estimate_source=ast.get_source_segment(router_source,methods['estimate']),decide_source=ast.get_source_segment(router_source,methods['decide']),completion_update_source=ast.get_source_segment(router_source,methods['observe_completion']),decide_conditions=conditions,note='Literal original code, no replacement by a 13-request rule. cost values are proxies, not E2E or delta-U predictions.'))
# Validate historical P10/P95 derivation, only sources already frozen before run.
service_samples=collections.defaultdict(list);means=[];dev=[]
for source in F['development_sources']:
 path=Path(source['path']);assert sha(path)==source['sha256'];r=read(path);rs=r['phases']['measurement']['request_records'];latest_unix=max(x['completion_unix_ns'] for x in rs);assert latest_unix<F['at_unix_ns']<WIN['start_unix_ns']
 for x in rs:service_samples[x['model'],x['task']].append(x['worker_latency_ms'])
 means.append(dict(fraction=r['slot']['G_fraction'],S=float(np.mean([x['worker_latency_ms'] for x in rs if x['model']==P['S_worker'] and x['task']=='lid']))));dev.append(dict(path=str(path),sha256=source['sha256'],latest_success_relative_to_module28_start_s=(latest_unix-WIN['start_unix_ns'])/1e9,available_before_freeze=True))
s_lower={t:float(np.quantile(service_samples[P['S_worker'],t],.1)) for t in ['lid','asr']};g_upper={t:float(np.quantile(service_samples[P['G_worker'],t],.95)) for t in ['lid','visual_qa']}
inflation=max(v['S'] for v in means if v['fraction']>0)-min(v['S'] for v in means if v['fraction']==0);background=F['rates']['visual_qa']*g_upper['lid']**2/2000+inflation*g_upper['lid']/s_lower['lid']
assert all(abs(s_lower[t]-P['S_lower_by_task_ms'][t])<1e-9 for t in s_lower);assert all(abs(g_upper[t]-P['G_upper_by_task_ms'][t])<1e-9 for t in g_upper);assert abs(inflation-P['S_inflation_per_pending_ms'])<1e-9 and abs(background-P['background_cost_ms'])<1e-9
save('development_parameter_verification.json',dict(status='MATCH_FROZEN',sources=dev,S_P10_ms=s_lower,G_P95_ms=g_upper,S_inflation_ms=inflation,background_ms=background,Module28_future_requests_used_to_fit_parameters=False,uncertainty='Empirical quantiles and penalties are not certified bounds or causal interference estimates.'))
# Successful response accounting, explicitly [0,600) and drain afterward.
ids=[x['request_id'] for x in RS];duplicates=[k for k,v in collections.Counter(ids).items() if v>1];missing={k:[x['request_id'] for x in RS if x.get(k) is None] for k in ['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','completion_time_ns']};bad_order=[x['request_id'] for x in RS if not x['scheduled_time_ns']<=x['enqueued_time_ns']<=x['actual_submit_time_ns']<=x['completion_time_ns']<=STOP]
success=[x for x in RS if x['status']=='completed'];sent=sum(START<=x['actual_submit_time_ns']<END for x in RS);within=sum(START<=x['completion_time_ns']<END for x in success);drain=sum(END<=x['completion_time_ns']<=STOP for x in success);unresolved=sum(x['actual_submit_time_ns']<END and (x['status']!='completed' or x['completion_time_ns']>=END) for x in RS)
all_run_rows=list(csv.DictReader((SRC/'all_gpu_runs.csv').open()));assert len(all_run_rows)==1;legal_U=None if all_run_rows[0]['U']=='' else float(all_run_rows[0]['U'])
expected=dict(sent_in_formal_window=7607,success_in_formal_window=7575,unfinished_at_window_end=32,success_during_drain=32,total_success_after_drain=7607)
actual=dict(sent_in_formal_window=sent,success_in_formal_window=within,unfinished_at_window_end=unresolved,success_during_drain=drain,total_success_after_drain=len(success));match=expected==actual
reconciliation=dict(status='VERIFIED' if match and not duplicates and not bad_order and not any(missing.values()) else 'EVIDENCE_INCOMPLETE',time_basis='Host perf_counter_ns relative to formal start; formal [0,600), drain [600,drain_end]',formal_seconds=DURATION,expected=expected,observed=actual,drain_seconds=(STOP-END)/1e9,offered_qps=len(RS)/DURATION,completed_in_window_qps=within/DURATION,cohort_completed_divided_by_formal_seconds=len(success)/DURATION,cohort_completed_divided_by_formal_plus_drain_seconds=len(success)/((STOP-START)/1e9),task_counts=dict(collections.Counter(x['task'] for x in RS)),duplicate_request_ids=duplicates,missing_required_fields={k:len(v) for k,v in missing.items()},timestamp_order_violations=len(bad_order),events_exactly_at_600={k:sum(x[k]==END for x in RS) for k in ['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','completion_time_ns']},warmup_excluded=True,original_gate=read(SRC/'module28_gate.json')['gate'],original_classifier=PH['stability']['stability_state_v2'],legal_U=legal_U,note='Drain completion does not count as completion inside600s and does not override original unstable/NULL U.')
assert legal_U is None and reconciliation['original_gate']=='DYNAMIC_EVIDENCE_GAP' and reconciliation['original_classifier']=='unstable';save('window_reconciliation.json',reconciliation)
# Journal is a redundant check, not a substitute for collector state-update logs.
rawmap={x['request_id']:x for x in RS};journal_seen=collections.Counter();journal_mismatches=[]
with (SRC/'request_journals/C_steady_s_only.jsonl').open() as stream:
 for line in stream:
  entry=json.loads(line)
  if entry['kind']=='REQUEST_TERMINAL' and entry['phase']=='measurement':
   x=entry['record'];journal_seen[x['request_id']]+=1;reference=rawmap.get(x['request_id'])
   if reference is None or any(x.get(k)!=reference.get(k) for k in ['task','model','status','scheduled_time_ns','actual_submit_time_ns','completion_time_ns']):journal_mismatches.append(x['request_id'])
save('journal_reconciliation.json',dict(terminal_measurement_records=sum(journal_seen.values()),unique_terminal_ids=len(journal_seen),duplicate_terminal_ids=[k for k,v in journal_seen.items() if v>1],raw_terminal_not_in_journal=len(set(rawmap)-set(journal_seen)),mismatches=journal_mismatches,note='Periodic checkpoint journal can omit final completions; authoritative full raw is retained. Missing journal terminal rows are not missing raw responses.'))
EVENTS=[(START,'WINDOW_START',{}),(END,'WINDOW_END',{}),(STOP,'DRAIN_END',{})]
for x in RS:
 for k,kind in [('scheduled_time_ns','OFFERED_ARRIVAL'),('enqueued_time_ns','HOST_ENQUEUE'),('actual_submit_time_ns','HOST_SEND_START')]:EVENTS.append((x[k],kind,x))
 if x['status']=='completed':EVENTS.append((x['completion_time_ns'],'SUCCESS_RESPONSE',x))
TIMELINE,SEGMENTS=duration_profile(EVENTS,START,END);table('s_only_queue_timeline.csv',TIMELINE)
# Exact unfinished sets before each planned arrival, with enqueue and response clocks.
enqueues=sorted(RS,key=lambda x:x['enqueued_time_ns']);responses=sorted(success,key=lambda x:x['completion_time_ns']);active={};ei=ci=0;arrival_rows=[];arrival_sets={}
for x in sorted(RS,key=lambda x:x['scheduled_time_ns']):
 t=x['scheduled_time_ns']
 while ei<len(enqueues) and enqueues[ei]['enqueued_time_ns']<t:y=enqueues[ei];active[y['request_id']]=y;ei+=1
 while ci<len(responses) and responses[ci]['completion_time_ns']<=t:active.pop(responses[ci]['request_id'],None);ci+=1
 sr=sorted([y for y in active.values() if y['model']==P['S_worker']],key=lambda y:y['enqueued_time_ns']);gr=[y for y in active.values() if y['model']==P['G_worker']];arrival_sets[x['request_id']]=(sr,gr)
 arrival_rows.append(dict(t_arrival_s=(t-START)/1e9,request_id=x['request_id'],task=x['task'],S_unfinished=len(sr),S_lid=sum(y['task']=='lid' for y in sr),S_asr=sum(y['task']=='asr' for y in sr),S_unfinished_ids=[y['request_id'] for y in sr],G_reader_unfinished=len(gr),data_quality='BEFORE_CURRENT_HOST_ENQUEUE; READER_RESPONSE_TIME; COLLECTOR_COMMIT_TIME_UNKNOWN'))
table('arrival_unfinished_sets.csv',arrival_rows)
bins=[('0',0,0),('1-4',1,4),('5-9',5,9),('10-19',10,19),('20-29',20,29),('>=30',30,math.inf)];distribution=[]
for label,lo,hi in bins:
 seconds=sum(x['duration_s'] for x in SEGMENTS if lo<=x['n']<=hi);distribution.append(dict(range=label,seconds=seconds,fraction=seconds/DURATION))
# Original classifier is recomputed exactly, without changing thresholds/gate/U.
thresholds=PH['stability']['thresholds'];models=F['plan']['models'];mapping=RAW['replica_to_family'];catalog={w:SimpleNamespace(service_time_ms=models[m]['service_time_ms']) for w,m in mapping.items()}
classifier_input=dict(request_records=RS,measurement_window=WIN,status='completed',errors=RAW['errors'],drain_timeout_seconds=60.,worker_identities=RAW['worker_identities']);evidence=StabilityNS['reconstruct'](classifier_input,catalog,thresholds);state=StabilityNS['classify'](evidence,thresholds)
assert state==PH['stability'];assert abs(evidence['tail_backlog_slope']-PH['stability_evidence']['tail_backlog_slope'])<1e-12
swrs=[x for x in RS if x['model']==P['S_worker']];s_input={**classifier_input,'request_records':swrs,'worker_identities':{P['S_worker']:RAW['worker_identities'][P['S_worker']]}};se=StabilityNS['reconstruct'](s_input,{P['S_worker']:catalog[P['S_worker']]},thresholds)
high=merge_intervals(SEGMENTS,lambda s:s['n']>=10);very_high=merge_intervals(SEGMENTS,lambda s:s['n']>=20);table('high_queue_intervals.csv',[dict(threshold=threshold,**r) for threshold,items in [(10,high),(20,very_high)] for r in items])
queue=dict(definition='S host-enqueued requests without successful reader response, not waiting queue',formal_seconds=DURATION,peak_S_unfinished=max(x['n'] for x in SEGMENTS),S_unfinished_at_600=sum(x['model']==P['S_worker'] and x['enqueued_time_ns']<END and x['completion_time_ns']>=END for x in RS),time_weighted_ranges=distribution,time_weighted_quantiles=time_quantiles(SEGMENTS),arrival_event_weighted_quantiles={f'P{q}':float(np.percentile([x['S_unfinished'] for x in arrival_rows],q)) for q in [50,90,95,99]},event_sampling='7607 planned arrival events across6tasks, before current enqueue; distinct from dwell-time-weighted distribution',high_queue_display_thresholds_only=dict(ge10_seconds=sum(r['duration_s'] for r in high),ge10_intervals=len(high),ge10_longest_s=max(r['duration_s'] for r in high),ge20_seconds=sum(r['duration_s'] for r in very_high),ge20_intervals=len(very_high),ge20_longest_s=max(r['duration_s'] for r in very_high)),original_system_tail=dict(slope_rps=evidence['tail_backlog_slope'],growth_requests=evidence['tail_backlog_growth'],increasing_fraction=evidence['tail_increasing_fraction'],bins=evidence['backlog_timeline'][-5:],method='OLS on exact30s mean submitted unfinished counts over final150s; unchanged original classifier'),S_only_submitted_tail=dict(slope_rps=se['tail_backlog_slope'],growth_requests=se['tail_backlog_growth'],increasing_fraction=se['tail_increasing_fraction']),original_state_reproduced=state['stability_state_v2'],display_bins_are_not_routing_thresholds=True)
save('queue_summary.json',queue)
# Normalize all clocks; native handler START/END is retrospective, never an online input.
normalized=[];offsets=[];native={};native_inconsistency=[]
for x in RS:
 payload=x['worker_response'];offset=x['actual_submit_unix_ns']-x['actual_submit_time_ns'];offsets.append(offset)
 assert payload['submit_ns']==x['actual_submit_unix_ns']
 begin=x['actual_submit_time_ns']+payload['start_ns']-x['actual_submit_unix_ns'];finish=x['actual_submit_time_ns']+payload['end_ns']-x['actual_submit_unix_ns'];native[x['request_id']]=(begin,finish)
 if begin<x['actual_submit_time_ns'] or finish>x['completion_time_ns'] or abs((finish-begin)/1e6-x['worker_latency_ms'])>1e-6:native_inconsistency.append(x['request_id'])
 normalized.append(dict(request_id=x['request_id'],task=x['task'],worker=x['model'],scheduled_s=(x['scheduled_time_ns']-START)/1e9,enqueued_s=(x['enqueued_time_ns']-START)/1e9,send_start_s=(x['actual_submit_time_ns']-START)/1e9,send_write_complete_s=(x['submit_write_completed_time_ns']-START)/1e9,success_response_s=(x['completion_time_ns']-START)/1e9,handler_start_s_retrospective=(begin-START)/1e9,handler_end_s_retrospective=(finish-START)/1e9,handler_clock_mapping='native Unix ns mapped with per-request host perf/Unix send pair; not live START ACK'))
table('normalized_request_times.csv',normalized)
save('time_basis_checks.json',dict(monotonic_window_start_ns=START,unix_window_start_ns=WIN['start_unix_ns'],send_pair_offset_span_ns=max(offsets)-min(offsets),max_send_pair_vs_window_offset_difference_ns=max(abs(v-(WIN['start_unix_ns']-START)) for v in offsets),native_handler_order_consistency_violations=native_inconsistency,handler_times_used_as_policy_inputs=False,clock_uncertainty='Pair clocks are sequential reads, not synchronized worker START ACK. Native interval checks are retrospective only. Reader receipt is not collector state-commit time.'))
# Recorded snapshots resolve four delayed collector receipts. Reconstruction uses
# their counts plus per-worker FIFO and only success reports received by cutoff.
workers=collections.defaultdict(list)
for x in sorted(RS,key=lambda x:x['enqueued_time_ns']):workers[x['model']].append(x)
worker_enqueue={w:[x['enqueued_time_ns'] for x in xs] for w,xs in workers.items()}
fifo={w:all(a['completion_time_ns']<b['completion_time_ns'] for a,b in zip(xs,xs[1:])) for w,xs in workers.items()};assert all(fifo.values())
observer=Router(P);removed={P['S_worker']:0,P['G_worker']:0};prefix_mode=False;prefix_last_time=None;prefix_known=True;first=None;formula_mismatches=[];snapshot_lag_events=[];max_ewma_seen=0.;strict_future_observation_count=0;potential=[];snapshot_analysis=[];all_receipts=sorted(x['completion_time_ns'] for x in RS)
for x in sorted(RS,key=lambda x:x['scheduled_time_ns']):
 scheduled=x['scheduled_time_ns'];sr_at_arrival,gr_at_arrival=arrival_sets[x['request_id']]
 # Strict planned-arrival lower estimate: reader responses up to scheduled only.
 # Exact collector-visible pending/status and policy state are not assumed.
 arrival_est=Router(P).estimate({P['S_worker']:sr_at_arrival,P['G_worker']:gr_at_arrival})
 row=dict(request_id=x['request_id'],task=x['task'],arrival_t_s=(scheduled-START)/1e9,observation_t_s=None,observation_delay_ms=None,strict_arrival_cutoff_s=(scheduled-START)/1e9,S_unfinished_at_planned_arrival=len(sr_at_arrival),S_arrival_composition=dict(collections.Counter(q['task'] for q in sr_at_arrival)),G_reader_unfinished_at_planned_arrival=len(gr_at_arrival),strict_arrival_C_S_reader_proxy_ms=arrival_est['S_system_cost_proxy_ms'],strict_arrival_C_G_reader_proxy_ms=arrival_est['G_system_cost_proxy_ms'],strict_arrival_snapshot_status='UNKNOWN_COLLECTOR_STATUS_COMMIT_TIME',target_eligible_guard='PASS' if x['task']==P['target'] else 'FAIL',actual_original_routing='S_ONLY_OR_FROZEN_BACKGROUND',single_step_counterfactual_only=True,evidence_sources='raw measurement events; frozen router code/config; no request body/answer fields')
 if x['task']!=P['target']:
  row.update(cost_gap_pass='UNKNOWN',G_empty_observed_guard='UNKNOWN',G_inflight_observed_guard='UNKNOWN',admission_observed_guard='UNKNOWN',capacity_observed_guard='UNKNOWN',hysteresis_prefix_guard='UNKNOWN',prediction_fallback_prefix_guard='UNKNOWN',cooldown_prefix_guard='UNKNOWN',token_prefix_guard='UNKNOWN',dynamic_G_state_prefix_guard='UNKNOWN',final_trigger_state='NOT_TARGET_TASK',reason='Original harness invokes router only for target LID; no live snapshot logged for other5 tasks');potential.append(row);continue
 d=x['routing_decision'];t=d['decision_time_ns'];clock=(t-START)/1e9;pending={};good=True;past_only=True
 for w,k in [(P['S_worker'],'S_pending'),(P['G_worker'],'G_pending')]:
  xs=workers[w][:bisect.bisect_left(worker_enqueue[w],t)];n=len(xs)-d[k]
  if n<removed[w] or n<0:good=False;n=max(removed[w],0)
  pending[w]=xs[n:]
  for q in xs[removed[w]:n]:
   if q['completion_time_ns']>t or q['completion_time_ns']>d['latest_observed_completion_time_ns']:past_only=False;continue
   observer.observe_completion(q,t/1e9)
   if q['model']==P['S_worker'] and q['task']=='lid':max_ewma_seen=max(max_ewma_seen,observer.service[P['S_worker'],'lid'])
  removed[w]=n
  reader_count=sum(q['completion_time_ns']>t for q in xs)
  if reader_count!=d[k]:snapshot_lag_events.append(dict(request_id=x['request_id'],observation_t_s=clock,worker=w,reader_unfinished=reader_count,logged_pending=d[k],explanation='Collector status lags reader timestamp; logged snapshot plus verified FIFO reconstructs an older observed-completion prefix.'))
 est=observer.estimate(pending)
 match_costs=all(abs(est[k]-d[k])<1e-8 for k in ['S_wait_est_ms','S_exec_est_ms','G_wait_est_ms','G_exec_est_ms','S_interference_est_ms','background_externality_est_ms','S_system_cost_proxy_ms','G_system_cost_proxy_ms'])
 if not match_costs:formula_mismatches.append(x['request_id'])
 good=good and past_only and match_costs and len(pending[P['S_worker']])==d['S_pending'] and len(pending[P['G_worker']])==d['G_pending']
 gap=d['S_system_cost_proxy_ms']-d['G_system_cost_proxy_ms'];emptygap=d['S_system_cost_proxy_ms']-(d['G_exec_est_ms']+d['S_interference_est_ms']+d['background_externality_est_ms'])
 row.update(observation_t_s=clock,observation_delay_ms=(t-scheduled)/1e6,observation_cutoff_s=clock,S_pending_logged=d['S_pending'],S_snapshot_composition=dict(collections.Counter(q['task'] for q in pending[P['S_worker']])),S_pending_ids=[q['request_id'] for q in pending[P['S_worker']]],G_pending_logged=d['G_pending'],G_target_inflight_logged=d['G_target_inflight'],S_wait_est_ms=d['S_wait_est_ms'],S_exec_est_ms=d['S_exec_est_ms'],G_wait_est_ms=d['G_wait_est_ms'],G_exec_est_ms=d['G_exec_est_ms'],S_interference_est_ms=d['S_interference_est_ms'],background_externality_est_ms=d['background_externality_est_ms'],C_S_ms=d['S_system_cost_proxy_ms'],C_G_ms=d['G_system_cost_proxy_ms'],cost_gap_ms=gap,C_G_if_assumed_empty_ms=d['G_exec_est_ms']+d['S_interference_est_ms']+d['background_externality_est_ms'],cost_gap_if_assumed_G_empty_ms=emptygap,entry_margin_ms=P['entry_margin_ms'],exit_margin_ms=P['exit_margin_ms'],entry_gap_guard='PASS' if gap>P['entry_margin_ms'] else 'FAIL',exit_gap_guard='PASS' if gap>P['exit_margin_ms'] else 'FAIL',G_empty_observed_guard='PASS' if d['G_pending']==0 else 'FAIL',G_inflight_observed_guard='PASS' if d['G_target_inflight']<P['G_max_target_inflight'] else 'FAIL',admission_observed_guard='PASS' if d['admitted'] else 'FAIL',capacity_observed_guard='PASS' if d['capacity_ok'] else 'FAIL',EWMA_S_ms=observer.service.get((P['S_worker'],'lid')),EWMA_G_target_ms=observer.service.get((P['G_worker'],'lid')),G_EWMA_status='NO_TARGET_RESULT_IN_PHASE; FROZEN_P95_FALLBACK',history_receipt_cutoff_proven=past_only,formula_matches_original_snapshot=match_costs,observed_state_reconstruction_guard='PASS' if good else 'UNKNOWN',assumed_G_idle_cost_evidence='SCENARIO_ONLY; actual observed G pending reported separately',strict_arrival_uses_later_logged_snapshot=False)
 # All quantities in *_planned_arrival use scheduled cutoff, not logged later
 # snapshots. Later snapshots are explicitly separate actual-call evidence.
 snapshot_has_after_arrival_receipt=d['latest_observed_completion_time_ns']>scheduled
 strict_future_observation_count+=snapshot_has_after_arrival_receipt
 row['snapshot_contains_response_after_planned_arrival']=snapshot_has_after_arrival_receipt
 if gap>P['entry_margin_ms']:row['cost_gap_pass']='PASS_ROBUST_TO_HYSTERESIS_MODE'
 elif gap<=P['exit_margin_ms']:row['cost_gap_pass']='FAIL'
 else:row['cost_gap_pass']='UNKNOWN_HYSTERESIS_MODE'
 if first is None and prefix_known and good:
  # Counterfactual prefix is still identical in routing to S-only. Frozen
  # decide() is used on a copy and stopped at the very first G choice.
  trial=copy.deepcopy(observer);trial.mode=prefix_mode;trial.last_time=prefix_last_time
  route,decision=trial.decide(x['request_id'],t/1e9,pending,admitted=d['admitted'],capacity_ok=d['capacity_ok'])
  row.update(hysteresis_prefix_guard='PASS' if trial.mode and gap>P['exit_margin_ms'] else 'FAIL',prediction_fallback_prefix_guard='PASS' if t/1e9>=observer.fallback_until else 'FAIL',cooldown_prefix_guard='PASS' if t/1e9>=observer.next_g_time else 'FAIL',token_prefix_guard='PASS' if trial.tokens>=1. or route==P['G_worker'] else 'FAIL',dynamic_G_state_prefix_guard='PASS' if d['G_pending']==0 and d['G_target_inflight']<P['G_max_target_inflight'] else 'FAIL',hysteresis_prefix_mode=trial.mode,tokens_before_single_step=observer.tokens,fallback_until_relative_s=observer.fallback_until-START/1e9 if observer.fallback_until else None,cooldown_until_relative_s=observer.next_g_time-START/1e9 if observer.next_g_time else None,prefix_scope='NO_EARLIER_COUNTERFACTUAL_G_ROUTE; identical observed S-only prefix; fixed recorded timing; not actual dynamic execution')
  prefix_mode=trial.mode;prefix_last_time=trial.last_time
  if route==P['G_worker']:
   first=copy.deepcopy(row);first['final_trigger_state']='SUPPORTED_FIRST_SINGLE_STEP_AT_RECORDED_DECISION';row['final_trigger_state']=first['final_trigger_state'];row['reason']='All frozen code guards support this one decision under no-earlier-diversion prefix; do not replay later queues'
  else:row['final_trigger_state']='BLOCKED_BEFORE_FIRST_SINGLE_STEP';row['reason']=decision['reason']
 else:
  if first is None:prefix_known=False
  row.update(hysteresis_prefix_guard='UNKNOWN',prediction_fallback_prefix_guard='UNKNOWN',cooldown_prefix_guard='UNKNOWN',token_prefix_guard='UNKNOWN',dynamic_G_state_prefix_guard='UNKNOWN',final_trigger_state='UNKNOWN_AFTER_FIRST_COUNTERFACTUAL_DIVERSION' if first else 'UNKNOWN_PREFIX_STATE',reason='Earlier hypothetical G route would change queue, service, EWMA, interference, cooldown and tokens; original S-only cannot provide those states')
 potential.append(row)
 snapshot_analysis.append(dict(request_id=x['request_id'],t_s=clock,cost_gap=gap,cost_gap_assumed_empty=emptygap,G_pending=d['G_pending'],S_pending=d['S_pending']))
assert not formula_mismatches
assert max_ewma_seen<=P['S_service_alarm_ms']
save('state_reconstruction_checks.json',dict(formula_comparisons=7005,formula_mismatches=formula_mismatches,collector_vs_reader_lag_events=snapshot_lag_events,FIFO_by_worker=fifo,EWMA_S_max_observed_prefix_ms=max_ewma_seen,S_service_alarm_ms=P['S_service_alarm_ms'],reconstruction_uses_only_success_receipts_at_or_before_each_recorded_snapshot=True,snapshots_with_latest_observed_receipt_after_planned_arrival=strict_future_observation_count,strict_planned_arrival_uses_those_later_receipts=False,limits=['Collector commit timestamps absent, so exact collector state at scheduled arrival is not recoverable merely by subtracting reader responses','Recorded LID decision snapshots support actual-call-time audit; these times must not be relabeled exact planned-arrival observations','Counterfactual prefix replay stops at first G choice; no repeated-diversion count']))
table('potential_trigger_events.csv',potential)
# Guard counts: observed S-only evidence and pre-first hypothetical policy state
# are deliberately distinct. Post-first dynamic states are UNKNOWN.
LID=[x for x in potential if x['task']=='lid']
def tri_count(field,rows):
 count=collections.Counter('PASS' if str(x.get(field,'UNKNOWN')).startswith('PASS') else 'FAIL' if str(x.get(field,'UNKNOWN')).startswith('FAIL') else 'UNKNOWN' for x in rows)
 return dict(PASS=count['PASS'],FAIL=count['FAIL'],UNKNOWN=count['UNKNOWN'],denominator=len(rows))
guard_fields=['entry_gap_guard','exit_gap_guard','cost_gap_pass','G_empty_observed_guard','G_inflight_observed_guard','admission_observed_guard','capacity_observed_guard','observed_state_reconstruction_guard','hysteresis_prefix_guard','prediction_fallback_prefix_guard','cooldown_prefix_guard','token_prefix_guard','dynamic_G_state_prefix_guard']
guards={name:tri_count(name,LID) for name in guard_fields}
actual_block=collections.Counter(x['reason'] for x in LID if x['final_trigger_state']=='BLOCKED_BEFORE_FIRST_SINGLE_STEP')
pointwise_cost_counts=dict(observed_G_empty_and_entry_cost_pass=sum(x['G_empty_observed_guard']=='PASS' and x['entry_gap_guard']=='PASS' for x in LID),entry_cost_pass_if_G_assumed_empty=sum(x['cost_gap_if_assumed_G_empty_ms']>P['entry_margin_ms'] for x in LID),entry_cost_pass_but_observed_G_busy=sum(x['entry_gap_guard']=='PASS' and x['G_empty_observed_guard']=='FAIL' for x in LID),warning='Independent recorded-baseline cost checks, NOT counts of actual/counterfactual dynamic diversions')
breakdown=dict(status='EVIDENCE_INCOMPLETE',scope='Observed S-only snapshots plus one counterfactual prefix decision; not dynamic-route feasibility or utility evidence',all_arrivals=7607,target_eligibility=tri_count('target_eligible_guard',potential),target_guard_denominator=7005,guards=guards,per_task={t:dict(arrivals=sum(x['task']==t for x in potential),router_invoked_in_original=(t=='lid'),supported_first_single_step=sum(x['task']==t and x['final_trigger_state']=='SUPPORTED_FIRST_SINGLE_STEP_AT_RECORDED_DECISION' for x in potential)) for t in sorted(set(x['task'] for x in potential))},blocking_reasons_before_first=dict(actual_block),first_supported_actual_decision_t_s=first['observation_t_s'] if first else None,first_trigger_at_exact_planned_arrival=None,strict_arrival_precision_reason='No collector status-commit or complete router-state checkpoint at exact scheduled_time_ns. Actual recorded decision occurs afterward; do not silently transfer its evidence backward.',pointwise_diagnostics=pointwise_cost_counts,exact_multi_diversion_count=None,post_first_policy_state='UNKNOWN',missing_evidence=['Per-arrival collector state-commit/received-observed completion IDs, including non-target arrivals','Persisted dynamic mode/tokens/last_time/next_g_time/fallback_until/service/predictions states (dynamic decide never ran here)','Counterfactual G target completions and resulting S/G queue/service/interference states after first diversion','Legal stable matched fixed/dynamic GPU controls; original S-only is unstable and U=NULL'])
save('guard_breakdown.json',breakdown)
# Persistence is measured on the actual baseline, with G-empty explicitly assumed.
sw=workers[P['S_worker']];sw_enq=worker_enqueue[P['S_worker']];sw_fin=[x['completion_time_ns'] for x in sw]
for s in SEGMENTS:
 t=START+round(s['start_s']*1e9);ne=bisect.bisect_right(sw_enq,t);nc=bisect.bisect_right(sw_fin,t);pending={P['S_worker']:sw[nc:ne],P['G_worker']:[]};est=Router(P).estimate(pending);s['hyp_G_empty_entry_gap']=est['S_system_cost_proxy_ms']-est['G_system_cost_proxy_ms']
cost_intervals=merge_intervals(SEGMENTS,lambda s:s['hyp_G_empty_entry_gap']>P['entry_margin_ms']);table('hypothetical_G_empty_cost_intervals.csv',cost_intervals)
first_info=None
if first:
 t=START+round(first['observation_t_s']*1e9);native_s=sum(a<=t<b for rid,(a,b) in native.items() if rawmap[rid]['model']==P['S_worker']);native_g=sum(a<=t<b for rid,(a,b) in native.items() if rawmap[rid]['model']==P['G_worker'])
 containing=next((s for s in cost_intervals if s['start_s']<=first['observation_t_s']<s['end_s']),None);high_containing=next((s for s in high if s['start_s']<=first['observation_t_s']<s['end_s']),None)
 first_info=dict(request_id=first['request_id'],task='lid',planned_arrival_t_s=first['arrival_t_s'],recorded_decision_t_s=first['observation_t_s'],difference_ms=first['observation_delay_ms'],S_pending=first['S_pending_logged'],S_composition=first['S_snapshot_composition'],G_pending_observed=first['G_pending_logged'],C_S_ms=first['C_S_ms'],C_G_ms=first['C_G_ms'],gap_ms=first['cost_gap_ms'],entry_ms=P['entry_margin_ms'],exit_ms=P['exit_margin_ms'],S_EWMA_ms=first['EWMA_S_ms'],G_target_EWMA='UNDEFINED; NO_FORMAL_TARGET_G_RESULTS',G_exec_est_ms=P['G_service_ms'],historical_service_parameters_known_before_Module28=True,original_guard_bits=dict(admitted=first['admission_observed_guard'],capacity_ok=first['capacity_observed_guard']),no_earlier_counterfactual_diversion=True,all_prior_frozen_call_site_decisions_blocked=True,native_S_handler_active_retrospective=native_s,native_G_handler_active_retrospective=native_g,S_not_yet_handler_started_retrospective=first['S_pending_logged']-native_s,hyp_G_empty_entry_interval_on_actual_baseline=containing,S_ge10_interval_on_actual_baseline=high_containing,exact_scheduled_arrival_all_guards_status='UNKNOWN',later_dynamic_decisions_or_utility='UNKNOWN; DO_NOT_REPLAY_BEYOND_FIRST_SINGLE_STEP')
save('first_trigger_evidence.json',first_info)
if first_info:
 i=first_info;duration=i['hyp_G_empty_entry_interval_on_actual_baseline']['duration_s'] if i['hyp_G_empty_entry_interval_on_actual_baseline'] else None
 first_text=f'''# 首个潜在触发：仅单步、实际调用时点\n\n请求`{i['request_id']}`，LID。计划到达t={i['planned_arrival_t_s']:.9f}s；原代码实际快照/调用t={i['recorded_decision_t_s']:.9f}s，相差{i['difference_ms']:.6f}ms。这些是两种不同证据时点，不能悄悄把稍后数据搬回计划到达。精确计划到达时的完整collector/policy状态为UNKNOWN。\n\n实际快照S未完成13个，组成全部LID；G未完成0个。C_S={i['C_S_ms']:.9f}ms，C_G={i['C_G_ms']:.9f}ms，差值{i['gap_ms']:.9f}ms，超过entry={i['entry_ms']:.9f}ms及exit={i['exit_ms']:.9f}ms。不是用“13个”替代公式。S/G等待、服务与惩罚逐项见potential_trigger_events.csv。\n\nG执行估计505.636697ms来自GPU前冻结P95；正式阶段没有G LID响应，G目标EWMA并不存在，代码取冻结值。S LID EWMA在该快照由已观察完成前缀恢复为{i['S_EWMA_ms']:.9f}ms。每个用到的响应都不晚于该快照，历史P10/P95开发源哈希和早于冻结的时间已核验。S-only期间EWMA的已观察前缀最大82.928060ms，低于118.061907ms报警线，未触发S膨胀回退。该最大值是逐时点因果更新后的审计汇总，不作为更早请求的输入。\n\n从阶段初始化到首个单步选择，仅在原始S-only前缀上更新冻结滞回并检查守卫；首次选择G时立即停止连续策略推断。之前没有G目标分流，因此令牌始终为1、无G目标完成冷却、无G目标服务预测误差；G为空、在途目标0，原日志admitted/capacity_ok为PASS。全部此前调用均能说明阻断原因。该结论以原记录的调用时点及已观测基线事件不变为条件，不是动态GPU运行记录。\n\n持续性：在原S-only的host入队/reader响应轨迹上，假定G为空且用冻结估计，首个机会所属entry差值区间为{i['hyp_G_empty_entry_interval_on_actual_baseline']}，持续约{duration}s。该区间只描述基线潜在代价窗口；一次实际改路后它可能缩短、消失或改变。S≥10的相邻基线区间见first_trigger_evidence.json。\n\n事后native handler区间显示该时点S有{i['native_S_handler_active_retrospective']}个handler执行，G有{i['native_G_handler_active_retrospective']}个handler执行；其余{i['S_not_yet_handler_started_retrospective']}个S host未完成请求尚未开始handler。不能把这些全部叫worker已接受等待队列，也不把RESULT返回的START/END用作在线特征。\n\n原capacity_ok只验证监控资源/错误/进程，不认证加入G目标后的容量、质量、背景干扰或全窗口稳定性。本次原分类器unstable及U=NULL保持不变。服务分位数和惩罚也不是严格上下界。故该单步支持不能证明真实完成更快或全系统U上升。\n\n第一次反事实G路由之后，S队列少了一个请求，G目标执行及回退、冷却、令牌、服务EWMA、干扰和背景等待都可能改变。本审计对后续动态状态标UNKNOWN，不给出真实多次分流次数。\n'''
else:first_text='# 首个潜在触发\n\nEVIDENCE_INCOMPLETE：没有满足全部证据条件的首个单步判断。只提供门槛接近事件，不能计算精确机会次数。\n'
(OUT/'first_trigger_analysis.md').write_text(first_text)
make_plot(SEGMENTS,first)
# Compact summary is suitable for returning verbatim without exposing raw bodies.
lines=['# Module29 第一阶段：离线触发机会审计\n\n','状态：**EVIDENCE_INCOMPLETE**。真实事件与实际调用时点的单步分析可复核；精确计划到达时点的完整状态、首次反事实分流后的动态轨迹及收益无法判断。全程仅离线只读，没有启动GPU推理或修改Module28。\n\n','## 观测到的事实\n\n',f'正式窗口600s，发送{sent}，窗口内成功响应{within}，窗口末未完成{unresolved}；排空{reconciliation["drain_seconds"]:.9f}s后累计成功响应{len(success)}。窗口内完成12.625QPS，offered约12.678333QPS。未发现重复、缺失必需时间戳或600s边界事件，warmup与排空分开核验。原DYNAMIC_EVIDENCE_GAP、unstable、合法U=NULL保持不变。\n\n',f'S的host入队未完成数峰值{queue["peak_S_unfinished"]}，600s末为{queue["S_unfinished_at_600"]}。未完成数包括执行、host/pipe等待与响应传播，**不等于等待队列长度**。以下为600s的时间加权比例，分箱不是路由阈值：\n\n','|S未完成数|持续秒数|时间比例|\n|---|---:|---:|\n']
for x in distribution:lines.append(f'|{x["range"]}|{x["seconds"]:.6f}|{x["fraction"]*100:.4f}%|\n')
lines.append(f'\n时间加权分位数：{queue["time_weighted_quantiles"]}。S≥10累计{queue["high_queue_display_thresholds_only"]["ge10_seconds"]:.6f}s，最长连续区间{queue["high_queue_display_thresholds_only"]["ge10_longest_s"]:.6f}s。原全系统末150s队列斜率+{evidence["tail_backlog_slope"]:.9f}请求/s，S单worker对应斜率+{se["tail_backlog_slope"]:.9f}请求/s；均按原30s bin积分与OLS复现。\n\n')
lines.append('7,005个LID实际快照中的8项代价全部复核一致；4次S快照比reader响应重建多1个未完成请求，用原快照和单worker FIFO恢复。到达/决策时间分列，精确到达历史量没有使用更晚响应。\n\n## 反事实潜在机会\n\n')
if first_info:
 i=first_info;lines.append(f'首个有证据支持的**实际调用时点单步潜在触发**：`{i["request_id"]}`，t={i["recorded_decision_t_s"]:.9f}s（计划到达t={i["planned_arrival_t_s"]:.9f}s）。S=13个LID，实测G pending=0；C_S={i["C_S_ms"]:.6f}ms、C_G={i["C_G_ms"]:.6f}ms、差值{i["gap_ms"]:.6f}ms，结合原滞回及全部守卫支持这一步。这里使用完整公式，不是固定13阈值。原S-only只调用estimate，未执行decide或G目标路由，所以仍是条件单步反事实。\n\n')
lines.append(f'仅LID是路由目标，ASR等其余602个到达不会调用该路由器；ASR会影响S队列组成与等待估计。首步之前阻断原因：{dict(actual_block)}。其余各守卫PASS/FAIL/UNKNOWN及分母见guard_breakdown.json。独立逐点代价检查的数量只是基线诊断，**不是动态执行后的分流次数**。\n\n## 无法判断\n\n')
lines.append('精确scheduled_time_ns处没有collector状态提交及完整路由器快照，因此该时点的首个完整触发为UNKNOWN；稍后实际调用证据不能无条件回溯。首次反事实分流之后的滞回、令牌、冷却、回退及S/G队列/服务状态也为UNKNOWN，不计算精确多次机会数量。G空闲假设与真实S-only G pending证据分别记录。不能判断动态正收益、合法ΔU或正式部署能力。\n\n')
lines.append('分析准备：已有可复现的事件账本、冻结公式核验与首步假设，足以提出后续测量假设；**尚不具备原冻结负载下合法独立GPU对照的充分准备条件**。缺口是精确观测/harvest与路由状态日志、G目标执行后的干扰/误差数据，以及合法稳定的匹配S-only/固定/动态对照。需另行预注册和授权后续实验，不能放宽原unstable规则、重选有利种子或追溯更改U。本次不启动任何后续实验。\n\n')
validation=verify_inputs();save('input_integrity_after.json',validation)
lines.append(f'输入完整性：{validation["checked_input_files"]}个输入（其中Module28目录{validation["checked_module28_files"]}个文件）大小与SHA-256核对不变，原文件集合未增删。\n')
(OUT/'summary.md').write_text(''.join(lines))
save('offline_audit_gate.json',dict(status='EVIDENCE_INCOMPLETE',window_reconciliation=reconciliation['status'],recorded_cost_formula_checks='MATCH_ALL7005',first_recorded_decision_single_step_supported=first is not None,first_exact_planned_arrival_all_guards_known=False,exact_multi_diversion_count=None,Module28_gate_unchanged=True,original_classifier='unstable',legal_U=None,GPU_inferences_started=0,optimizer_modified=False,next_Module_started=False,input_integrity=validation,completed_and_stopped=True))
print(json.dumps(dict(status='EVIDENCE_INCOMPLETE',output_directory=str(OUT),window=reconciliation['observed'],time_weighted_ranges=distribution,first=first_info,guard_counts=guards,integrity=validation),ensure_ascii=False,indent=2))
