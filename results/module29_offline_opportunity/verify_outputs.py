"""Independent CPU output/accounting/integrity verification; no GPU imports."""
import sys,os
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,csv,hashlib,math,collections
OUT=Path(__file__).resolve().parent
freeze=json.loads((OUT/'input_freeze.json').read_text());src=Path(freeze['read_only_source_directory']);raw=json.loads((src/'raw_runs/C_steady_s_only.json').read_text());rs=raw['phases']['measurement']['request_records'];w=raw['phases']['measurement']['window'];p=json.loads((src/'router_parameters.json').read_text());checks=[]
def load(name):return json.loads((OUT/name).read_text())
def passed(name,**data):checks.append(dict(name=name,passed=True,**data))
# Exact queue area from per-request clipped residence times, independent of event sweep.
q=load('queue_summary.json');duration=(w['end_time_ns']-w['start_time_ns'])/1e9
area=sum(max(0,min(x['completion_time_ns'],w['end_time_ns'])-max(x['enqueued_time_ns'],w['start_time_ns']))/1e9 for x in rs if x['model']==p['S_worker'])
with (OUT/'s_only_queue_timeline.csv').open() as f:events=list(csv.DictReader(f))
# Aggregate simultaneous state changes, then integrate host unfinished state.
last=w['start_time_ns'];n=0;integral=0.;peak=0;bytime=collections.defaultdict(list)
for x in events:bytime[int(x['time_ns'])].append(x)
for t,rows in sorted(bytime.items()):
 integral+=n*max(0,min(t,w['end_time_ns'])-max(last,w['start_time_ns']))/1e9
 n=int(rows[-1]['S_unfinished_after'])
 if t<w['end_time_ns']:peak=max(peak,n)
 last=t
assert math.isclose(area,integral,abs_tol=1e-7) and peak==q['peak_S_unfinished'];assert n==0
assert math.isclose(sum(x['seconds'] for x in q['time_weighted_ranges']),600.,abs_tol=1e-8) and math.isclose(sum(x['fraction'] for x in q['time_weighted_ranges']),1.,abs_tol=1e-10)
passed('queue_time_area_independent_per_request_identity',area_request_seconds=area,integrated_request_seconds=integral,peak=peak,total_time_seconds=600.)
rec=load('window_reconciliation.json');sent=sum(w['start_time_ns']<=x['actual_submit_time_ns']<w['end_time_ns'] for x in rs);within=sum(x['status']=='completed' and x['completion_time_ns']<w['end_time_ns'] for x in rs);drain=sum(x['status']=='completed' and x['completion_time_ns']>=w['end_time_ns'] for x in rs)
assert (sent,within,drain)==(7607,7575,32) and rec['observed']['sent_in_formal_window']==sent and rec['observed']['success_in_formal_window']==within
passed('window_boundaries_counts_and_drain',sent=sent,within=within,drain=drain)
with (OUT/'potential_trigger_events.csv').open() as f:triggers=list(csv.DictReader(f))
assert len(triggers)==len(rs) and {x['request_id'] for x in triggers}=={x['request_id'] for x in rs}
first=[x for x in triggers if x['final_trigger_state']=='SUPPORTED_FIRST_SINGLE_STEP_AT_RECORDED_DECISION'];assert len(first)==1;first=first[0]
assert first['request_id']=='measurement:1071'
assert json.loads(first['S_snapshot_composition'])=={'lid':13} and int(first['G_pending_logged'])==0
for k in ['hysteresis_prefix_guard','prediction_fallback_prefix_guard','cooldown_prefix_guard','token_prefix_guard','dynamic_G_state_prefix_guard','admission_observed_guard','capacity_observed_guard','observed_state_reconstruction_guard']:assert first[k]=='PASS'
assert float(first['cost_gap_ms'])>p['entry_margin_ms'] and math.isclose(float(first['C_S_ms'])-float(first['C_G_ms']),float(first['cost_gap_ms']),abs_tol=1e-10)
cutoff=float(first['observation_t_s']);past={x['request_id']:x for x in rs if x['completion_time_ns']<=w['start_time_ns']+round(cutoff*1e9)};ids=json.loads(first['S_pending_ids']);assert all(raw_id not in past for raw_id in ids)
for x in triggers:
 if x['task']=='lid' and float(x['observation_t_s'])>cutoff:
  assert all(x[k]=='UNKNOWN' for k in ['hysteresis_prefix_guard','prediction_fallback_prefix_guard','cooldown_prefix_guard','token_prefix_guard','dynamic_G_state_prefix_guard'])
 if x['task']=='lid':assert x['history_receipt_cutoff_proven']=='True' and x['formula_matches_original_snapshot']=='True' and x['strict_arrival_uses_later_logged_snapshot']=='False'
passed('first_single_step_and_no_post_diversion_route_inference',first_id=first['request_id'],recorded_decision_t_s=cutoff)
g=load('guard_breakdown.json')
for key,x in g['guards'].items():assert x['PASS']+x['FAIL']+x['UNKNOWN']==x['denominator']==7005
assert g['exact_multi_diversion_count'] is None and g['first_trigger_at_exact_planned_arrival'] is None and g['status']=='EVIDENCE_INCOMPLETE';passed('guard_denominators_and_unknowns')
changes=[]
for x in freeze['inputs']:
 h=hashlib.sha256()
 with Path(x['path']).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 if h.hexdigest()!=x['sha256'] or Path(x['path']).stat().st_size!=x['bytes']:changes.append(x['path'])
assert not changes
assert json.loads((src/'module28_gate.json').read_text())['gate']=='DYNAMIC_EVIDENCE_GAP' and raw['phases']['measurement']['stability']['stability_state_v2']=='unstable' and rec['legal_U'] is None
passed('all_input_hashes_and_module28_gate_stability_U_unchanged',inputs=len(freeze['inputs']),changes=changes)
for name in ['input_inventory.md','s_only_queue_timeline.csv','window_reconciliation.json','queue_summary.json','potential_trigger_events.csv','guard_breakdown.json','first_trigger_analysis.md','summary.md']:
 assert (OUT/name).is_file()
passed('required_artifacts_complete')
(OUT/'final_verification.json').write_text(json.dumps(dict(passed=True,status='EVIDENCE_INCOMPLETE',checks=checks,GPU_inferences_started=0,Module28_modified=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),input_files=len(freeze['inputs']),first_id=first['request_id'],input_hash_changes=0),indent=2))
