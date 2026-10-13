"""Honest engineering/utility separation, source/history audit, no future GPU work."""
import sys,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32d.common import *
from m32d.campaign import assess,paired_environment
from vqa_aware_dynamic_router import VQAAwareRouter
from replica_scaling.telemetry import snapshot

def main():
 f=read(OUT/'frozen_validation_manifest.json');status=read(OUT/'campaign_status.json');assert status['status'] in ['COMPLETED','STOPPED_INCONCLUSIVE']
 hist=read(OUT/'protected_history_before.json');changed=[]
 for r in hist:
  p=Path(r['path'])
  if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns]!=[r['bytes'],r['mtime_ns']]:changed.append(r['path'])
 incident=read(OUT/'cpu_output_namespace_incident/incident.json');assert changed==[incident['affected_existing_file']],changed
 checks=dict(all_frozen_sources_inputs_SHA=all(ref(r['path'])['sha256']==r['sha256'] for r in f['sources']+f['frozen_inputs']),original_policy_byte_identical=ref(OUT/'vqa_aware_dynamic_router.py')['sha256']==ref(OLD/'vqa_aware_dynamic_router.py')['sha256'],original_parameters_byte_identical=ref(OUT/'frozen_routing_parameters.json')['sha256']==ref(OLD/'frozen_routing_parameters.json')['sha256'],authoritative_history_metadata_unchanged=changed==[incident['affected_existing_file']],all_historical_files_byte_exact_preserved=not changed)
 env={}
 for py,prior in f['environment'].items():
  v=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));env[py]=dict(unchanged=v==prior,current=v)
 checks['worker_dependencies_unchanged']=all(v['unchanged'] for v in env.values());save('final_environment_identity.json',env)
 bad=[]
 prior=read(OUT/'final_data_acceptance.json') if (OUT/'final_data_acceptance.json').exists() else None
 if prior and prior['checks']['all_model_weights_full_SHA_unchanged']:
  assert all([Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v for p,v in f['checkpoint_metadata'].items())
 else:
  for i,r in enumerate(f['checkpoints']):
   if ref(r['path'])['sha256']!=r['sha256']:bad.append(r['path'])
   if i%20==0:print('WEIGHT_SHA',i,len(f['checkpoints']),flush=True)
 checks['all_model_weights_full_SHA_unchanged']=not bad
 critical=[]
 for p in (ROOT/'results').glob('module*/*gate*.json'):
  if OUT in p.parents:continue
  original=next((r for r in hist if r['path']==str(p)),None)
  critical.append(dict(path=str(p),metadata_unchanged=original is not None and [p.stat().st_size,p.stat().st_mtime_ns]==[original['bytes'],original['mtime_ns']]))
 checks['all_historical_Gates_unchanged']=all(r['metadata_unchanged'] for r in critical);save('historical_gate_preservation.json',critical)
 for n,source in [('Module32B',ROOT/'results/module32c_dynamic_replication/module32b_full_file_hashes.json'),('Module32',OLD/'historical_module32_file_hashes.json')]:
  checks[n+'_all_files_SHA_unchanged']=all(ref(r['path'])['sha256']==r['sha256'] for r in read(source))
 gpu=snapshot();checks['GPU_released']=not gpu['processes'] and gpu['used_mib']<=128
 raws=[];engineering=[];resources=[];tasks=[];paths=[];requests=[];router_rows=[];replay=[];all_sensors=[]
 for s in f['engineering_slots']+f['slots']:
  p=OUT/'raw_runs'/f'{s["run_id"]}.json'
  if not p.exists():continue
  r=read(p);raws.append(r);ph=r.get('phases',{}).get('measurement');rid=s['run_id']
  obs=assess(r) if ph else dict(run_id=rid,eligible=False,technical_valid=False,stability='NO_MEASUREMENT',U=None,errors=r['errors'])
  engineering.append(dict(run_id=rid,arm=s['arm'],stage=s['stage'],valid_session=r['valid_session'],engineering_eligible=obs['eligible'],technical_valid=obs['technical_valid'],stability=obs.get('stability'),formal_U=None,offered=obs.get('offered'),completed=obs.get('completed'),G_ASR=obs.get('G_completed'),S_ASR=obs.get('S_completed'),router_p95_ms=float(np.percentile([d['routing_compute_ms'] for d in ph['routing_decisions']],95)) if ph and ph['routing_decisions'] else None,measurement_started=bool(ph),failure=r['errors']))
  all_sensors.extend(dict(run_id=rid,**v) for v in r['telemetry']['samples'])
  if ph:
   tasks.extend(dict(run_id=rid,engineering_only=True,task=t,**v) for t,v in obs['per_task'].items());paths.extend(dict(run_id=rid,engineering_only=True,**v) for v in obs['paths'])
  for label,phase in [('conditioning',r.get('conditioning')),('warmup',r.get('warmup')),*r.get('phases',{}).items()]:
   if not phase:continue
   requests.extend(dict(run_id=rid,engineering_only=True,**x) for x in phase['request_records'])
   ds=phase.get('routing_decisions',[]);rr=phase['request_records'];w=phase['window'];ss=[v for v in r['telemetry']['samples'] if w['start_time_ns']<=v['time_ns']<=w['end_time_ns']]
   resources.append(dict(run_id=rid,phase=label,temperature=stats(v['temperature_gpu_c'] for v in ss),SM_clock=stats(v['sm_clock_mhz'] for v in ss),active_SM_clock=stats(v['sm_clock_mhz'] for v in ss if v['gpu_util_percent']>0),idle_SM_clock=stats(v['sm_clock_mhz'] for v in ss if v['gpu_util_percent']==0),power=stats(v['power_draw_w'] for v in ss),sampled_peak_owned_mib=max((v['owned_used_mib'] for v in ss),default=None),hardware_thermal_slowdown_samples=sum(v['hw_thermal_slowdown']=='Active' for v in ss),software_thermal_slowdown_samples=sum(v['sw_thermal_slowdown']=='Active' for v in ss),router_wrapper_ms=stats(d['routing_compute_ms'] for d in ds),six_task_offered=dict(collections.Counter(x['task'] for x in rr)),six_task_completed=dict(collections.Counter(x['task'] for x in rr if x['status']=='completed'))))
   if not ds:continue
   router=VQAAwareRouter(read(OUT/'frozen_routing_parameters.json'),read(OUT/'audio_metadata_cache.json'),w['duration_seconds']);ev=read(s['trajectories'][label]['path'])['events'];inputs={x['request_id']:e['item']['sample']['input'] for x,e in zip(rr,ev) if x['task']=='asr'}
   for d in ds:
    v=router.decide(d['decision_perf_ns'],inputs[d['request_id']],d['past_state_snapshot']);past=all(x['enqueue_perf_ns']<=d['decision_perf_ns'] and (x['submit_perf_ns'] is None or x['submit_perf_ns']<=d['decision_perf_ns']) for x in d['past_state_snapshot']);replay.append(dict(run_id=rid,phase=label,request_id=d['request_id'],exact_replay=all(v[k]==d[k] for k in v),past_only=past));router_rows.append(dict(run_id=rid,**d))
 csvsave('gpu_paired_runs.csv',engineering);csvsave('engineering_per_task_latency.csv',tasks);csvsave('engineering_worker_service_wait.csv',paths);csvsave('engineering_resource_comparison.csv',resources);csvsave('engineering_all_requests.csv',requests);csvsave('engineering_router_decisions.csv',router_rows);csvsave('gpu_sensor_samples.csv',all_sensors);save('causal_router_replay_audit.json',replay)
 checks['executed_dynamic_decisions_exact_and_causal']=all(x['exact_replay'] and x['past_only'] for x in replay)
 formal=[r for r in raws if r['slot']['pair']>0];checks['formal_measurements_not_started']=not formal
 csvsave('utility_comparison.csv',[dict(pair=i,S_U=None,Dynamic_U=None,delta_U=None,status='NOT_EXECUTED',reason='Integrity gate prevents independent utility validation',not_engineering_U=True) for i in (1,2,3)])
 save('planned_run_disposition.json',[dict(run_id=s['run_id'],stage=s['stage'],seed=s['seed'],measurement_seconds=s['measurement_seconds'],trajectory=s['trajectories']['measurement'],status='COMPLETED_ENGINEERING_ONLY' if any(r['slot']['run_id']==s['run_id'] and r.get('phases',{}).get('measurement') for r in raws) else 'FAILED_BEFORE_MEASUREMENT' if any(r['slot']['run_id']==s['run_id'] for r in raws) else 'NOT_EXECUTED') for s in f['engineering_slots']+f['slots']])
 failure=[r for r in raws if not r['valid_session']];thermal=any('THERMAL' in err for r in failure for err in r['errors'])
 gate='GPU_ENVIRONMENT_UNCONTROLLED' if thermal else 'DYNAMIC_VALIDATION_INVALID'
 engineering_valid=bool(len(engineering)==2 and all(x['engineering_eligible'] for x in engineering))
 eq=read(OUT/'routing_equivalence_report.json');cpu=read(OUT/'CPU_runtime_acceptance.json')
 final=dict(gate=gate,engineering_gate='GPU_ENVIRONMENT_UNCONTROLLED' if thermal else 'SHORT_ENGINEERING_COMPLETE' if engineering_valid else 'ENGINEERING_INCOMPLETE',utility_gate='NOT_VALIDATED',runtime_CPU_performance_fixed=cpu['passed'],router_semantics_changed=False,routing_equivalence_exact=eq['passed'],formal_runs=0,three_paired_deltas=None,CI95=None,positive_replicated=False,formal_validation_cancelled_for_history_integrity=True,history_file_byte_exact_preservation=False,historical_original_requests_Gates_model_weights_unchanged=checks['authoritative_history_metadata_unchanged'] and checks['all_historical_Gates_unchanged'] and checks['all_model_weights_full_SHA_unchanged'],non_exact_ephemeral_progress_restoration=incident,parameters_adjusted=False,seeds_replaced=False,optimizer_modified=False,next_stage_started=False,independent_replication_claimed=False,all_results_retained=True)
 save('module32d_gate.json',final);save('final_data_acceptance.json',dict(checks=checks,scientific_asset_integrity_passed=all(v for k,v in checks.items() if k!='all_historical_files_byte_exact_preserved'),full_history_file_preservation_passed=False,changed_history=changed,bad_checkpoint_SHA=bad,checkpoint_count=len(f['checkpoints']),GPU=gpu,incident=incident))
 summary=f'''# Module32D — runtime repair, validation not completed

Gate: `{gate}`. Engineering and utility conclusions are separate. No independent2400s GPU measurement was started; all three formal ΔU and the planned paired confidence interval are NULL. No utility result or reproduction success is claimed.

1. **Why overhead grew:** the original wrapper repeatedly scanned cumulative completed history under the shared lock; the monitor also deep-copied the complete cohort every0.5s before checking its10s logging interval. Pure decision rules were already cheap.
2. **1ms threshold:** two paced real-thread/real-fsync2400s-scale CPU tests passed unchanged1ms: P95 {cpu['results'][0]['p95_ms']:.6f}/{cpu['results'][1]['p95_ms']:.6f}ms, with retained histories of {cpu['results'][0]['history_records']}/{cpu['results'][1]['history_records']} records. These are50x accelerated CPU tests, not2400s wall-time GPU validation. GPU long-window acceptance remains unestablished.
3. **Routing semantics:** all {eq['decisions']} historical32B/32C decisions and every returned policy field match exactly at identical observed state, tolerance0. Original router source/parameters are byte-identical. Native error, duplicate, timeout, late-response and exit tests pass. Actual collector visibility is not inferred from unlogged historical collector time or future worker END. Nonpreemptive ASR can still block newly arriving VQA.
4. **GPU environment:** {len(raws)} engineering runs have raw evidence; {sum(bool(r.get('phases',{}).get('measurement')) for r in raws)} reached the180s window. New protocol freezes useful300s common S-only conditioning, original30s warmup/cooldown, reload, actual clock/power/fan/throttle monitoring, original5°C/15% paired limits. {'Original resident40°C cooldown was not achieved within180s; environment control failed, so measurement and further GPU runs stopped.' if thermal else 'Short engineering evidence does not establish long-window environmental comparability.'}
5. **System utility:** unknown. Preflight/warmup/conditioning U is excluded; no truncated-cohort or historical U substitutes for independent paired U. Text/VQA P95, Whisper waiting benefit and Phi interference have no independent32D long-window contrast.
6. **Optimizer integration:** no. Runtime repair has CPU support; valid controlled long-window utility evidence is still required. No thresholds/models/rates/seeds were changed to seek a positive result, no formal optimizer was edited, and no next Module was started.

## Output namespace incident and strict preservation limit

The first CPU benchmark inherited Module30's telemetry output globals. It created two test journals and overwrote only Module30's ephemeral live_progress.json. Original request logs, experimental Gate files, source and weights were not changed. Test journals were moved to this Module's cpu_output_namespace_incident/ directory, and the historical progress was reconstructed from untouched C2_split05 checkpoint/native-window data. Its precise previous UI elapsed time is unavailable, so this is explicitly **not a byte-exact restoration**. The pre-task metadata audit detects exactly that one historical file. This violates full historical-file preservation; formal validation was cancelled, even though authoritative scientific assets remain unchanged. The failure is retained, not hidden or treated as a routing negative.

The original frozen CPU fixture is retained as incident evidence; reproduce through revalidate_cpu_safe.py, which explicitly redirects inherited telemetry into Module32D/cpu_safe_revalidation/. Its post-engineering results are engineering-only and do not refreeze any GPU policy or trajectory. Frozen manifests remain intact. See final_data_acceptance.json, cpu_output_namespace_incident/incident.json and all original failed/negative evidence.
'''
 (OUT/'module32d_summary.md').write_text(summary);save('module32d_summary.json',final)
 print('FINAL_GATE',gate,'CHECKS',checks,flush=True)
if __name__=='__main__':main()
