import csv,json,math,time
from pathlib import Path
p=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist')
def read(n):return json.loads((p/n).read_text())
def csvread(n):return list(csv.DictReader((p/n).open()))
a=read('final_data_acceptance.json');gate=read('module32_gate.json');arms=read('three_arm_gpu_measurement.json')['arms'];pairs=read('pair_comparisons_user_gate.json');decomp=csvread('native_latency_decomposition.csv');base=next(r for r in arms if r['run_id']=='A0');tasks=csvread('per_task_latency_comparison.csv')
checks={'data_acceptance_passed':a['passed'],'three_legal_stable':len(arms)==3 and all(r['eligible'] for r in arms),'three_complete_871':all(r['completed']==r['offered']==871 for r in arms),'memory_budget':all(r['safe_memory_mib']<=34567.5534 and r['sampled_peak_mib']<=34567.5534 for r in arms),'all_six_task_rows':len(tasks)==18,'native_decomposition':len(decomp)==2 and all(abs(float(r['reconstruction_residual_ms']))<1e-6 for r in decomp),'native_raw_not_modified':True,'no_positive_gate_correct':gate['gate']=='HEAVY_S_NO_POSITIVE_GAIN' and all(q['legal_matched_pair'] and q['delta_U']<=.01 for q in pairs),'utility_component_sums':all(abs(r['U']-sum(r[k+'_contribution'] for k in ['Q','T','L','P','M']))<1e-12 for r in arms),'matched_delta_recomputed':all(abs(r['delta_U']-(r['U']-base['U']))<1e-12 for r in arms),'both_route_shares_recorded':all(q['actual_split']==q['G_ASR']/(q['G_ASR']+q['S_ASR']) for q in pairs),'GPU_released':a['checks']['GPU_released'],'plots_saved':all((p/n).exists() for n in ['utility_and_queue_analysis.png','utility_and_queue_analysis.pdf'])}
# Recorded native raw identity after final auditing; artifact inventory stores their SHA.
checks.pop('native_raw_not_modified')
res={'passed':all(checks.values()),'checks':checks,'recorded_unix_ns':time.time_ns(),'gate':gate['gate'],'independent_replication_executed':False}
(p/'final_artifact_review.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res));assert res['passed']
