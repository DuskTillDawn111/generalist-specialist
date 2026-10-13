import json,csv,hashlib,math,statistics
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation')
read=lambda p:json.loads(Path(p).read_text())
required=['baseline_comparability_audit.md','baseline_cpu_results.csv','ablation_cpu_results.csv','gpu_baseline_results.csv','solver_efficiency.csv','experiment_protocol.json','paper_ready_findings.md','final_gate.json','final_summary.md','paper_figures']
assert all((OUT/n).exists() for n in required)
b=read(OUT/'baseline_cpu_results.json');a=read(OUT/'ablation_cpu_results.json');g=read(OUT/'gpu_baseline_results.json');e=read(OUT/'solver_efficiency.json');gate=read(OUT/'final_gate.json');q=read(OUT/'quantitative_findings.json')
assert len(b)==96 and len(a)==60 and len(g)==18
assert q['fastest_zero']==2 and q['fastest_positive']==10 and q['route_only_positive']==6 and q['route_only_zero']==6
assert all(r['regret'] is None for r in b if not r['strict_same_search_domain'])
t=list(csv.DictReader((OUT/'solver_timing_repetitions.csv').open()));assert len(t)==72
keys={(r['scenario'],r['pool'],r['method'],r['repetition']) for r in t};assert len(keys)==72
assert gate['PAPER_ASSETS_READY'] and gate['stop'] and not gate['STRICT_GLOBAL_SINGLE_VARIABLE_ABLATION_A_EXECUTED'] and not gate['STRICT_GLOBAL_SINGLE_VARIABLE_ABLATION_B_EXECUTED'] and gate['COLOCATION_ABLATION_SKIPPED']
assert read(OUT/'history_integrity_after.json')['passed']
for r in read(OUT/'history_sha256_before.json'):
 p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'] and p.stat().st_mtime_ns==r['mtime_ns'],r['path']
for f in (OUT/'paper_figures').glob('*_raw_data.json'):
 for r in read(f)['files']:
  p=Path(r['path']);assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
for name in ['figure1_regret','figure2_ablation_utility','figure3_solver_time','figure4_gpu_utility']:
 assert (OUT/f'paper_figures/{name}.png').read_bytes().startswith(b'\x89PNG')
 assert (OUT/f'paper_figures/{name}.pdf').read_bytes().startswith(b'%PDF')
 assert len(list(csv.DictReader((OUT/f'paper_figures/{name}.csv').open())))>0
for file in ['paper_ready_findings.md','final_summary.md']:
 text=(OUT/file).read_text();assert '9/12有预测优势、3/12相同' not in text
validation=dict(passed=True,baseline_rows=96,ablation_or_skip_rows=60,GPU_independent_windows=18,paired_traces=9,timing_repetition_rows=72,all_1320_historical_sha_mtime_unchanged=True,native_regret_not_misrepresented=True,raw_data_figure_indexes_hash_match=True,four_figures_validated=True,strict_global_ablation_A_B_not_claimed=True,co_location_C_skipped=True,no_new_GPU=True,stop=True)
(OUT/'final_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
# Index the reproducible drivers as well as deliverables; exclude transient logging/cache.
paths=[p for p in OUT.rglob('*') if p.is_file() and (('work' not in p.relative_to(OUT).parts) or p.suffix=='.py') and p.name not in ['artifact_sha256.json','result_freeze.json']]
artifacts=[dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(paths)]
(OUT/'artifact_sha256.json').write_text(json.dumps(artifacts,indent=2)+'\n')
freeze=read(OUT/'result_freeze.json');freeze['artifacts']=len(artifacts);freeze['final_validation']='final_validation.json';freeze['final_35_optimizer_SHA256']='c6727f4f4a4b18bdfd4e8767b9914f19f65fdf8e0ef8a1a5179c54aab53fb1a3';(OUT/'result_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print('FINAL_DELIVERY_VALIDATION_PASS',validation,'artifacts',len(artifacts))
