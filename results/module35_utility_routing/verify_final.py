import sys,json,csv,math,hashlib,time,platform
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing'
sys.path[:0]=[str(OUT),str(ROOT)]
from m35.common import *
from scipy.stats import t,ttest_1samp
from replica_scaling.telemetry import snapshot
required=['routing_gap_attribution.md','u_aligned_optimizer.py','optimizer_correctness_tests.json','cpu_oracle_comparison.csv','holdout_results.csv',
    'solver_runtime_comparison.csv','pool_expansion_decisions.csv','gpu_utility_comparison.csv','paper_figures','module35_gate.json','module35_summary.md']
assert all((OUT/n).exists() for n in required)
assert read(OUT/'optimizer_correctness_tests.json')['all_passed']
assert read(OUT/'history_integrity_after.json')['passed']
assert read(OUT/'pool_nesting_validation.json')['passed']
assert read(OUT/'campaign_status.json')['status']=='COMPLETED'
assert len(read(OUT/'gpu_observations.json'))==4
f=read(OUT/'frozen_gpu_manifest.json')
for r in f['sources']+f['native_sources']+f['inputs']:
    # Timings/report files are deliberately outside the measurement manifest.
    assert ref(r['path'])['sha256']==r['sha256'],r['path']
frz=read(OUT/'optimizer_source_freeze.json');assert all(ref(r['path'])['sha256']==r['sha256'] for r in frz['sources'])
comparisons=[]
for c in read(OUT/'gpu_paired_contrasts.json'):
    vals=[p['delta_U'] for p in c['pairs'] if p['delta_U'] is not None]
    if len(vals)<2:continue
    import numpy as np
    mean=float(np.mean(vals));sd=float(np.std(vals,ddof=1));half=float(t.ppf(.975,len(vals)-1)*sd/math.sqrt(len(vals)))
    p=float(ttest_1samp(vals,0).pvalue)
    old=c['statistics'];errors=dict(mean=abs(mean-old['mean']),ci=max(abs(a-b) for a,b in zip([mean-half,mean+half],old['ci95'])),p=abs(p-old['p']))
    assert max(errors.values())<1e-9
    comparisons.append(dict(contrast=c['contrast'],scipy_match=True,max_errors=errors))
gs=snapshot();assert not gs['processes'] and gs['used_mib']<=128
gate=read(OUT/'module35_gate.json');assert gate['gates']['CORE_EVIDENCE_READY']
save('final_validation.json',dict(passed=True,required_outputs=required,scipy_statistics_checks=comparisons,
    frozen_sources_unchanged=True,history_unchanged=True,GPU_released=gs,CPU_python=sys.version,
    numpy_version=np.__version__,platform=platform.platform(),remaining_experiments=[],no_next_module=True))
catalogue=[]
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or 'work' in p.relative_to(OUT).parts or p.name in ['scientific_artifact_sha256.json','result_freeze.json','verify_final.log']:continue
    catalogue.append(ref(p))
save('scientific_artifact_sha256.json',catalogue)
save('result_freeze.json',dict(module=35,status='FROZEN_CORE_COMPLETE',at_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    artifact_catalogue=ref(OUT/'scientific_artifact_sha256.json'),optimizer_source=ref(OUT/'u_aligned_optimizer.py'),gates=gate['gates'],
    scope='Frozen finite K1 CPU and one holdout workload; limited TEXT10 GPU paired reuse/supplement. GPU micro-differences unresolved.',stop=True))
print('FINAL_VALIDATION_PASS',len(catalogue),flush=True)
