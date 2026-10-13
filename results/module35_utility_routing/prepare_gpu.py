"""Freeze only the missing C/TEXT10 selected plan; reuse A/B matched windows."""
import sys,time,copy,shutil
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module35_utility_routing';OLD=ROOT/'results/module34_candidate_pool'
sys.path[:0]=[str(OUT),str(OLD),str(ROOT)]
from m35.common import *
from replica_scaling.telemetry import snapshot
assert read(OUT/'CPU_stage_completed.json')['passed']
assert not (OUT/'frozen_gpu_manifest.json').exists()
checks=read(OUT/'optimizer_correctness_tests.json');assert checks['all_passed']
dec=list(__import__('csv').DictReader((OUT/'cpu_oracle_comparison.csv').open()))
c=next(r for r in dec if r['scenario']=='TEXT_HALF_MID18' and r['pool']=='C' and r['algorithm']=='U_ALIGNED_DP')
x=json.loads(c['deployment']);routing=json.loads(c['routing'])
old=read(OLD/'frozen_gpu_manifest_v2.json');plan=copy.deepcopy(old['plan'])
registry={}
for i,m in enumerate(sorted(x)):
    entry=copy.deepcopy(old['registry_original'][m]);p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
    p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker.py'))+','+repr(entry['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n")
    entry['script']=str(p);registry[m]=entry
plan['registry']=registry;plan['environment_overrides'].update(PYTHONPATH=str(OUT)+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
cache=OUT/'work/cache';cache.mkdir(parents=True,exist_ok=True)
for name in ['cuda','triton','hf']:
    shutil.copytree(OLD/'work/cache'/name,cache/name,dirs_exist_ok=True)
baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
assert all(baseline[k]==old['GPU'][k] for k in ['uuid','name','driver','total_mib'])
slots=[]
for s in old['slots']:
    if s['group']!='A_TEXT_HALF_MID18' or s['arm']!='ORACLE':continue
    slots.append(dict(s,run_id=f'M35_{s["stage"]}{s["repetition"]}_C_TEXT10_SELECTED',candidate_id=c['algorithm']+'_C_TEXT10',
        group='C_TEXT_HALF_MID18',arm='SELECTED',pool='C',deployment=x,routing=routing,predicted_U=float(c['predicted_U'])))
protocol=dict(frozen_unix_ns=time.time_ns(),windows='30s warmup, 120s development, three 240s independent windows;60s bounded drain;fresh workerPIDs each slot',
    scenarios=['TEXT_HALF_MID18'],selection='Only previously missing C/TEXT10 winner. Oracle, new optimizer, old20A and MIN_MEMORY_COVER_FASTEST choose identical C plan; share one measurement.',
    paired_contrasts=['C selected minus B CPU_U_ORACLE','C selected minus B original20A','C selected minus A CPU_U_ORACLE'],
    historical_A_B='Exact identical request trajectories, samples, uniforms, GPU UUID/driver, budget, quality, normalization, thresholds and native measurement settings from34. Reused observations retain originalraw.',
    independent_statistics='Three shared seeds; paired mean difference and descriptive t95CI; Holm correction for three new planned contrasts; require both arms eligible and operating-state comparability. Cross-session order/time not randomized: disclose residual confounding.',
    environment_rule='Same GPU UUID/name/driver/total; final postwarm cooling sample temperature difference<=5C; independent window meanSM-clock relative difference<=5%. Report window temperatures/power separately because utilization differs by deployment; no posthoc rejection by U.',
    inconclusive_rule='CI covers0 or either arm illegal/unstable or environmentalcheck fails -> cannot claim advantage.',
    no_further_GPU_on_small_differences=True,not_full_real_GPU_oracle=True,method_equivalence_not_independent_repeats=True)
save('frozen_gpu_protocol.json',protocol)
f=dict(module=35,frozen_unix_ns=time.time_ns(),manifest_name='frozen_gpu_manifest.json',plan=plan,GPU=baseline,slots=slots,safe_factor=old['safe_factor'],thermal_protocol=old['thermal_protocol'],
    native_sources=old['native_sources'],sources=[ref(p) for p in (OUT/'m35').glob('*.py')]+[ref(p) for p in (OUT/'launchers').glob('*.py')]+[ref(OUT/'restricted_worker.py'),ref(OUT/'u_aligned_optimizer.py')],
    inputs=[ref(OUT/'frozen_gpu_protocol.json'),ref(OUT/'cpu_oracle_comparison.csv'),ref(OUT/'optimizer_source_freeze.json')]+list({s['trajectories'][l]['path']:s['trajectories'][l] for s in slots for l in ['warmup','measurement']}.values()),
    expected_distinct=1,expected_runs=4,reused_windows=16,full_pool_GPU_oracle=False)
save('frozen_gpu_manifest.json',f)
print('FROZEN',len(slots),x,flush=True)
