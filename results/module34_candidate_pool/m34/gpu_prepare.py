from .common import *
import random,shutil
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from replica_scaling.telemetry import snapshot

def trajectory(rates,seconds,seed,label,sample_path,scenario):
    pool=SamplePool(sample_path,list(rates),seed);events=arrivals(rates,seconds,seed,label)
    for e in events:e['item']=pool.next(e['task'])
    name=f'trajectories/{scenario}_{label}_{seed}_{seconds:g}.json';save(name,dict(duration=seconds,seed=seed,label=label,rates=rates,events=events));return ref(OUT/name)
def main():
    assert (OUT/'CPU_stage_completed.json').exists() and not (OUT/'frozen_gpu_manifest.json').exists()
    assert read(OUT/'quality_and_profile_preservation.json')['passed']
    fm=read(OUT/'frozen_problem_manifest.json');gp=fm['GPU_budget'];sp=read(fm['space']['path']);d=read(sp['pools']['C']['path']);dec=read(OUT/'optimizer_decisions.json')
    groups=[('A','TEXT_HALF_MID18','20A_U_GRID'),('B','TEXT_HALF_MID18','20A_U_GRID'),('C','ASR32B','FASTEST_PATH_STATIC')];cs=[]
    for pool,scenario,method in groups:
        for arm,algorithm in [('ORACLE','CPU_U_ORACLE'),('COMPETITOR',method)]:
            row=next(x for x in dec if x['pool']==pool and x['scenario']==scenario and x['algorithm']==algorithm)
            assert row['status']=='FEASIBLE' and row['finite_admissible']
            cs.append(dict(row,group=pool+'_'+scenario,arm=arm))
    assert len({identity(c['deployment'],c['routing']) for c in cs})==6
    save('gpu_selection_record.json',dict(before_any_GPU=True,groups=groups,candidates=cs,
        amendment='Original CPU freeze budget permits six plans. Added A largest20A search gap from CPU development results; original manifest retained byte-for-byte. B maximal fastest/20A regret; C predeclared ASR32B. No GPU observations used.',full_pool_GPU_oracle=False))
    old=read(ROOT/'results/module32b_vqa_aware_routing/frozen_gpu_manifest.json');plan=copy.deepcopy(old['plan']);plan['models']=d['models'];registry={}
    original_registry=dict(read(ROOT/'results/module11_replica_and_cost/measurement_plan.json')['registry']);original_registry.update(read(ROOT/'results/module14_heavy_replica/measurement_plan.json')['registry']);original_registry.update(old['plan']['registry'])
    needed=sorted({m for c in cs for m in c['deployment']})
    for i,m in enumerate(needed):
        entry=copy.deepcopy(original_registry[m]);p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
        p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker.py'))+','+repr(entry['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n")
        entry['script']=str(p);registry[m]=entry
    plan['registry']=registry;plan['environment_overrides'].update(PYTHONPATH=str(OUT)+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    # Preserve exact native generated-cache bytes in the new writable namespace.
    cache=OUT/'work/cache';cache.mkdir(parents=True,exist_ok=True)
    for name in ['cuda','triton']:
        src=ROOT/'results/module33_mixed_oracle/work/cache'/name
        if src.exists():shutil.copytree(src,cache/name,dirs_exist_ok=True)
    modules=cache/'hf/modules';modules.mkdir(parents=True,exist_ok=True)
    (modules/'__init__.py').touch();(modules/'transformers_modules').mkdir(exist_ok=True);(modules/'transformers_modules/__init__.py').touch()
    copied=[]
    for modeldir in ['phi4-multimodal','minicpm-o-4_5']:
        dst=modules/'transformers_modules'/modeldir;dst.mkdir(exist_ok=True);(dst/'__init__.py').touch()
        for src in (ROOT/'models'/modeldir).glob('*.py'):
            target=dst/src.name;target.write_bytes(src.read_bytes());copied.append(dict(source=ref(src),target=ref(target)))
    save('native_cache_preseed.json',dict(no_inference_changes=True,byte_identical_dynamic_modules=copied))
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
    assert all(baseline[k]==old['GPU'][k] for k in ['uuid','name','driver','total_mib'])
    traces={};slots=[]
    for scenario in {c['scenario'] for c in cs}:
        rates=sp['scenarios'][scenario]['rates'];traces[scenario]={}
        for seed,seconds in [(gp['development_seed'],gp['development_seconds'])]+[(s,gp['independent_seconds']) for s in gp['independent_seeds']]:
            traces[scenario][str(seed)]={label:trajectory(rates,sec,seed,label,plan['sample_pool']['path'],scenario) for label,sec in [('warmup',30),('measurement',seconds)]}
    order=list(cs);random.Random(340102).shuffle(order)
    for stage,rep,seed,seconds,ordered in [('DEVELOPMENT',0,gp['development_seed'],gp['development_seconds'],order)]+[('INDEPENDENT',i+1,s,gp['independent_seconds'],cs if i%2==0 else list(reversed(cs))) for i,s in enumerate(gp['independent_seeds'])]:
        for c in ordered:slots.append(dict(run_id=f'{stage}{rep}_{c["group"]}_{c["arm"]}',candidate_id=c['candidate_id'],group=c['group'],arm=c['arm'],pool=c['pool'],scenario=c['scenario'],
            deployment=c['deployment'],routing=c['routing'],lambda_rps=sp['scenarios'][c['scenario']]['rates'],seed=seed,measurement_seconds=seconds,safe_memory_budget_mib=sp['budget_mib'],stage=stage,repetition=rep,trajectories=traces[c['scenario']][str(seed)],predicted_U=c['predicted_U']))
    media={};inputs=[]
    for scen in traces.values():
        for tr in scen.values():
            for r in tr.values():
                inputs.append(r)
                for e in read(r['path'])['events']:
                    inp=e['item']['sample'].get('input')
                    if isinstance(inp,str) and inp.startswith('/') and inp not in media:media[inp]=ref(inp)
    f=dict(module=34,frozen_unix_ns=time.time_ns(),plan=plan,GPU=baseline,slots=slots,safe_factor=d['safe_factor'],thermal_protocol=old['thermal_protocol'],
        problem_manifest=ref(OUT/'frozen_problem_manifest.json'),registry_original={m:original_registry[m] for m in needed},
        sources=[ref(p) for p in (OUT/'m34').glob('*.py')]+[ref(p) for p in (OUT/'launchers').glob('*.py')]+[ref(OUT/'restricted_worker.py')],
        native_sources=[ref(ROOT/p) for p in ['module20/optimizer.py','module9bc/frozen_v3.py','joint_deploy/measurement/deployment_runtime.py','module11b/measure.py','performance_model/local_calibration/stability_v2.py']]+[ref(original_registry[m]['script']) for m in needed],
        inputs=[ref(OUT/'gpu_selection_record.json'),ref(OUT/'candidate_space_definition.json'),ref(OUT/'native_cache_preseed.json'),ref(plan['sample_pool']['path']),*media.values(),*inputs],
        expected_distinct=6,expected_runs=24,full_pool_GPU_oracle=False,independent_arms_selected_before_development=True,no_algorithm_changes_after_GPU=True)
    save('frozen_gpu_manifest.json',f);print('GPU_FROZEN',len(slots),needed,flush=True)
if __name__=='__main__':main()
