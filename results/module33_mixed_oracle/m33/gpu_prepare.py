"""All eight configurations and all random seeds frozen before new GPU runs."""
from .common import *
import random
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from replica_scaling.telemetry import snapshot

def trajectory(rates,seconds,seed,label,sample_path):
    pool=SamplePool(sample_path,list(rates),seed);events=arrivals(rates,seconds,seed,label)
    for e in events:e['item']=pool.next(e['task'])
    name=f'trajectories/{label}_{seed}_{seconds:g}.json';save(name,dict(duration=seconds,seed=seed,label=label,rates=rates,events=events))
    return ref(OUT/name)

def main():
    assert (OUT/'CPU_stage_completed.json').exists();assert not (OUT/'frozen_gpu_manifest.json').exists()
    fm=read(OUT/'frozen_problem_manifest.json');gp=fm['GPU_protocol'];space=read(fm['space']['path']);d=read(space['pools']['SMALL6']['path']);rates=space['loads']['ASR32B']['rates']
    source=ROOT/'results/module32b_vqa_aware_routing/frozen_gpu_manifest.json';old=read(source);plan=copy.deepcopy(old['plan'])
    plan['models']=d['models'];registry={m:copy.deepcopy(old['plan']['registry'][m]) for m in d['models']}
    for i,(m,entry) in enumerate(registry.items()):
        path=OUT/'launchers'/f'worker_{i}.py';path.parent.mkdir(exist_ok=True)
        path.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker.py'))+','+repr(entry['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n")
        entry['script']=str(path)
    plan['registry']=registry;plan['environment_overrides'].update(PYTHONPATH=str(OUT)+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
    assert all(baseline[k]==old['GPU'][k] for k in ['uuid','name','driver','total_mib'])
    dev={label:trajectory(rates,sec,gp['development_seed'],label,plan['sample_pool']['path']) for label,sec in [('warmup',30),('measurement',gp['development_seconds'])]}
    pairtraces={}
    for seed in gp['independent_seeds']:
        pairtraces[str(seed)]={label:trajectory(rates,sec,seed,label,plan['sample_pool']['path']) for label,sec in [('warmup',30),('measurement',gp['independent_pair_seconds'])]}
    candidates=read(OUT/'gpu_candidates.json');order=list(candidates);random.Random(gp['development_order_seed']).shuffle(order);slots=[]
    for c in order:
        slots.append(dict(run_id='DEV_'+c['candidate_id'],candidate_id=c['candidate_id'],deployment=c['deployment'],routing=c['routing'],
            lambda_rps=rates,seed=gp['development_seed'],measurement_seconds=gp['development_seconds'],safe_memory_budget_mib=space['budget_mib'],
            stage='DEVELOPMENT_ALL_CANDIDATES',trajectories=dev,predicted_U=c['predicted_U']))
    # SHA each input media file used by the frozen traces; also retain full model
    # checkpoint history baseline (m33.history) without modifying any weight.
    media={}
    for trace in [*dev.values(),*(v for tr in pairtraces.values() for v in tr.values())]:
        for e in read(trace['path'])['events']:
            inp=e['item']['sample'].get('input')
            if isinstance(inp,str) and inp.startswith('/'):
                if inp not in media:media[inp]=ref(inp)
    f=dict(module=33,frozen_unix_ns=time.time_ns(),plan=plan,GPU=baseline,slots=slots,pair_trajectories=pairtraces,
        safe_factor=d['safe_factor'],thermal_protocol=old['thermal_protocol'],problem_manifest=ref(OUT/'frozen_problem_manifest.json'),
        selected_pool_catalog=space['pools']['SMALL6'],registry_original={m:old['plan']['registry'][m] for m in d['models']},
        native_sources=[ref(ROOT/p) for p in ['module20/optimizer.py','module9bc/frozen_v3.py','module9bc/integration.py','method_v2/engine.py','method_v2/integer_search.py','joint_deploy/measurement/deployment_runtime.py','module30/gpu.py','module30/telemetry.py','module30/telemetry_v2.py','module11b/measure.py','performance_model/local_calibration/stability_v2.py']]+[ref(old['plan']['registry'][m]['script']) for m in d['models']],
        sources=[ref(p) for p in (OUT/'m33').glob('*.py')]+[ref(p) for p in (OUT/'launchers').glob('*.py')]+[ref(OUT/'restricted_worker.py')],
        inputs=[ref(plan['sample_pool']['path']),*media.values(),*dev.values(),*(v for tr in pairtraces.values() for v in tr.values())],
        adaptations=['Module32F/32G isolation/cache namespace only','Module30 SafeMonitorV2 own output namespace, timing wrapper','Original static phase/runtime, slot window120/240s; no dynamic policy','Read-only corrected nvidia-smi clock_event fields'],
        expected_all_candidates=8,no_topK_filter=True,no_config_changes_after_GPU=True)
    save('frozen_gpu_manifest.json',f)
    print('GPU_FROZEN_ALL',len(slots),'order',[s['candidate_id'] for s in slots],flush=True)
if __name__=='__main__':main()
