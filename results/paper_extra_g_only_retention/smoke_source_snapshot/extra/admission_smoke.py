"""Single co-residency/path feasibility check, never scores utility."""
from .isolation import install
install('supervisor')
from .common import *
import sys, traceback
sys.path.insert(0,str(ROOT))
from .gpu_native import AuditedRuntime, cool
from .telemetry_v2 import SafeMonitorV2
from module11b.measure import prime
from module22.gpu import PoissonConfig
from replica_scaling.telemetry import snapshot, release


def prepare():
    a=read(OUT/'admission_smoke_protocol.json');old=read(ROOT/'results/module34_candidate_pool/frozen_gpu_manifest_v2.json')
    plan=copy.deepcopy(old['plan']);d=read(OUT/'pools/P2.json');plan['models']=d['models'];registry={}
    for i,m in enumerate(a['deployment']):
        entry=copy.deepcopy(old['registry_original'][m]);p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
        p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker.py'))+','+repr(entry['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n")
        entry['script']=str(p);registry[m]=entry
    plan['registry']=registry
    plan['environment_overrides'].update(PYTHONPATH=str(OUT)+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    # Copy bytes only: confinement deliberately rejects chmod/utime from copy2.
    copied=0
    for name in ['cuda','triton','hf']:
        source=ROOT/'results/module35_utility_routing/work/cache'/name
        for p in source.rglob('*'):
            target=OUT/'work/cache'/name/p.relative_to(source)
            if p.is_dir():target.mkdir(parents=True,exist_ok=True)
            elif p.is_file():
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());copied+=1
    slot=dict(run_id='STAGE_A_CORESIDENT_PATH_SMOKE',deployment=a['deployment'],routing=a['routing'],lambda_rps=a['rates'],seed=351101,safe_memory_budget_mib=a['budget_mib'])
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
    assert all(baseline[k]==old['GPU'][k] for k in ['uuid','name','driver','total_mib'])
    freeze=dict(slot=slot,plan=plan,GPU=baseline,safe_factor=d['safe_factor'],thermal_protocol=old['thermal_protocol'],
        protocol=ref(OUT/'admission_smoke_protocol.json'),candidate_pool=ref(OUT/'pools/P2.json'),cached_files_copied=copied,
        sources=[ref(p) for p in (OUT/'extra').glob('*.py')]+[ref(p) for p in (OUT/'launchers').glob('*.py')]+[ref(OUT/'restricted_worker.py')],
        native_sources=[ref(ROOT/'module20/optimizer.py'),ref(ROOT/'module11b/measure.py'),ref(ROOT/'joint_deploy/measurement/deployment_runtime.py')]+[ref(old['registry_original'][m]['script']) for m in a['deployment']],
        sample_pool=ref(plan['sample_pool']['path']),frozen_unix_ns=time.time_ns(),utility_measurement=False)
    save('admission_smoke_manifest.json',freeze);return freeze


def main():
    assert not (OUT/'admission_smoke_result.json').exists(), 'Exactly one attempt; no outcome-based retries'
    assert (OUT/'history_sha256_before.json').exists()
    f=prepare();slot=f['slot'];pre=cool(slot,f);save('smoke_thermal_prelaunch.json',pre)
    baseline=pre['samples'][-1];plan=f['plan']
    for k,v in plan['environment_overrides'].items():os.environ[k]=v
    mapping={m+'#0':m for m in slot['deployment']};routes={t:{m+'#0':q for m,q in v.items()} for t,v in slot['routing'].items()}
    config=PoissonConfig(deployment={m:1 for m in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,
        arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=30.,measurement_seconds=1.,startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Admission feasibility serial path probes only; no utility window')
    config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
    runtime=AuditedRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=SafeMonitorV2(runtime,slot,slot['safe_memory_budget_mib'])
    raw=dict(status='STARTING',slot=slot,utility_measurement=False,U=None,errors=[],protocol=f['protocol'],manifest=ref(OUT/'admission_smoke_manifest.json'),baseline_GPU=baseline)
    started=time.perf_counter();monitor.start()
    try:
        print('SMOKE_START',list(slot['deployment']),flush=True);runtime.start();monitor.check()
        raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
        raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
        raw['idle_samples']=monitor.idle_window('postprime_idle',2.);monitor.check()
        raw['safe_idle_memory_mib']=f['safe_factor']*sum(v['owned_used_mib'] for v in raw['idle_samples'])/len(raw['idle_samples'])
        assert raw['safe_idle_memory_mib']<=slot['safe_memory_budget_mib']+1e-6
        raw['all_paths_completed']=all(r['status']=='completed' for r in raw['path_probe_records'])
        assert raw['all_paths_completed']
        raw['status']='COMPLETED'
    except Exception as exc:
        raw['status']='FAILED';raw['errors'].append(repr(exc));raw['traceback']=traceback.format_exc()
    finally:
        try:raw['shutdown']=runtime.close()
        except Exception as exc:raw['errors'].append('shutdown:'+repr(exc))
        try:monitor.close()
        except Exception as exc:raw['errors'].append('monitor:'+repr(exc))
        raw['errors']+=runtime.errors+monitor.errors
        raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,error_details=monitor.error_details)
        from .isolation import VIOLATIONS
        raw['unexpected_write_violations']=list(VIOLATIONS)
        raw['passed']=bool(raw['status']=='COMPLETED' and not raw['errors'] and raw['gpu_release']['verified'] and not VIOLATIONS and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw.get('shutdown',{}).values()))
        raw['elapsed_seconds']=time.perf_counter()-started;raw['campaign_finished']=True
        save('admission_smoke_result.json',raw)
    print('SMOKE_FINISHED',raw['passed'],'seconds',raw['elapsed_seconds'],'safe_memory',raw.get('safe_idle_memory_mib'),'errors',raw['errors'],flush=True)


if __name__=='__main__':main()
