"""Six unchanged native workers, sequential context creation, real path probes."""
import traceback
from m32f.isolation import install,VIOLATIONS
install('supervisor')
from m32f.common import *
from m32f.harness import AuditedRuntime
from m32f.monitor import IncrementalMonitor
from module22.gpu import PoissonConfig
from module11b.measure import prime
from replica_scaling.telemetry import snapshot,release
f=read(OUT/'frozen_validation_manifest.json');assert all(read(OUT/'single_model_runs'/f'{i}.json')['passed'] for i in range(6))
assert not (OUT/'co_residency_raw.json').exists(),'One co-residency check'
for k,v in f['plan']['environment_overrides'].items():os.environ[k]=v
slot=f['slots'][1];mapping={m+'#0':m for m in slot['deployment']};routes={t:{m+'#0':q for m,q in row.items()} for t,row in slot['routing'].items()}
cfg=PoissonConfig(deployment={k:1 for k in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=0,measurement_seconds=1,startup_timeout_seconds=300,drain_timeout_seconds=60,shutdown_timeout_seconds=15)
cfg.validate({k:f['plan']['models'][m]['supported_tasks'] for k,m in mapping.items()})
before=snapshot();assert not before['processes'] and before['used_mib']<=128
rt=AuditedRuntime(cfg,{k:f['plan']['registry'][m] for k,m in mapping.items()});mon=IncrementalMonitor(rt,dict(run_id='CO_RESIDENCY'),slot['safe_memory_budget_mib']);mon.start();result=dict(passed=False,errors=[],baseline=before)
try:
 rt.start();mon.check();result['identities']={k:dict(pid=w.process.pid,READY=rt.ready[k]) for k,w in rt.workers.items()}
 result['path_probe_records']=prime(rt,cfg,f['plan'],mapping,slot);mon.check()
 result['idle_samples']=mon.idle_window('co_residency',2.);mon.check()
 result['safe_memory_mib']=f['safe_factor']*float(np.mean([s['owned_used_mib'] for s in result['idle_samples']]))
 result['passed']=result['safe_memory_mib']<=slot['safe_memory_budget_mib'] and all(r['status']=='completed' for r in result['path_probe_records'])
except BaseException as exc:result['errors'].append(repr(exc));result['traceback']=traceback.format_exc()
finally:
 result['shutdown']=rt.close();mon.close();result['errors']+=rt.errors+mon.errors;result['telemetry']=mon.samples;result['control_probes']=mon.control_probes
 result['GPU_release']=release(before,[w.process.pid for w in rt.workers.values() if w.process]);result['unexpected_writes']=VIOLATIONS
 guards=[read(OUT/'work/control_guards'/f"{v['pid']}_exit.json") for v in result.get('identities',{}).values() if (OUT/'work/control_guards'/f"{v['pid']}_exit.json").exists()]
 result['worker_exit_guards']=guards;result['passed']=bool(result['passed'] and len(guards)==6 and not any(v['unexpected_writes'] for v in guards) and not result['errors'] and not VIOLATIONS and result['GPU_release']['verified'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result['shutdown'].values()))
 result['U']=None;save('co_residency_raw.json',result);save('co_residency_acceptance.json',dict(passed=result['passed'],errors=result['errors'],safe_memory_mib=result.get('safe_memory_mib'),control_checks=len(mon.control_probes),control_max_ms=max(mon.check_seconds,default=0)*1000,raw=ref(OUT/'co_residency_raw.json'),U=None))
print('CO_RESIDENCY_FINISH',result['passed'],result['errors'],flush=True)
raise SystemExit(0 if result['passed'] else 1)
