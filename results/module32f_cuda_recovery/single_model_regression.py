"""Separate fresh controllers: one unchanged native GPU worker per invocation."""
import sys,traceback
from m32f.isolation import install,VIOLATIONS
install('supervisor')
from m32f.common import *
from m32f.harness import AuditedRuntime
from m32f.monitor import IncrementalMonitor
from module22.gpu import PoissonConfig
from module11b.measure import prime
from replica_scaling.telemetry import snapshot,release
f=read(OUT/'frozen_validation_manifest.json');index=int(sys.argv[1]);model=f['initialization_order'][index];key=model+'#0'
assert not (OUT/'single_model_runs'/f'{index}.json').exists(),'No single-worker retry'
for k,v in f['plan']['environment_overrides'].items():os.environ[k]=v
before=snapshot();assert not before['processes'] and before['used_mib']<=128
tasks=f['plan']['models'][model]['supported_tasks'];rates={t:.2 for t in tasks}
config=PoissonConfig(deployment={key:1},workload_rps=rates,routing_rps={t:{key:.2} for t in tasks},arrival_process='poisson',random_seed=320490,warmup_seconds=0,measurement_seconds=1,startup_timeout_seconds=300,drain_timeout_seconds=60,shutdown_timeout_seconds=15)
config.validate({key:tasks});rt=AuditedRuntime(config,{key:f['plan']['registry'][model]})
m=IncrementalMonitor(rt,dict(run_id=f'SINGLE_{index}'),f['slots'][0]['safe_memory_budget_mib']);result=dict(model=model,index=index,passed=False,errors=[],baseline=before);m.start()
try:
 rt.start();m.check();result['READY']=rt.ready[key];result['worker_pid']=rt.workers[key].process.pid
 result['path_probe_records']=prime(rt,config,f['plan'],{key:model},dict(seed=320490));m.check()
 result['passed']=all(r['status']=='completed' for r in result['path_probe_records'])
except BaseException as exc:result['errors'].append(repr(exc));result['traceback']=traceback.format_exc()
finally:
 result['shutdown']=rt.close();m.close();result['errors']+=rt.errors+m.errors
 result['GPU_release']=release(before,[w.process.pid for w in rt.workers.values() if w.process])
 result['control_checks']=m.control_probes;result['control_publish']=m.publish_seconds;result['telemetry']=m.samples
 result['controller_unexpected_writes']=VIOLATIONS
 pid=result.get('worker_pid')
 exitpath=OUT/'work/control_guards'/f'{pid}_exit.json'
 result['worker_guard']=read(OUT/'work/control_guards'/f'{pid}.json') if pid else None
 result['worker_exit_guard']=read(exitpath) if exitpath.exists() else None
 result['passed']=bool(result['passed'] and not result['errors'] and not VIOLATIONS and result['GPU_release']['verified'] and result['worker_exit_guard'] is not None and not result['worker_exit_guard']['unexpected_writes'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result['shutdown'].values()))
 result['U']=None;save(f'single_model_runs/{index}.json',result)
print('SINGLE_MODEL_FINISH',index,model,result['passed'],result['errors'],flush=True)
raise SystemExit(0 if result['passed'] else 1)
