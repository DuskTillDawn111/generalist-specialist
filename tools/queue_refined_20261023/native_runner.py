"""Opt-in native collection for causal dispatch; default CLI is CPU validation.

Reuse frozen worker transport, monitors, priming, timing and metric primitives.
Dynamic identities/evaluation are independent of static SWRR certificates.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.spillover_capacity_20261022.exposure import frozen_api
from tools.queue_refined_20261023.protocol import make_dispatcher,validate_contract,validate_packet,digest,receipt,write
from tools.queue_refined_20261023.runtime_window import window


def frozen_runtime():
    api=frozen_api();package=api['trace'].__package__
    return importlib.import_module(package+'.runtime.runner')


def run(system,winner,samples,policy_contract,*,backend='mock',workers=None,duration=600.,warmup=30.,seed=113,
        arrival_process='poisson',startup_timeout=300.,drain_timeout=300.,admission=None,
        capacity_evidence=None,allow_unverified_capacity_measurement=False,evaluator=None):
    """Return terminal raw, preserving original records plus causal decisions.

    Mock launches original CPU protocol workers with monitoring disabled. Native
    requires independent admission for all potential policy paths. Policy state
    is recreated per phase; model residency and warmup drain follow original.
    """
    if backend not in {'mock','native'}:raise ValueError('Unknown backend')
    validate_contract(policy_contract)
    if winner['deployment']!=policy_contract['deployment']:raise ValueError('Resident plan/contract mismatch')
    if system.weights!=policy_contract['weights'] or system.latency_reference_ms!=policy_contract['latency_refs_ms']:
        raise ValueError('System utility weights/references differ from policy identity')
    if system.workload.get('text')!=policy_contract['text_rate']:raise ValueError('Text workload differs from policy identity')
    for task,row in policy_contract['background_routing'].items():
        if sum(row.values())!=system.workload.get(task):raise ValueError('Background workload differs from policy identity')
    original=frozen_runtime();aliases,routes=original.expand_routes(winner['deployment'],winner['routing'])
    config=original.MeasurementConfig(deployment={a:1 for a in aliases},workload_rps=dict(system.workload),routing_rps=routes,
        warmup_seconds=warmup,measurement_seconds=duration,arrival_process=arrival_process,random_seed=seed,
        startup_timeout_seconds=startup_timeout,drain_timeout_seconds=drain_timeout,routing_source='causal_policy_contract_v1')
    config.validate({a:system.catalog['models'][m]['supported_tasks'] for a,m in aliases.items()})
    registry=original.registry_for(system,aliases,backend,workers or {})
    from tools.queue_refined_20261023 import admission as dynamic_admission
    identity=dynamic_admission.policy_identity(policy_contract)
    raw=dict(schema_version=1,backend=backend,config=system.as_dict(),transport_config=config.as_dict(),
        replica_to_family=aliases,plan=copy.deepcopy(winner),status='starting',errors=[],request_records=[],
        warmup_request_records=[],priming_request_records=[],priming_policy='largest_three_per_task',
        priming_size_metric='media_file_bytes_or_text_utf8_bytes_v1',measurement_window=None,preflight=None,
        worker_registry=registry,admission_evidence=dict(verified=False,reason='MOCK_BACKEND',pairs=[]),
        memory_evidence={},energy_evidence={},per_task_metrics={},system_metrics=None,
        policy_contract=copy.deepcopy(policy_contract),policy_identity=identity,strategy=policy_contract['strategy'],
        sample_pool_path=str(Path(samples).resolve()),samples_path=str(Path(samples).resolve()),
        dynamic_capacity_evidence=copy.deepcopy(capacity_evidence),
        allow_unverified_capacity_measurement=allow_unverified_capacity_measurement,
        measurement_claim='development causal native observation; no optimum or independent confirmation claim',
        static_SWRR_certificate_applicable=False)
    runtime=original.DeploymentRuntime(config,registry);monitor=None;power=None;before=after=None;startup_begin=None
    try:
        if backend=='native':
            checks=dynamic_admission.preflight(system,winner,samples,workers or {},policy_contract,admission=admission,
                require_gpu=True,capacity_evidence=capacity_evidence,
                allow_unverified_capacity_measurement=allow_unverified_capacity_measurement)
            raw['preflight']=checks;raw['admission_evidence']=checks['admission_evidence']
            if not checks['ready']:raise ValueError('Dynamic native preflight failed: '+repr(checks['issues']))
        else:checks=None
        device=checks['device'] if checks else '0'
        monitor=original.MemoryMonitor(runtime,system.gpu_memory_mib,enabled=backend=='native',device=device)
        power=original.PowerMonitor(enabled=backend=='native',device=device)
        power.sample();startup_begin=time.perf_counter_ns();power.start()
        runtime.start();monitor.sample();monitor.start();monitor.check()
        pool=original.SamplePool(samples,list(system.workload),seed)
        for alias,family in aliases.items():
            for task in system.catalog['models'][family]['supported_tasks']:
                if system.workload.get(task,0)<=0:continue
                for item in original.largest_inputs(pool,task):
                    row=original.record('priming',len(raw['priming_request_records']),task,alias,item,time.perf_counter_ns())
                    raw['priming_request_records'].append(row);runtime.enqueue(row,item['sample'])
        runtime.drain(raw['priming_request_records'],drain_timeout)
        if any(r['status']!='completed' for r in raw['priming_request_records']):raise RuntimeError('Priming failed')
        if warmup>0:
            raw['warmup_window']=window(runtime,config,samples,'warmup',warmup,raw['warmup_request_records'],monitor,make_dispatcher(policy_contract))
            if any(r['status']!='completed' for r in raw['warmup_request_records']):raise RuntimeError('Warmup failed')
        before=monitor.sample();monitor.check();raw['measurement_window']={}
        window(runtime,config,samples,'measurement',duration,raw['request_records'],monitor,
            make_dispatcher(policy_contract),raw['measurement_window'])
        after=monitor.sample();monitor.check()
        raw['status']='completed' if all(r['status']=='completed' for r in raw['request_records']) else 'completed_with_failures'
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='failed';raw['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if raw['measurement_window'] and 'drain_end_time_ns' not in raw['measurement_window']:
            raw['measurement_window'].update(aborted=True,drain_end_time_ns=time.perf_counter_ns())
        if monitor:monitor.close()
        if power:power.close()
        try:raw['shutdown']=runtime.close()
        except Exception as exc:
            raw['shutdown']={};raw['errors'].append('shutdown: '+str(exc));raw['status']='failed'
        raw['errors'].extend(runtime.errors)
        if runtime.errors or any(v['forced'] or v['exit_code']!=0 or not v['bye_received'] for v in raw['shutdown'].values()):raw['status']='failed'
        for key in ['request_records','warmup_request_records','priming_request_records']:
            for row in raw[key]:
                if row['status'] in {'planned','queued','submitted'}:row.update(status='incomplete',error='run ended before response')
            original.annotate_records(raw[key])
    measured=before is not None and after is not None and monitor is not None and not monitor.failed
    nominal=system.catalog['safe_factor']*sum(system.catalog['memory_table_mib'][m][str(n)] for m,n in winner['deployment'].items())
    memory_samples=monitor.samples if monitor else []
    safe=system.catalog['safe_factor']*max(s['owned_used_mib'] for s in [before,after,*memory_samples]) if measured else nominal
    raw['memory_evidence']=dict(measured=measured,safe_memory_mib=safe,resource_valid=measured and safe<=system.gpu_memory_mib
        and all(not s['foreign_pids'] and s['owned_used_mib']<=system.gpu_memory_mib for s in memory_samples),
        scope='owned-process idle memory and sampled peak' if measured else 'catalog planning surrogate; diagnostic only',
        samples=memory_samples,before=before,after=after)
    w=raw['measurement_window'];power_samples=power.samples if power else []
    run_j=original.integrate_power(power_samples,w['start_time_ns'],w['end_time_ns']) if w else None
    once_j=original.integrate_power(power_samples,startup_begin,w['start_time_ns']) if w and startup_begin is not None else None
    raw['energy_evidence']=dict(kind='measured' if backend=='native' else 'unmeasured',scope='whole_gpu_board',
        coverage_valid=backend=='native' and power is not None and not power.failed and run_j is not None and once_j is not None,
        run_joules=run_j,startup_joules=once_j,samples=power_samples,sampling_errors=power.errors if power else [],
        startup_begin_time_ns=startup_begin,scope_note='total GPU board energy including load/prime/warmup; excludes CPU and bounded post-window drain')
    if w:
        raw['per_task_metrics'],raw['system_metrics'],raw['model_metrics']=original.summarize(raw['request_records'],config,w)
    try:
        measurement_id=dynamic_admission.measurement_identity(system,raw)
        raw['measurement_identity']=measurement_id
        raw['memory_evidence']['measurement_identity']=measurement_id
        raw['energy_evidence'].update(measurement_identity=measurement_id,profile_key=measurement_id)
        if evaluator is None:
            from tools.queue_refined_20261023.evaluation import evaluate_run
            evaluator=evaluate_run
        raw['evaluation']=evaluator(system,raw)
    except Exception as exc:
        raw['errors'].append('evaluation: '+str(exc));raw['status']='failed'
        raw['evaluation']=dict(utility=dict(U=None,eligible=False,reason='DYNAMIC_EVALUATION_FAILED'),stability={})
    return raw


def execute_job(packet,job_id,output,*,backend='native',execute=False):
    check=validate_packet(packet)
    if not check['valid']:raise ValueError('Packet source changed: '+repr(check['changed']))
    if not execute:raise ValueError('Explicit execute=True required to start workers')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    job=next(j for j in packet['jobs'] if j['id']==job_id);base_receipt=job.get('base_packet',packet['base_packet']);base=json.loads(Path(base_receipt['path']).read_text())
    original=frozen_runtime();package=original.__package__.split('.runtime')[0]
    static=importlib.import_module(package+'.static_study');configuration=importlib.import_module(package+'.config')
    output.mkdir(parents=True,exist_ok=False)
    try:
        if backend=='native':
            binding=copy.deepcopy(base['jobs'][0]);binding['plan']=job['nominal_plan'];binding['seed']=job['seed']
            binding['identity'].update(seed=job['seed'],horizon_seconds=job['identity']['horizon_seconds'],warmup_seconds=job['identity']['warmup_seconds'])
            config,admission_path,workers=static.bind_native_job(base,binding,output/'native-binding')
            config['catalog']['performance_mode']='task_replica_proxy';config['catalog']['memory_mode']='standalone_proxy'
        else:config=copy.deepcopy(base['config']);admission_path=None;workers={}
        config['random_seed']=job['seed'];config['horizon_seconds']=job['identity']['horizon_seconds']
        raw=run(configuration.SystemConfig(**config),job['nominal_plan'],Path(base['inputs'])/'runtime-inputs.jsonl',job['policy_contract'],
            backend=backend,workers=workers,duration=job['identity']['horizon_seconds'],warmup=job['identity']['warmup_seconds'],seed=job['seed'],
            arrival_process=job['identity']['arrival_process'],drain_timeout=config['drain_seconds'],admission=str(admission_path) if admission_path else None,
            capacity_evidence=job.get('capacity_evidence'))
    except (Exception,KeyboardInterrupt) as exc:
        # Binding/pre-start failures have no fabricated measured resources or
        # request results. Preserve their own terminal before supervisor STOP.
        raw=dict(schema_version=1,backend=backend,status='failed',errors=[f'{type(exc).__name__}: {exc}'],
            failure_stage='NATIVE_BINDING_OR_RUN_INITIALIZATION',plan=job['nominal_plan'],policy_contract=job['policy_contract'],
            strategy=job['strategy'],request_records=[],warmup_request_records=[],priming_request_records=[],
            measurement_window=None,memory_evidence={},energy_evidence={},shutdown={},
            evaluation=dict(utility=dict(U=None,eligible=False,reason='FAILED_BEFORE_COMPLETE_RUN')),
            measurement_claim=job['identity'].get('measurement_claim','failed development collection'))
    raw.update(experiment_job_id=job_id,experiment_phase=job.get('phase','calibration'),executed_measurement_identity=job['identity'],
        executed_measurement_id=digest(job['identity']),source_packet=base_receipt)
    raw['measurement_claim']=job['identity'].get('measurement_claim',raw['measurement_claim'])
    write(output/'run.json',raw);return raw


run_job=execute_job


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--job')
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--backend',choices=['native','mock'],default='native');a=p.parse_args()
    packet=json.loads(a.packet.read_text())
    if not a.execute:print(json.dumps(validate_packet(packet)))
    else:
        if not a.job or a.output is None:p.error('--job and --output required for execution')
        result=execute_job(packet,a.job,a.output,backend=a.backend,execute=True)
        print(json.dumps(dict(status=result['status'],raw=receipt(a.output/'run.json'))))
        raise SystemExit(0 if result['status'] in {'completed','completed_with_failures'} else 1)
