#!/usr/bin/env python3
"""Prepare five current-device native baseline contexts; CPU only, never execute."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
ORIGINAL=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
REQUIREMENTS=ROOT/'docs/seven_pool_recalibration_requirements_20261022.json'
AUDIT=ROOT/'docs/seven_pool_missing_state_evidence_audit_20261022.json'
DESTINATION=ROOT/'artifacts/static-spillover-study-20261022/seven-pool-missing-baseline-preparation-v1'
PROTOCOL='missing_standalone_full_resident_baseline_v1'
CONTEXTS=[('generalist_1',3,['text']),('generalist_1',4,['text','image','ocr']),
          ('generalist_2',4,['text']),('generalist_2',5,['text','image','ocr']),('generalist_3',4,['ocr'])]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def sequence(ids,namespace,repeats):
    result=[]
    for repetition in range(repeats):
        order=list(ids);random.Random(f'{namespace}:{repetition}').shuffle(order);result.extend(order)
    return result


def import_api():
    sys.path.insert(0,str(SOURCE))
    from gsdeploy.static_study import _bindings,bind_native_job
    from gsdeploy.runtime.workload import SamplePool,largest_inputs
    if not Path(sys.modules['gsdeploy.static_study'].__file__).resolve().is_relative_to(SOURCE):raise ValueError('Fresh frozen source import required')
    return _bindings,bind_native_job,SamplePool,largest_inputs


def make_jobs(config,samples):
    _,_,SamplePool,largest_inputs=import_api()
    rows=[json.loads(line) for line in Path(samples).read_text().splitlines() if line.strip()]
    pools={t:[r['pool_id'] for r in rows if r['task_type']==t] for t in ['text','image','ocr','asr']}
    if {t:len(v) for t,v in pools.items()}!={'text':32,'image':64,'ocr':32,'asr':32}:raise ValueError('Original input pools differ')
    pool=SamplePool(samples,list(pools),0);jobs=[]
    for family,n,tasks in CONTEXTS:
        context_id=f'{family}-n{n}';aliases={f'{family}#{i}':family for i in range(n)}
        priming=[]
        for alias in aliases:
            for task in config['catalog']['models'][family]['supported_tasks']:
                for item in largest_inputs(pool,task,3):
                    i=len(priming);priming.append(dict(request_id=f'priming:{i}',model=alias,task=task,pool_id=item['pool_id'],arrival_index=i))
        phases=[]
        for task in tasks:
            ids=sequence(pools[task],f'{PROTOCOL}:{context_id}:primary:{task}',3)
            primary={a:[] for a in aliases};support={a:[] for a in aliases};warmup={a:[] for a in aliases}
            for i,pid in enumerate(ids):
                # Global pool unit preserved, with predeclared balanced mapping.
                alias=list(aliases)[i%n]
                primary[alias].append(dict(request_id=f'measurement:{task}:{i}',task=task,model=alias,pool_id=pid,
                    role='primary',arrival_index=i,sequence=i,repetition=i//len(pools[task]),alias_sequence=len(primary[alias])))
            for alias in aliases:
                warmup[alias]=[dict(task=task,model=alias,pool_id=pid) for pid in sequence(
                    pools[task],f'{PROTOCOL}:{context_id}:warmup:{task}:{alias}',1)]
                values=sequence(pools[task],f'{PROTOCOL}:{context_id}:support:{task}:{alias}',math.ceil(256/len(pools[task])))[:256]
                support[alias]=[dict(request_id=f'support:{task}:{alias}:{i}',task=task,model=alias,pool_id=pid,role='support',
                    arrival_index=len(ids)+list(aliases).index(alias)*256+i,support_sequence=i) for i,pid in enumerate(values)]
            historical=config['catalog']['task_replica_profiles'][family][task]
            nearest=min((int(k) for k in historical),key=lambda k:abs(n-k));base=historical[str(nearest)]
            seconds=3600 if task=='ocr' else 600
            phase=dict(task=task,primary_requests=primary,support_requests=support,warmup_requests=warmup,input_pool_ids=pools[task],repetitions=3,
                global_primary_count=len(ids),planned_primary_per_alias={a:len(v) for a,v in primary.items()},
                planned_unique_primary_inputs_per_alias={a:len({r['pool_id'] for r in v}) for a,v in primary.items()},
                measurement_timeout_seconds=seconds,warmup_seconds=30,warmup_timeout_seconds=330,support_limit_per_alias=256,
                risk_proxy=dict(nearest_measured_n=nearest,mean_service_ms=base['mean_service_ms'],source=base['source'],
                    historical_ideal_work_seconds=len(ids)*base['mean_service_ms']/1000/n,
                    label='Historical different-state work estimate only; not service/capacity bound or expected runtime guarantee'))
            phase['phase_identity_sha256']=digest(phase);phases.append(phase)
        proxy=config['catalog']['memory_table_mib'][family][str(n)];factor=config['catalog']['safe_factor']
        job=dict(context_id=context_id,model=family,replicas=n,resident_deployment={family:n},aliases=aliases,
            full_resident_supported_tasks=config['catalog']['models'][family]['supported_tasks'],priming_requests=priming,phases=phases,
            primary_counts_by_task={p['task']:p['global_primary_count'] for p in phases},
            static_residency_proxy=dict(raw_mib=proxy,safe_mib=proxy*factor,budget_mib=config['gpu_memory_mib'],
                is_lower_bound=False,is_feasibility_certificate=False,exclude_state=False,
                scope='Original memory table proxy, all task lazy paths not guaranteed resident; actual peak must be measured.'),
            timeouts=dict(startup_per_alias_seconds=300,priming_request_seconds=300,priming_hard_wall_seconds=3600,
                native_request_seconds=300,graceful_shutdown_seconds=10,terminate_seconds=3,kill_seconds=3,
                context_hard_wall_seconds=300*n+3600+sum(p['warmup_timeout_seconds']+p['measurement_timeout_seconds'] for p in phases)+120),
            failure_policy='Stop this context with raw failure retained; later tasks NOT_STARTED_UNKNOWN; no automatic retry/next-context or cross-state exclusion.',
            applicability='Standalone same family, full four-task lazy capability resident, target task active with bounded same-task support; different from historical single-task priming and formal low-rate SWRR.',
            exact_failure_scope='A failure diagnoses this exact current-GPU startup/priming/activity/request contract; stronger idle-task priming failure does not exclude the original single-task-active domain.',
            U=None)
        job['context_identity_sha256']=digest(job);jobs.append(job)
    return jobs,pools


def prepare(destination=DESTINATION):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('New immutable preparation directory required')
    original=json.loads(ORIGINAL.read_text());requirements=json.loads(REQUIREMENTS.read_text());audit=json.loads(AUDIT.read_text())
    if audit['counts']['missing_states']!=9:raise ValueError('Expected nine unresolved states')
    required={(r['model'],r['task'],r['replicas']) for r in requirements['missing_path_states']}
    if required!={(m,t,n) for m,n,tasks in CONTEXTS for t in tasks}:raise ValueError('Context plan must cover exactly nine original missing states')
    config=copy.deepcopy(original['config']);config['catalog']['measurement_context']={
        **config['catalog'].get('measurement_context',{}),'gpu_uuid':'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    # CPU identity binding over three families only; deployment is not executed,
    # and no simultaneous three-family feasibility claim is made.
    _,binder,_,_=import_api();destination.mkdir(parents=True,exist_ok=False)
    binding_packet={**original,'config':config};identity_request=dict(plan={'deployment':{m:1 for m in ['generalist_1','generalist_2','generalist_3']}},seed=0,identity={'horizon_seconds':600})
    native_config,admission_path,workers=binder(binding_packet,identity_request,destination/'native-binding')
    samples=Path(original['inputs'])/'runtime-inputs.jsonl'
    write(destination/'system-config.json',native_config);write(destination/'workers.json',workers)
    write(destination/'admission.json',json.loads(admission_path.read_text()))
    jobs,pools=make_jobs(native_config,samples)
    declaration=dict(schema='gsdeploy.missing_state_baseline_declaration.v1',status='PREPARED_CPU_ONLY_NOT_EXECUTED',
        scope='Five current-device development contexts for nine missing native standalone path/n states; does not change original seven-pool search domain or activate a performance model.',
        global_primary_unit='Complete fixed task pool, each input exactly3 primary observations globally across n aliases; not3 per input per alias.',
        primary_assignment='One independently shuffled full pool per repetition; global index modulo n assigns alias; alias coverage explicitly disclosed.',
        support='After own primary ends, same-task predeclared independent support cursor until all primary complete; at most256 per alias per task, one outstanding per alias; all support retained separately.',
        full_priming='All n workers READY first, then largest3 by frozen size proxy for text/image/ocr/asr on every resident alias, serial globally; stronger than historical single-task profiling.',
        phase_order='Declared tasks, fresh30s target-task warmup fully drained before each measurement; cursor independent; no fresh worker reload between tasks.',
        support_failure='Support cap exhausted, timeout, error or unknown response invalidates that phase; no tail-only fallback or drop of failed rows.',
        memory='Original safe_factor/49140MiB; process-owned peak with foreign-process checks from before READY through final failure sample; raw samples retained; no proxy/lower-bound exclusions.',
        quality='No absolute quality admission; full native predictions retained for later frozen scoring; no historical quality overwrite, no formal U.',
        formal_constraints='Formal rho<=.8/SWRR/score/stability unchanged. No formal validate_plan on these unknown-service diagnostic contexts; formal feasibility remains UNKNOWN.',
        independent_validation=False,old_transfer_active=False,execution='Explicit single-context --execute only after preparation review; this preparation never executes; no automatic full queue.',
        current_gpu_infeasibility_evidence='None supplied for these9path/n before execution. Historical failure UUID533a... and plateau heuristics cannot exclude currentUUID8a... states.',
        no_cross_context_pruning=True,no_auto_retry=True,no_auto_next_context=True,prepared_only=True,GPU_started=False,U=None)
    write(destination/'discovery-protocol.json',declaration)
    write(destination/'cpu-risk-check.json',dict(contexts=[dict(context_id=j['context_id'],static_residency_proxy=j['static_residency_proxy'],
        priming_count=len(j['priming_requests']),priming_counts_by_task=dict(Counter(r['task'] for r in j['priming_requests'])),
        phases=[{k:p[k] for k in ['task','global_primary_count','planned_primary_per_alias','planned_unique_primary_inputs_per_alias','measurement_timeout_seconds','risk_proxy']} for p in j['phases']],
        timeouts=j['timeouts'],no_exclusion=True) for j in jobs],U=None,GPU_started=False))
    bindings={b['path']:b['sha256'] for b in import_api()[0](Path(original['inputs']),SOURCE)}
    for p in [ORIGINAL,REQUIREMENTS,AUDIT,Path(__file__).resolve(),Path(__file__).with_name('check.py').resolve(),Path(__file__).with_name('run.py').resolve(),
        ROOT/'docs/seven_pool_missing_baseline_preparation_20261022.md',ROOT/'tests/test_seven_pool_missing_baseline.py',
        *destination.rglob('*.json')]:bindings[str(p.resolve())]=sha(p)
    packet=dict(schema='gsdeploy.missing_state_baseline_packet.v1',protocol=PROTOCOL,status='PREPARED_NOT_EXECUTED',diagnostic_only=True,U=None,
        frozen_source=str(SOURCE),system_config=str(destination/'system-config.json'),workers=str(destination/'workers.json'),admission=str(destination/'admission.json'),
        samples=str(samples),gpu_uuid=native_config['catalog']['measurement_context']['gpu_uuid'],device='0',gpu_memory_mib=native_config['gpu_memory_mib'],
        safe_factor=native_config['catalog']['safe_factor'],pools=pools,jobs=jobs,bindings=bindings,bindings_sha256=digest(bindings),
        completion_claim=False,current_device_verified=False,execution_authorization='Preparation only; this session did not invoke--execute.',GPU_started=False)
    write(destination/'experiment-packet.json',packet)
    # A separate checker is deliberately also runnable without subprocesses/GPU.
    spec=importlib.util.spec_from_file_location('_missing_baseline_check',Path(__file__).with_name('check.py'));checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    result=checker.validate_packet(packet)
    write(destination/'preparation-receipt.json',dict(packet_sha256=sha(destination/'experiment-packet.json'),validation=result,
        checkpoint_bindings={m:native_config['catalog']['measurement_context']['runtime_checkpoint_sha256'][m] for m in ['generalist_1','generalist_2','generalist_3']},
        original_domain_preserved=True,production_math_modified=False,GPU_started=False,U=None))
    return dict(packet=str(destination/'experiment-packet.json'),sha256=sha(destination/'experiment-packet.json'),validation=result,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DESTINATION);args=parser.parse_args();print(json.dumps(prepare(args.output)))
