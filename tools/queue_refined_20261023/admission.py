"""Identity and prior route envelopes for opt-in dynamic native measurements.

Quality is scored, never thresholded. Rho uses exactly the original fixed
task/replica proxy and .8 tolerance; token-bucket exposure is a prior bound,
not an observed routing share or a physical stability guarantee.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from tools.spillover_capacity_20261022.exposure import frozen_api
_API=frozen_api();_PACKAGE=_API['trace'].__package__
_evidence=importlib.import_module(_PACKAGE+'.evidence')
digest=_evidence.digest;verify_quality_evidence=_evidence.verify_quality_evidence;quality_covers=_evidence.quality_covers
resolve_service=importlib.import_module(_PACKAGE+'.performance').resolve_service
SamplePool=_API['workload'].SamplePool

SCHEMA='gsdeploy.queue_aware.admission.v1'

def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def policy_identity(contract):
    return canonical_digest(dict(namespace='gsdeploy.queue_aware.policy.v1',contract=contract))

def measurement_identity(config,raw):
    """Bind measured resources to the exact dynamic policy and dispatch log."""
    w=raw['measurement_window'];transport=raw['transport_config']
    return canonical_digest(dict(namespace='gsdeploy.queue_aware.measurement.v1',
        policy_identity=policy_identity(raw['policy_contract']),strategy=raw['strategy'],
        deployment=raw['plan']['deployment'],workload=config.workload,
        seconds=w['duration_seconds'],seed=transport['random_seed'],arrival_process=transport['arrival_process'],
        trace_sha256=w['trace_sha256'],dispatch_decisions_sha256=w['dispatch_decisions_sha256'],
        start_time_ns=w['start_time_ns'],end_time_ns=w['end_time_ns'],original_system_config=config.as_dict()))

def _finite(value,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)

def source_checks(contract):
    sources=contract.get('source_fingerprints',{})
    if not isinstance(sources,dict) or not sources:return False,['MISSING_POLICY_SOURCE_FINGERPRINTS']
    issues=[]
    required=[Path(__file__).with_name(n) for n in ['admission.py','evaluation.py','policy.py','protocol.py','runtime_window.py','native_runner.py']]
    if contract.get('guard') and contract.get('strategy') in {'queue_latency','queue_quality'}:required.append(Path(__file__).with_name('guarded_policy.py'))
    if contract.get('refinement') is not None:
        required += [Path(__file__).with_name(n) for n in ['calibration.py','refined_policy.py']]
        try:
            model=contract['refinement']['model']
            if digest(model['path'])!=model['sha256']:issues.append('CALIBRATION_MODEL_RECEIPT_MISMATCH')
        except (OSError,KeyError,TypeError):issues.append('CALIBRATION_MODEL_UNAVAILABLE')
    source_root=Path(_API['trace'].__file__).resolve().parent
    required += [source_root/p for p in ['evidence.py','performance.py','quality.py','prediction.py','stability.py','metrics.py','energy.py','trace.py','routing.py','runtime/workload.py','runtime/deployment.py','runtime/runner.py','runtime/telemetry.py']]
    if not {str(p.resolve()) for p in required}.issubset({str(Path(p).resolve()) for p in sources}):issues.append('MISSING_REQUIRED_RUNTIME_OR_EVALUATION_SOURCE_FINGERPRINTS')
    for path,expected in sources.items():
        try:
            if not isinstance(expected,str) or digest(path)!=expected:issues.append('SOURCE_FINGERPRINT_MISMATCH:'+str(path))
        except (OSError,TypeError):issues.append('SOURCE_UNAVAILABLE:'+str(path))
    return not issues,issues

def potential_paths(config,winner,contract):
    """Return declared executable candidate paths, not paths observed afterward."""
    aliases=contract['aliases'];deployment=winner['deployment'];expected={f'{m}#{i}':m for m,n in deployment.items() for i in range(n)}
    if {a:s['family'] for a,s in aliases.items()}!=expected:raise ValueError('Policy aliases differ from resident deployment')
    paths=set();strategy=contract['strategy'];background=contract.get('background_routing',{})
    for task,rate in config.workload.items():
        if task=='text':
            if strategy in {'no_split_swrr','queue_specialist_only'}:families={'text'}
            elif strategy in {'fixed_1of32_swrr','queue_latency','queue_quality'}:families={'text','generalist_2'}
            else:raise ValueError('Unknown declared strategy')
        else:
            row=background.get(task,{})
            if not row or not math.isclose(sum(row.values()),rate,abs_tol=1e-10):raise ValueError('Background route does not conserve original task load')
            families={m for m,q in row.items() if q>0}
        for m in families:
            if m not in deployment or task not in config.catalog['models'][m]['supported_tasks']:raise ValueError('Potential path not executable')
            if not any(s['family']==m and task in s['supported_tasks'] for s in aliases.values()):raise ValueError('Policy alias capability missing')
            paths.add((m,task))
    return paths

def cover_plan(config,winner,contract):
    paths=potential_paths(config,winner,contract)
    return dict(deployment=dict(winner['deployment']),routing={t:{m:config.workload[t]/sum(tt==t for _,tt in paths) for m,tt in sorted(paths) if tt==t} for t in config.workload})

def capacity_envelope(config,winner,contract):
    """Conservative prior family-rate envelope under the original proxy gate.

    Caps for different families may sum above offered load. The dynamic G2
    bound requires the separately audited causal guard, never post-hoc shares.
    Burst1 remains explicit; rho is the original long-run rate proxy only.
    """
    try:
        paths=potential_paths(config,winner,contract);strategy=contract['strategy'];caps={m:{} for m in winner['deployment']}
        for m,t in sorted(paths):
            rate=config.workload[t]
            if t=='text' and m=='generalist_2':
                if strategy=='fixed_1of32_swrr':rate/=32
                else:
                    guard=contract.get('guard') or {}
                    if (not isinstance(guard,dict) or guard.get('schema')!='queue_aware_token_bucket_v1' or guard.get('task')!='text'
                        or guard.get('family')!='generalist_2' or not _finite(guard.get('rate_rps'),True)
                        or not _finite(guard.get('burst'),True) or guard['burst']!=1 or guard['rate_rps']>rate):
                        raise ValueError('MISSING_PRIOR_DYNAMIC_PATH_GUARD')
                    rate=guard['rate_rps']
            elif t=='text' and strategy=='fixed_1of32_swrr':rate*=31/32
            elif t!='text':rate=contract['background_routing'][t][m]
            caps[m][t]=rate
        mode=config.catalog.get('performance_mode','independent')
        if mode not in {'independent','task_replica_proxy'}:raise ValueError('NO_PROVED_ROUTE_INDEPENDENT_ORIGINAL_SERVICE_ENVELOPE')
        probe={t:{m:caps[m][t] for m,tt in sorted(paths) if tt==t} for t in config.workload}
        service,evidence=resolve_service(config.catalog,winner['deployment'],probe,config.workload,
            config.horizon_seconds,config.random_seed,config.arrival_process)
        if service is None:raise ValueError('ORIGINAL_PATH_SERVICE_UNKNOWN')
        rho={m:sum(q*service[m][t]/1000 for t,q in row.items())/config.catalog['capacity_factors'][m][str(winner['deployment'][m])] for m,row in caps.items()}
        memory_mode=config.catalog.get('memory_mode','standalone_proxy')
        if memory_mode not in {'standalone_proxy','independent'}:raise ValueError('NO_ORIGINAL_ADDITIVE_MEMORY_ENVELOPE')
        memory=config.catalog['safe_factor']*sum(config.catalog['memory_table_mib'][m][str(n)] for m,n in winner['deployment'].items())
        passed=max(rho.values(),default=0)<=.8+1e-8 and memory<=config.gpu_memory_mib+1e-6
        return dict(schema=SCHEMA,status='VERIFIED_ORIGINAL_PROXY_ENVELOPE' if passed else 'ORIGINAL_PROXY_ENVELOPE_EXCEEDS_LIMIT',
            constraints_verified=passed,capacity_verified=passed,memory_verified=True,policy_identity=policy_identity(contract),
            rate_caps_by_family_task=caps,rho_by_family=rho,max_rho=max(rho.values(),default=0),rho_threshold=.8,rho_tolerance=1e-8,
            safe_memory_mib=memory,service_evidence=evidence,guard_required=strategy in {'queue_latency','queue_quality'},
            guard=contract.get('guard'),observed_routing_share_used=False,physical_stability_certified=False,
            burst_scope='Token-bucket burst is explicit and is not substituted for the original long-run rho gate.')
    except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
        return dict(schema=SCHEMA,status='UNKNOWN',constraints_verified=False,capacity_verified=False,memory_verified=False,
            reason=str(exc),rho_by_family=None,observed_routing_share_used=False,physical_stability_certified=False)

def quality_identity_covers(config,raw):
    try:
        plan=cover_plan(config,raw['plan'],raw['policy_contract']);proxy=dict(raw,plan=plan)
        return quality_covers(config,proxy)
    except (KeyError,TypeError,ValueError):return False

def preflight(config,winner,samples,registry,policy_contract,admission=None,require_gpu=False,
              capacity_evidence=None,allow_unverified_capacity_measurement=False):
    issues=[];sources_ok,source_issues=source_checks(policy_contract);issues+=source_issues
    try:
        plan=cover_plan(config,winner,policy_contract)
        if policy_contract.get('deployment',winner['deployment'])!=winner['deployment']:raise ValueError('Policy deployment mismatch')
        for m in winner['deployment']:
            worker=registry[m];command=worker.get('command',[]);exe=shutil.which(command[0]) if command else None
            if not exe or not os.access(exe,os.X_OK):issues.append('WORKER_INTERPRETER_UNAVAILABLE:'+m)
            cwd=Path(worker.get('cwd',Path.cwd()))
            for key in ['GS_MODEL_PATH','GS_PROMPT_FILE']:
                p=Path(worker.get('env',{}).get(key,''));p=p if p.is_absolute() else cwd/p
                if not worker.get('env',{}).get(key) or not p.exists():issues.append('WORKER_FILE_UNAVAILABLE:'+m+':'+key)
            if not Path(worker.get('env',{}).get('GS_MODEL_PATH','')).is_absolute():issues.append('WORKER_CHECKPOINT_NOT_ABSOLUTE:'+m)
        pool=SamplePool(samples,list(config.workload),config.random_seed)
        for t,rows in pool.samples.items():
            if t!='text':
                for row in rows:
                    if not Path(row['sample'].get('input','')).is_file():issues.append('SAMPLE_ASSET_UNAVAILABLE:'+row['pool_id'])
        if digest(samples)!=config.catalog['quality']['identity']['input_pool_sha256']:issues.append('INPUT_POOL_FINGERPRINT_MISMATCH')
        if not config.catalog['quality'].get('references'):issues.append('QUALITY_REFERENCES_UNAVAILABLE')
        quality=verify_quality_evidence(config,plan,registry,admission)
        if not quality.get('verified'):issues.append('QUALITY_IDENTITY_UNVERIFIED')
    except (KeyError,TypeError,ValueError,OSError) as exc:
        quality=dict(verified=False,reason=str(exc),pairs=[]);issues.append('INPUT_OR_IDENTITY_CHECK_FAILED:'+str(exc))
    capacity=capacity_envelope(config,winner,policy_contract)
    if capacity_evidence is not None and capacity_evidence!=capacity:issues.append('SUPPLIED_CAPACITY_EVIDENCE_MISMATCH')
    if not capacity['constraints_verified'] and not allow_unverified_capacity_measurement:issues.append('PRIOR_CAPACITY_OR_MEMORY_UNVERIFIED')
    devices={registry.get(m,{}).get('env',{}).get('CUDA_VISIBLE_DEVICES','0') for m in winner['deployment']}
    if len(devices)!=1 or any(not isinstance(d,str) or not d or ',' in d or d=='-1' for d in devices):issues.append('WORKERS_REQUIRE_SAME_SINGLE_GPU')
    device=next(iter(devices)) if len(devices)==1 else '0';gpu=None
    if require_gpu and not issues:
        try:
            result=subprocess.run(['nvidia-smi','-i',device,'--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
            rows=[r.split(',') for r in result.stdout.splitlines() if r.strip()]
            if len(rows)!=1 or len(rows[0])!=4:raise ValueError('Expected one GPU')
            uuid,name,driver,memory=(v.strip() for v in rows[0]);gpu=dict(uuid=uuid,name=name,driver=driver,memory_mib=float(memory))
            if gpu['memory_mib']<config.gpu_memory_mib:issues.append('GPU_MEMORY_BELOW_CONFIGURED_BUDGET')
            active=subprocess.run(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
            if active.stdout.strip():issues.append('GPU_HAS_EXISTING_COMPUTE_PROCESSES')
        except (OSError,ValueError,subprocess.SubprocessError) as exc:issues.append('GPU_UNAVAILABLE:'+str(exc))
    return dict(ready=not issues,issues=issues,scope='Prior original proxy envelope and native identity checks; original runtime stability remains independently evaluated.',
        admission_evidence=quality,capacity_evidence=capacity,policy_identity=policy_identity(policy_contract),
        sample_pool_sha256=digest(samples),device=device,gpu=gpu,gpu_checked=require_gpu,
        development_measurement_only=not capacity['constraints_verified'],absolute_quality_gate=False)
