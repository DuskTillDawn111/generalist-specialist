"""Fresh CPU forecast workers after a v5 protocol freeze; never collect GPU data."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .known_input_gate_v5 import _verify_protocol
from .learned_availability_forecast_v16 import LearnedAvailabilityForecast
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .prospective_flow_contract_v11 import bind_public,original_flow_defaults

SPEC_SCHEMA='gsdeploy.frozen_fresh_CPU_campaign_worker.v16'
WORKER_SCHEMA='gsdeploy.fresh_CPU_campaign_worker_receipt.v16'
COMPLETE='COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST'
MAX_EVENTS=5_000_000


def run_worker(spec_reference,protocol_reference):
    """No preflight results enter this function; construct one actual provider."""
    spec=checked(spec_reference);protocol=checked(protocol_reference)
    required={'schema','protocol_path','plan','configuration','candidate_id','public_inputs','routes',
              'training_index','output_directory','max_events'}
    if (set(spec)!=required or spec['schema']!=SPEC_SCHEMA or spec['max_events']!=MAX_EVENTS
            or spec['protocol_path']!=protocol_reference['path']):
        raise ValueError('Exact frozen fresh-worker specification required')
    out=Path(spec['output_directory']);out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('forecast.json','worker-receipt.json')):
        raise ValueError('Fresh worker output files must not already exist')
    began=time.monotonic();result_ref=None;before_verified=False;after_verified=False;after_attempted=False;raw_audit=[]
    try:
        _verify_protocol(protocol);before_verified=True
        if (spec_reference not in protocol['sources']
                or spec['configuration'] not in protocol['models'].values()
                or spec['training_index'] not in protocol['training_data']
                or any(spec[k] not in protocol['sources'] for k in ('plan','public_inputs','routes'))):
            raise ValueError('Worker inputs and specification must be frozen in this protocol')
        candidate=next((c for c in protocol['candidates'] if c['id']==spec['candidate_id']),None)
        if candidate is None or protocol['models'][candidate['model_id']]!=spec['configuration']:
            raise ValueError('Frozen protocol candidate and exact model required')
        started_unix_ns=time.time_ns()
        if started_unix_ns<=protocol['created_unix_ns']:
            raise ValueError('Provider construction must start after protocol freeze')
        configuration=checked(spec['configuration']);plan=checked(spec['plan'])
        public=checked(spec['public_inputs'])['forecast_inputs'];routes=checked(spec['routes'])
        training=checked(spec['training_index']);indexed={r['run_id'].removeprefix('raw-sha256:') for r in training['records']}
        model=LearnedAvailabilityForecast(configuration,allow_unvalidated_training_hypothesis=True)
        raw_audit=model.availability_training_audit
        if not raw_audit or any(a['raw']['sha256'] not in indexed for a in raw_audit):
            raise ValueError('Actual availability training raw omitted from complete frozen index')
        defaults=original_flow_defaults(plan,model.bundle['forecast_defaults'])
        rows=bind_public(plan,spec['candidate_id'],public,routes,defaults,plan['weights'])
        result=model.execute(spec['plan'],spec['candidate_id'],public,routes,
            model.bundle['startup']['training_model'],max_events=MAX_EVENTS)
        save(out/'forecast.json',result);result_ref=receipt(out/'forecast.json')
        diagnostic=result.get('static_provider_compilation',{})
        if (result.get('status')!=COMPLETE or result.get('full_public_cohort')!=rows
                or result.get('CPU_representation_version')!=16
                or diagnostic.get('nodes_or_edges_dropped') is not False
                or diagnostic.get('fitted_values_changed') is not False
                or result.get('configuration')!=configuration or result.get('plan')!=spec['plan']
                or result.get('candidate_id')!=spec['candidate_id']):
            raise ValueError('Fresh v16 worker must finish the complete frozen cohort')
        after_attempted=True;_verify_protocol(protocol);after_verified=True
        status=COMPLETE;error=None
    except Exception as exc:
        traceback.print_exc();status='WORKER_FAILED_PROHIBITS_SEAL';error=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        if before_verified and not after_attempted:
            try:_verify_protocol(protocol);after_verified=True
            except Exception as exc:
                status='WORKER_FAILED_PROHIBITS_SEAL'
                error=dict(original_error=error,type=type(exc).__name__,reason=str(exc),stage='POST_WORKER_SOURCE_VERIFICATION')
    summary=dict(schema=WORKER_SCHEMA,candidate_id=spec['candidate_id'],status=status,error=error,
        spec=spec_reference,protocol=protocol_reference,result=result_ref,pid=os.getpid(),
        seconds=time.monotonic()-began,frozen_protocol_verified_before=before_verified,
        frozen_protocol_verified_after=after_verified,availability_training_audit=raw_audit,
        actual_availability_training_raw_all_in_frozen_index=status==COMPLETE,
        provider_constructed_in_worker_after_freeze=status==COMPLETE,
        preflight_cells_reused=False,GPU_started=False)
    save(out/'worker-receipt.json',summary)
    return summary


def _terminate(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)


def run_parallel(spec_references,protocol_reference,*,timeout_seconds=7200.,_command_factory=None):
    """Spawn every fresh worker before waiting; fail closed and reap all workers."""
    if (len(spec_references)!=4 or type(timeout_seconds) not in (float,int)
            or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
        raise ValueError('Exactly four workers and finite positive CPU timeout required')
    specs=[checked(ref) for ref in spec_references]
    if len({s['candidate_id'] for s in specs})!=4:raise ValueError('Four distinct candidate workers required')
    protocol=checked(protocol_reference);_verify_protocol(protocol)
    processes=[];handles=[];started=time.monotonic();failure=None;cleanup_errors=[]
    try:
        for spec,reference in zip(specs,spec_references):
            if reference not in protocol['sources']:raise ValueError('All worker specifications must be frozen')
            out=Path(spec['output_directory']);out.mkdir(parents=True,exist_ok=True)
            handle=(out/'worker.log').open('x');handles.append(handle)
            command=([sys.executable,'-u','-m',__name__,'--spec',reference['path'],
                '--spec-sha256',reference['sha256'],'--protocol',protocol_reference['path'],
                '--protocol-sha256',protocol_reference['sha256']] if _command_factory is None
                else _command_factory(reference,protocol_reference))
            processes.append(subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Fresh CPU worker exited unsuccessfully')
            if time.monotonic()-started>timeout_seconds:raise TimeoutError('Fresh CPU workers exceeded declared timeout')
            time.sleep(.05)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Fresh CPU worker exited unsuccessfully')
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        for process in processes:
            try:_terminate(process)
            except BaseException as exc:
                cleanup_errors.append(dict(pid=process.pid,type=type(exc).__name__,reason=str(exc)))
        for handle in handles:
            try:handle.close()
            except BaseException as exc:
                cleanup_errors.append(dict(type=type(exc).__name__,reason=str(exc)))
        if cleanup_errors:
            failure=dict(type='WorkerCleanupFailure',reason='Every cleanup was attempted',
                original_error=failure,errors=cleanup_errors)
    workers=[]
    for spec,reference,process in zip(specs,spec_references,processes):
        path=Path(spec['output_directory'])/'worker-receipt.json'
        workers.append(dict(candidate_id=spec['candidate_id'],spec=reference,pid=process.pid,exit_code=process.returncode,
            receipt=receipt(path) if path.exists() else None))
    return dict(schema='gsdeploy.parallel_fresh_CPU_campaign.v16',
        status='ALL_FOUR_CPU_WORKERS_EXITED_ZERO' if failure is None and len(workers)==4 else 'FAILED_PROHIBITS_SEAL',
        error=failure,workers=workers,all_workers_reaped=all(p.poll() is not None for p in processes),
        elapsed_seconds=time.monotonic()-started,timeout_seconds=timeout_seconds,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('spec','spec-sha256','protocol','protocol-sha256'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    result=run_worker(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.protocol,sha256=args.protocol_sha256))
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']==COMPLETE else 1)
