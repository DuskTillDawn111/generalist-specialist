"""Immutable batch-owner collection plan and explicit, one-attempt runner.

prepare/CPU fake collection never loads GPU models. Real collection requires
--execute, keeps declared G2+OCR background owners resident, and uses the new
single text weight owner (a different physical deployment from text x2).
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gsdeploy.runtime.batching import BatchConfiguration
from gsdeploy.runtime.batch_owner import run_arrivals, run_retained_arrivals
from gsdeploy.forecasting.batch_profiles import batch_input_key

SCHEMA = 'gsdeploy.batch_profile_collection.v2'


def read(path):return json.loads(Path(path).read_text())
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                   separators=(',',':'),allow_nan=False).encode()).hexdigest()
def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def checked(ref):
    if receipt(ref['path']) != ref:raise ValueError('Frozen source changed: '+ref['path'])
    return Path(ref['path'])
def save(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def prepare(output, spec_path, source_root, samples_path, schedules_path, configuration,
            *, background_registry=None):
    """Freeze independent public partitions and source/physical identities on CPU."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if not isinstance(configuration,BatchConfiguration):raise ValueError('BatchConfiguration required')
    source_root=Path(source_root).resolve();spec=read(spec_path)
    if (spec['candidate']['model']!='nvidia/Llama3-ChatQA-1.5-8B' or spec['candidate'].get('slot')!='text'
            or spec['execution']['family']!='chatqa'):
        raise ValueError('Explicit ChatQA text spec required')
    if not (source_root/'gsdeploy/backends/compact_native.py').exists():
        raise ValueError('Frozen native source-root required')
    samples,schedules=read(samples_path),read(schedules_path)
    if set(samples)!={'training','validation'} or set(schedules)!={'training','validation'}:
        raise ValueError('Immutable training and validation partitions required')
    fingerprints={};sample_ids={}
    for split in ('training','validation'):
        if not samples[split] or not schedules[split]:raise ValueError('Both splits must be nonempty')
        ids=[s['id'] for s in samples[split]]
        if len(set(ids))!=len(ids) or any(s.get('task')!='text' for s in samples[split]):
            raise ValueError('Distinct public text sample IDs required within split')
        sample_ids[split]=set(ids)
        fingerprints[split]={digest({k:s[k] for k in ('context','question')}) for s in samples[split]}
        request_ids=[r['request_id'] for r in schedules[split]]
        if (len(set(request_ids))!=len(request_ids) or any(r.get('sample_id') not in set(ids)
                or type(r.get('arrival_ns')) is not int or r['arrival_ns']<0 for r in schedules[split])):
            raise ValueError('Invalid public split schedule')
    if sample_ids['training']&sample_ids['validation'] or fingerprints['training']&fingerprints['validation']:
        raise ValueError('Training/validation public input overlap')
    if set(r['request_id'] for r in schedules['training'])&set(r['request_id'] for r in schedules['validation']):
        raise ValueError('Training/validation request IDs overlap')
    backgrounds={}
    owner_environment={'CUDA_VISIBLE_DEVICES':'0'}
    if spec['execution'].get('allocator_conf'):
        owner_environment['PYTORCH_CUDA_ALLOC_CONF']=spec['execution']['allocator_conf']
    if background_registry is not None:
        registry=read(background_registry)
        for family in ('generalist_2','ocr'):
            if family not in registry:raise ValueError('Keep both G2 and OCR resident')
            entry=copy.deepcopy(registry[family])
            if '--execute' not in entry['command'] or '--source-root' not in entry['command']:
                raise ValueError('Explicit native background entry required')
            if entry.get('env',{}).get('CUDA_VISIBLE_DEVICES')!='0':
                raise ValueError('All owners must explicitly use the same declared device 0')
            backgrounds[family+'#0']=entry
    paths={Path(__file__).resolve(),Path(spec_path).resolve(),Path(samples_path).resolve(),Path(schedules_path).resolve()}
    paths.update((ROOT/'gsdeploy').rglob('*.py'));paths.update((source_root/'gsdeploy').rglob('*.py'))
    for key in ('protocol','classmap'):
        paths.add(Path(spec['resources'][key]).resolve())
    if spec['execution'].get('peft_adapter'):
        paths.add(Path(spec['execution']['peft_adapter']['manifest']).resolve())
    if background_registry is not None:
        paths.add(Path(background_registry).resolve())
        paths.add(Path(__file__).with_name('native_collect.py'))
        for entry in backgrounds.values():
            command=entry['command'];paths.add(Path(command[1]).resolve())
            background_spec=Path(entry['spec']).resolve();paths.add(background_spec)
            frozen_source=Path(command[command.index('--source-root')+1]);paths.update(frozen_source.rglob('*.py'))
            doc=read(background_spec)
            if doc.get('adapter_source'):paths.add(Path(doc['adapter_source']))
            for key in ('protocol','classmap','inference_source'):
                if doc['resources'].get(key):paths.add(Path(doc['resources'][key]))
    source_receipts=[receipt(p) for p in sorted(paths)]
    output.mkdir(parents=True,exist_ok=False)
    new_spec=copy.deepcopy(spec)
    new_spec['batch_owner_execution']=dict(schema=SCHEMA,backend='same_owner_chatqa_hf_batch_v2',
        text_weight_owner_count=1,logical_text_aliases=['text#0','text#1'],
        common_submatrix_merge=False,prefix_cache_reuse=False)
    save(output/'text-owner-spec.json',new_spec)
    save(output/'background-registry.json',backgrounds)
    partitions={}
    for split in ('training','validation'):
        directory=output/split;directory.mkdir()
        save(directory/'samples.json',samples[split]);save(directory/'arrivals.json',schedules[split])
        partitions[split]=dict(samples=receipt(directory/'samples.json'),arrivals=receipt(directory/'arrivals.json'),
            requested=len(schedules[split]),role=split,actual_profiles_reused=False)
    config=dict(max_batch_size=configuration.max_batch_size,max_padded_tokens=configuration.max_padded_tokens,
                max_wait_ns=configuration.max_wait_ns,length_buckets=list(configuration.length_buckets))
    deployment=dict(schema='gsdeploy.batch_physical_deployment.v2',
        new_physical_owners={'shared_text':1,'generalist_2#0':1,'ocr#0':1},
        new_text_weight_copies=1,logical_text_routes=['text#0','text#1'],
        comparison_only_baseline={'text#0':1,'text#1':1,'generalist_2#0':1,'ocr#0':1},
        baseline_profiles_not_reused=True,baseline_capacity_g_not_inherited=True,
        residency_billing='New three-owner owned-process sampled VRAM and startup; no old two-replica memory credit',
        background_activity='IDLE_RESIDENT_ONLY; active-background request protocol not implemented',
        background_residency_required=['generalist_2#0','ocr#0'])
    save(output/'deployment.json',deployment)
    plan=dict(schema=SCHEMA,status='CPU_PREPARED_GPU_UNVERIFIED',source_receipts=source_receipts,
        text_spec=receipt(output/'text-owner-spec.json'),source_root=str(source_root),
        owner_environment=owner_environment,
        backgrounds=receipt(output/'background-registry.json'),deployment=receipt(output/'deployment.json'),
        configuration=config,partitions=partitions,
        execution_blockers=[] if backgrounds else ['G2_AND_OCR_BACKGROUND_BINDINGS_REQUIRED'],
        old_T8_V4_training_reused=False,independent_inputs_disjoint=True,
        hardware_resource_counters='UNKNOWN: no new successful SM/DRAM/L2 measurement',
        forecast_ready_scope='Actual serial prepare/owner availability must be modeled; no unconditional arrival+delay transfer',
        final_gain_eligible=False)
    save(output/'plan.json',plan)
    return plan


def audit(plan_path):
    plan=read(plan_path)
    if plan.get('schema')!=SCHEMA:raise ValueError('Wrong plan schema')
    refs=plan['source_receipts']+[plan[k] for k in ('text_spec','backgrounds','deployment')]
    refs += [ref for partition in plan['partitions'].values() for key,ref in partition.items() if key in ('samples','arrivals')]
    for ref in refs:checked(ref)
    return plan


def _profile_receipts(directory,report,plan_ref,split):
    runtime_ref=receipt(directory/'runtime.json');requests={}
    request_dir=directory/'requests';request_dir.mkdir()
    for row in report['results']:
        path=request_dir/(hashlib.sha256(row['request_id'].encode()).hexdigest()+'.json')
        save(path,dict(schema=SCHEMA+'.request',split=split,plan=plan_ref,runtime=runtime_ref,**row))
        requests[row['request_id']]=receipt(path)
    batches=[]
    profiles=directory/'batches';profiles.mkdir()
    by_id={r['request_id']:r for r in report['results']}
    for i,batch in enumerate(report['batches']):
        selected=[by_id[rid] for rid in batch['selection']['members']]
        key_rows=[dict(task='text',pool_id=r['pool_id'],execution_key=batch['backend_metrics']['compatibility_key']) for r in selected]
        key=batch_input_key(key_rows,batch['selection'])
        path=profiles/(str(i)+'.json')
        save(path,dict(schema=SCHEMA+'.actual_batch',split=split,plan=plan_ref,runtime=runtime_ref,
            batch_input_key=key,actual_members=batch['selection']['members'],
            request_receipts=[requests[r['request_id']] for r in selected],
            actual_batch=batch,resource_demand='UNKNOWN',ordered_kernel_plan='UNKNOWN',
            fit_eligible=False,phase_scope='Measured owner prepare/generate/receipt/next-ready wall events only'))
        batches.append(receipt(path))
    save(directory/'profile-index.json',dict(schema=SCHEMA+'.index',split=split,plan=plan_ref,
        runtime=runtime_ref,requests=requests,batches=batches,not_for_training=True,
        missing=['ordered kernel/resource evidence','qualified prep/service/power/memory model'],
        final_gain_eligible=False))


def collect(plan_path,split,*,execute=False,backend_factory=None,clock=None,wait=None):
    """One immutable split attempt; backend injection is CPU TEST ONLY."""
    if split not in ('training','validation'):raise ValueError('Unknown split')
    if not execute and backend_factory is None:raise RuntimeError('GPU collection requires explicit --execute')
    if execute and backend_factory is not None:raise ValueError('CPU doubles cannot be passed to a GPU execution')
    plan_path=Path(plan_path).resolve();plan_ref=receipt(plan_path);plan=audit(plan_path)
    if execute and plan['execution_blockers']:raise ValueError(';'.join(plan['execution_blockers']))
    if execute and any(os.environ.get(k)!=v for k,v in plan['owner_environment'].items()):
        raise ValueError('Declared owner CUDA/allocator environment must be applied before GPU loading')
    directory=plan_path.parent/split/'collection';directory.mkdir(exist_ok=False)
    save(directory/'attempt.json',dict(schema=SCHEMA+'.attempt',plan=plan_ref,split=split,
        execute=execute,CPU_test_only=not execute,retry_permitted=False))
    partition=plan['partitions'][split];config=plan['configuration']
    configuration=BatchConfiguration(config['max_batch_size'],config['max_padded_tokens'],
        config['max_wait_ns'],tuple(config['length_buckets']))
    options=dict(owner='shared_text',parent_receipt=plan_ref,output=directory/'runtime.json')
    if clock is not None:options['clock']=clock
    if wait is not None:options['wait']=wait
    runtime=monitor=gpu_lock=None;telemetry={'status':'UNKNOWN','reason':'CPU test; no physical telemetry','samples':[]}
    cleanup_errors=[]
    try:
        if execute:
            gpu_lock=open(ROOT/'artifacts/.gpu-experiment.lock','a')
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from gsdeploy.runtime.deployment import DeploymentRuntime
            from gsdeploy.runtime.schema import MeasurementConfig
            from .native_collect import BoardMonitor,descendant_or_self
            registry=read(checked(plan['backgrounds']))
            runtime=DeploymentRuntime(MeasurementConfig(deployment=dict.fromkeys(registry,1),
                workload_rps={},routing_rps={},idle_residency=True,startup_timeout_seconds=600),registry)
            class OwnedProxy:
                @property
                def workers(self):
                    return {**runtime.workers,'shared_text':SimpleNamespace(process=SimpleNamespace(pid=os.getpid()))}
                @property
                def ready(self):return {**runtime.ready,'shared_text':dict(pid=os.getpid())}
            proxy=OwnedProxy()
            try:monitor=BoardMonitor(proxy);monitor.start()
            except Exception as exc:telemetry=dict(status='UNKNOWN',reason=repr(exc),samples=[])
            runtime.start()
            for alias,worker in runtime.workers.items():
                pid=runtime.ready[alias].get('pid')
                if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):
                    raise ValueError('Background READY is not an owned process')
            audit(plan_path)
            if monitor is not None:options['event_sink']=lambda event:monitor.check()
            report=run_retained_arrivals(read(checked(partition['samples'])),read(checked(partition['arrivals'])),
                configuration,checked(plan['text_spec']),plan['source_root'],execute=True,
                expected_spec_sha256=plan['text_spec']['sha256'],**options)
        else:
            report=run_arrivals(read(checked(partition['samples'])),read(checked(partition['arrivals'])),
                                configuration,backend_factory,**options)
        audit(plan_path)
        _profile_receipts(directory,report,plan_ref,split)
        completed=dict(schema=SCHEMA+'.completed',plan=plan_ref,split=split,
            status='CPU_TEST_ONLY' if not execute else 'COLLECTED_REQUIRES_QUALIFICATION',
            delivered=report['delivered'],requested=partition['requested'],
            runtime=receipt(directory/'runtime.json'),profile_index=receipt(directory/'profile-index.json'),
            background_residency='CPU_TEST_UNKNOWN' if not execute else copy.deepcopy(runtime.ready),
            old_T8_V4_training_reused=False,not_for_training=True,final_gain_eligible=False)
    except Exception as exc:
        save(directory/'failed.json',dict(schema=SCHEMA+'.failed',plan=plan_ref,split=split,
            error=repr(exc),retry_permitted=False,not_for_training=True,final_gain_eligible=False))
        raise
    finally:
        if monitor is not None:
            try:monitor.close()
            except Exception as exc:cleanup_errors.append(repr(exc))
            telemetry=dict(status='UNKNOWN' if monitor.errors or not monitor.samples else 'SAMPLED_BOARD_ONLY',
                errors=monitor.errors,samples=monitor.samples,
                owned_peak_sampled_mib=max((s['owned_used_mib'] for s in monitor.samples),default=None),
                memory_limit='Owned PID sampled peak can miss transients; CUDA allocator peaks separate',
                power_scope='Whole board, not per model; SM/DRAM/L2 counters UNKNOWN')
        save(directory/'telemetry.json',telemetry)
        if runtime is not None:
            try:
                runtime.close()
                save(directory/'background-shutdown.json',runtime.shutdown_report)
            except Exception as exc:cleanup_errors.append(repr(exc))
        if gpu_lock is not None:
            fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    if cleanup_errors:
        save(directory/'failed.json',dict(schema=SCHEMA+'.failed',plan=plan_ref,split=split,
            error=cleanup_errors,retry_permitted=False,not_for_training=True,final_gain_eligible=False))
        raise RuntimeError('Collector cleanup failed: '+repr(cleanup_errors))
    completed['telemetry']=receipt(directory/'telemetry.json')
    if runtime is not None:completed['background_shutdown']=receipt(directory/'background-shutdown.json')
    save(directory/'completed.json',completed)
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare')
    for name in ('output','spec','source-root','samples','schedules'):p.add_argument('--'+name,required=True)
    p.add_argument('--background-registry');p.add_argument('--max-batch-size',type=int,default=2)
    p.add_argument('--max-padded-tokens',type=int,required=True);p.add_argument('--max-wait-ns',type=int,required=True)
    p.add_argument('--length-buckets',type=int,nargs='*',default=[])
    p=sub.add_parser('collect');p.add_argument('--plan',required=True)
    p.add_argument('--split',choices=['training','validation'],required=True);p.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':
        result=prepare(args.output,args.spec,args.source_root,args.samples,args.schedules,
            BatchConfiguration(args.max_batch_size,args.max_padded_tokens,args.max_wait_ns,tuple(args.length_buckets)),
            background_registry=args.background_registry)
    else:result=collect(args.plan,args.split,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
