"""Source-bound native full-arrival flow collection with parent-receipt E2E.

Static routes validate the forecast before enabling its online decisions. Every
candidate receives the identical declared arrival/input cohort, including requests
that queue. This module never substitutes observed future outcomes into forecasts.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import time

from gsdeploy.resources import prepare_registry, validate_resources
from gsdeploy.routing import Router
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.tasks import page_ocr_text, score_result
from gsdeploy.quality import sample_totals
from .background_collect import ROOT, context_identity, descendant_or_self, receipt, save
from .startup_collect_v4 import StartupBoardMonitor as BoardMonitor
from .profiling import digest, RESIDENTS

SCHEMA = 'gsdeploy.native_flow_plan.v1'
BASE = ROOT/'artifacts/static-study-v6-20261021/inputs'


def collection_source_paths():
    return [Path(__file__).resolve(),*[Path(__file__).with_name(name) for name in (
        'startup_collect_v4.py','background_collect.py','prediction_observed_v2.py','flow_metrics.py')]]


def resource_configuration(max_active=None):
    value=dict(schema='gsdeploy.resource_configuration.v1',device='0',
        mode='shared' if max_active is None else 'request_admission',
        allocations={a:({} if max_active is None else {'max_inflight':1}) for a in RESIDENTS})
    if max_active is not None:
        value['groups']={'gpu':dict(aliases=list(RESIDENTS),max_active=max_active)}
    return validate_resources(value,RESIDENTS)


def prepare(profile_dir,output,*,duration=90.,warmup=0.,drain=120.,seed=627301,
            text_rate=1.85,background_rate=.05,include_admission=False):
    """Predeclare a finite pilot domain; these seeds are not final confirmation."""
    if min(duration,drain,text_rate,background_rate)<=0 or warmup<0:
        raise ValueError('Positive duration, drain and offered rates required')
    profile_dir,output=Path(profile_dir).resolve(),Path(output).resolve()
    context=json.loads((profile_dir/'context.json').read_text())
    profile_plan=json.loads((profile_dir/'plan.json').read_text())
    training_raw_path=profile_dir/'raw.json'
    training_raw=json.loads(training_raw_path.read_text()) if training_raw_path.exists() else None
    if training_raw is not None and (training_raw.get('status')!='completed' or
            any(r.get('split')!='training' for r in training_raw['request_records'])):
        raise ValueError('Prephase memory proxy requires completed training-only raw')
    prephase_peak=max((s['owned_used_mib'] for s in training_raw['board_samples']),default=None) if training_raw else None
    ctx=context_identity(context)
    if ctx!=profile_plan['execution_context_sha256']:
        raise ValueError('Profile execution context mismatch')
    known={(a['task'],a['pool_id']) for a in profile_plan['input_assignments'] if a['view']=='known_input'}
    inputs={t:sorted(pool for task,pool in known if task==t) for t in ('text','image','asr','ocr')}
    if any(not rows for rows in inputs.values()):
        raise ValueError('All four task input domains required')
    rates=dict(text=text_rate,image=background_rate,asr=background_rate,ocr=background_rate)
    rng=random.Random('full-flow-inputs:'+str(seed))
    trace=[]
    # Explicit periodic arrivals with task-specific offsets cover every task and
    # are public inputs to the forecast, not secretly replayed observed outcomes.
    for task,rate in rates.items():
        offset={'text':0.,'image':.19,'asr':.47,'ocr':.73}[task]/rate
        index=0
        while offset+index/rate<duration:
            trace.append(dict(offset_ns=round((offset+index/rate)*1e9),task=task,
                              pool_id=inputs[task][rng.randrange(len(inputs[task]))]))
            index+=1
    trace.sort(key=lambda r:(r['offset_ns'],r['task']))
    for i,row in enumerate(trace):
        row['request_id']='measurement:'+str(i)
    aliases={a:dict(family=f,supported_tasks=(['text','image','asr','ocr'] if f=='generalist_2' else
                 ['ocr'] if f=='ocr' else ['text'])) for a,f in RESIDENTS.items()}
    candidates=[]
    for mode,max_active in ([('shared',None),('admission1',1),('admission2',2)] if include_admission else [('shared',None)]):
        for label,fraction,spill in [('balanced',.5,0.),('skewed',.25,0.),('spill_low',.5,1/6),('spill_high',.5,1/3)]:
            text_route={'text#0':(1-spill)*fraction,'text#1':(1-spill)*(1-fraction)}
            if spill:
                text_route['generalist_2#0']=spill
            routes={'text':text_route,'image':{'generalist_2#0':1.},
                    'asr':{'generalist_2#0':1.},'ocr':{'ocr#0':1.}}
            router=Router(routes)
            candidates.append(dict(id=mode+'_'+label,execution_context_sha256=ctx,
                routes={r['request_id']:router.choose(r['task']) for r in trace},
                routing_weights=routes,allow_spill=bool(spill),
                predicted_prephase_peak_mib=prephase_peak,
                resource_configuration=resource_configuration(max_active)))
    catalog=json.loads((BASE/'catalog.json').read_text())
    plan=dict(schema=SCHEMA,kind='prospective_pilot',execution_context_sha256=ctx,
        context=receipt(profile_dir/'context.json'),profile_plan=receipt(profile_dir/'plan.json'),
        catalog=receipt(BASE/'catalog.json'),
        prephase_memory_evidence=dict(source=receipt(training_raw_path),raw_sampled_peak_mib=prephase_peak,
            scope='Training empirical peak proxy for new prephase maximum; no safety bound') if training_raw else None,
        runner_source=receipt(__file__),window_seconds=duration,drain_seconds=drain,warmup_seconds=warmup,
        collector_sources=[receipt(p) for p in collection_source_paths()],
        seed=seed,arrival_process='declared_periodic',aliases=aliases,workload_rps=rates,
        latency_reference_ms=dict(text=1000,image=250,asr=3000,ocr=10000),weights={k:1/6 for k in 'QTLPME'},
        memory_safe_factor=catalog['safe_factor'],memory_budget_mib=49140,energy_reference_watts=450.,
        startup_amortization_seconds=3600.,trace=trace,candidates=candidates,
        baseline_candidate_id='shared_balanced',queue_sample_offsets_ns=[round(i*duration/100*1e9) for i in range(101)],
        startup_comparison_policy=dict(kind='common_deployment_backend_omitted',
            group_id='fixed-residents-shared-no-warmup',
            candidate_ids=[c['id'] for c in candidates if c['id'].startswith('shared_')]) if warmup==0 else None,
        scope='Independent known-input full-arrival queue and static-candidate pilot; not final gain confirmation')
    plan['trace_sha256']=digest(trace)
    plan['plan_sha256']=digest(plan)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'plan.json',plan)
    return plan


def checked(reference):
    if receipt(reference['path'])!=reference:
        raise ValueError('Changed bound source: '+reference['path'])
    return Path(reference['path'])


def execute(plan_path,candidate_id,output,*,protocol_path=None,predictions_path=None,run_id=None):
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=json.loads(plan_path.read_text())
    if plan.get('schema')!=SCHEMA or digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Flow plan schema/hash mismatch')
    gate_arguments=(protocol_path,predictions_path,run_id)
    if any(v is not None for v in gate_arguments) and not all(v is not None for v in gate_arguments):
        raise ValueError('Validation requires protocol, predictions and run ID together')
    precollection=None
    if protocol_path is not None:
        if plan.get('split')=='training' or plan.get('phase') in ('training','predictor_training'):
            raise ValueError('Training plans cannot be relabeled as independent validation')
        from .prediction_observed_v2 import verify_before_collection
        protocol=json.loads(Path(protocol_path).read_text())
        if any(receipt(p) not in protocol.get('sources',[]) for p in collection_source_paths()):
            raise ValueError('Validation protocol must freeze the actual flow collector source closure')
        precollection=verify_before_collection(protocol_path,predictions_path,plan_path,
            run_id=run_id,candidate_id=candidate_id,output_root=output)
    checked(plan['runner_source'])
    for ref in plan.get('collector_sources',[]):checked(ref)
    checked(plan['catalog'])
    context=json.loads(checked(plan['context']).read_text())
    profile_plan=json.loads(checked(plan['profile_plan']).read_text())
    if context_identity(context)!=plan['execution_context_sha256']:
        raise ValueError('Execution context mismatch')
    for source in context['execution_sources']:
        checked(source['frozen'])
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Changed worker execution source: '+source['original_path'])
    candidate=next(c for c in plan['candidates'] if c['id']==candidate_id)
    registry={}
    for alias,family in RESIDENTS.items():
        entry=copy.deepcopy(context['worker_bindings'][family])
        entry['command'][1]=str(Path(__file__).with_name(context['worker_variant']))
        entry['env'].update(context['threads']);entry['env']['PYTHONUNBUFFERED']='1'
        registry[alias]=entry
    registry=prepare_registry(registry,candidate['resource_configuration'])
    # READY checks include every route used by the flow or priming, even when
    # the measured text route happens to omit the generalist.
    readiness_routes={'text':{'text#0':.5,'text#1':.49,'generalist_2#0':.01},
                      'image':{'generalist_2#0':1.},'asr':{'generalist_2#0':1.},'ocr':{'ocr#0':1.}}
    config=MeasurementConfig(deployment=dict.fromkeys(RESIDENTS,1),workload_rps=plan['workload_rps'],
        routing_rps=readiness_routes,startup_timeout_seconds=600,shutdown_timeout_seconds=30,
        drain_timeout_seconds=plan['drain_seconds'],measurement_seconds=plan['window_seconds'],
        routing_source='frozen_full_arrival_static_candidate')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'invocation.json',dict(plan=receipt(plan_path),candidate_id=candidate_id,runner=receipt(__file__)))
    runtime=DeploymentRuntime(config,registry)
    monitor=None
    raw=dict(schema='gsdeploy.native_flow_collection.v1',kind='measured',backend='native',status='starting',
        execution_context_sha256=plan['execution_context_sha256'],plan=receipt(plan_path),context=plan['context'],
        candidate_id=candidate_id,trace_sha256=plan['trace_sha256'],resource_configuration=candidate['resource_configuration'],
        resident_aliases=list(RESIDENTS),replica_to_family=RESIDENTS,
        request_records=[],priming_request_records=[],warmup_request_records=[],errors=[],
        effective_objective={k:plan[k] for k in ('workload_rps','latency_reference_ms','weights','memory_safe_factor',
            'memory_budget_mib','energy_reference_watts','startup_amortization_seconds')})
    if precollection is not None:
        raw.update(split='validation',run_id=run_id,precollection=receipt(precollection['receipt_path']))
    pool={r['pool_id']:r for r in (json.loads(line) for line in (BASE/'runtime-inputs.jsonl').read_text().splitlines())}
    lock=open(ROOT/'artifacts/.gpu-experiment.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        if precollection is not None:
            # Another experiment can hold the GPU lock for minutes. Recheck
            # frozen identities after that wait, before this collection starts.
            from .prediction_observed_v2 import _bindings
            for key in ('protocol','predictions','plan','adapter_source','scorer_source'):
                checked(precollection[key])
            checked(raw['precollection'])
            _bindings(protocol_path,predictions_path,plan_path,run_id)
            # Record the real collection boundary before any board sampling or
            # worker startup, after waiting for the exclusive GPU lock. Never
            # reconstruct this timestamp from completed output metadata.
            raw['collection_started_unix_ns']=time.time_ns()
            if raw['collection_started_unix_ns']<=precollection['checked_unix_ns']:
                raise RuntimeError('Collection must start after prospective prediction check')
        monitor=BoardMonitor(runtime)
        monitor.phase='startup'
        monitor.sample();raw['startup_begin_ns']=time.perf_counter_ns();monitor.start()
        runtime.start();raw['startup_end_ns']=time.perf_counter_ns()
        for alias,worker in runtime.workers.items():
            pid=runtime.ready[alias].get('pid')
            if type(pid) is not int or not descendant_or_self(pid,worker.process.pid):
                raise RuntimeError('Unowned READY PID: '+alias)
        raw['ready']=copy.deepcopy(runtime.ready)
        monitor.sample();monitor.check()
        print(json.dumps(dict(event='ready',candidate=candidate_id)),flush=True)
        monitor.phase='priming'
        for alias,family in RESIDENTS.items():
            tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
            for task in tasks:
                selected=sorted((r for r in profile_plan['inputs'] if r['task']==task),
                                key=lambda r:r['input_size'],reverse=True)[:3]
                for item in selected:
                    row=record('priming',len(raw['priming_request_records']),task,alias,pool[item['pool_id']],time.perf_counter_ns())
                    raw['priming_request_records'].append(row);runtime.enqueue(row,pool[item['pool_id']]['sample'])
                    runtime.drain([row],300)
                    if row['status']!='completed':
                        raise RuntimeError('Priming failed')
        def wait(target):
            while time.perf_counter_ns()<target:
                monitor.check();time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
        # Warmup is separately offered/drained, never inserted into measurement.
        monitor.phase='warmup'
        warmup_start=time.perf_counter_ns()
        for index,item in enumerate(plan['trace']):
            if item['offset_ns']>=plan['warmup_seconds']*1e9:
                break
            target=warmup_start+item['offset_ns'];wait(target)
            row=record('warmup',index,item['task'],candidate['routes'][item['request_id']],pool[item['pool_id']],target)
            raw['warmup_request_records'].append(row);runtime.enqueue(row,pool[item['pool_id']]['sample'])
        wait(warmup_start+round(plan['warmup_seconds']*1e9));runtime.drain(raw['warmup_request_records'],plan['drain_seconds'])
        if any(r['status']!='completed' for r in raw['warmup_request_records']):
            raise RuntimeError('Warmup failed')
        monitor.sample();monitor.check()
        start=time.perf_counter_ns();deadline=start+round(plan['window_seconds']*1e9)
        monitor.phase='measurement'
        raw.update(phase_start_ns=start,deadline_ns=deadline)
        for i,item in enumerate(plan['trace']):
            row=record('measurement',i,item['task'],candidate['routes'][item['request_id']],pool[item['pool_id']],start+item['offset_ns'])
            if row['request_id']!=item['request_id']:
                raise ValueError('Record request identity mismatch')
            raw['request_records'].append(row)
        for i,(item,row) in enumerate(zip(plan['trace'],raw['request_records'])):
            wait(row['scheduled_time_ns']);runtime.enqueue(row,pool[item['pool_id']]['sample'])
            if i%50==0:
                print(json.dumps(dict(event='flow_progress',offered=i+1,total=len(plan['trace']))),flush=True)
        wait(deadline);raw['arrival_loop_end_ns']=time.perf_counter_ns()
        monitor.phase='drain'
        runtime.drain(raw['request_records'],plan['drain_seconds'])
        raw['observation_end_ns']=time.perf_counter_ns();monitor.sample();monitor.check()
        raw['status']='completed'
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if monitor:
            monitor.close();raw['board_samples']=monitor.samples;raw['telemetry_errors']=monitor.errors
        runtime.close();raw['shutdown']=runtime.shutdown_report;raw['runtime_errors']=runtime.errors
        raw['admission_snapshot_after']=runtime.admission.snapshot() if runtime.admission else None
        raw['collection_complete']=raw['status']=='completed'
        raw['qualification_issues']=[]
        if any(r['status']!='completed' for r in raw['request_records']):
            raw['qualification_issues'].append('INCOMPLETE_OR_ERROR_REQUESTS')
        if runtime.errors or raw.get('telemetry_errors'):
            raw['qualification_issues'].append('RUNTIME_OR_TELEMETRY_ERRORS')
        if any(s.get('forced') or s.get('exit_code')!=0 or not s.get('bye_received') for s in runtime.shutdown_report.values()):
            raw['qualification_issues'].append('UNCLEAN_SHUTDOWN')
        save(output/'raw.json',raw)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    # Labels and original references are first read after inference has finished.
    catalog=json.loads((BASE/'catalog.json').read_text());totals={};quality_errors=[]
    for row in raw['request_records']:
        if row['status']!='completed':
            continue
        try:
            result=copy.deepcopy(row['worker_response'])
            if row['task']=='ocr':
                result['prediction']=page_ocr_text(result['prediction'])
            reference=catalog['quality']['references'][row['task']][row['pool_id']]
            numerator,denominator=sample_totals(row['task'],score_result(reference,result))
            totals[row['request_id']]=dict(task=row['task'],numerator=numerator,denominator=denominator)
        except (KeyError,TypeError,ValueError) as exc:
            quality_errors.append(dict(request_id=row['request_id'],error=str(exc)))
    evidence=dict(kind='measured_per_request',totals_by_request=totals,errors=quality_errors,
        source_receipts=[receipt(BASE/'catalog.json'),receipt(__file__),receipt(ROOT/'gsdeploy/tasks.py'),receipt(ROOT/'gsdeploy/quality.py')])
    # Keep the originally saved native raw immutable; annotated raw binds it.
    annotated={**raw,'quality_evidence':evidence,'native_raw':receipt(output/'raw.json')}
    save(output/'scored-raw.json',annotated)
    print(json.dumps(dict(status=raw['status'],candidate=candidate_id,requests=len(raw['request_records']),
                          quality_count=len(totals),errors=raw['errors'],output=str(output/'scored-raw.json'))),flush=True)
    return annotated


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--profile-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=90.);p.add_argument('--warmup',type=float,default=0.)
    p.add_argument('--drain',type=float,default=120.);p.add_argument('--seed',type=int,default=627301)
    p.add_argument('--text-rate',type=float,default=1.85);p.add_argument('--background-rate',type=float,default=.05)
    p.add_argument('--include-admission',action='store_true')
    p=sub.add_parser('execute');p.add_argument('--plan',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    p.add_argument('--protocol');p.add_argument('--predictions');p.add_argument('--run-id')
    args=parser.parse_args()
    if args.op=='prepare':
        plan=prepare(args.profile_dir,args.output,duration=args.duration,warmup=args.warmup,drain=args.drain,
            seed=args.seed,text_rate=args.text_rate,background_rate=args.background_rate,include_admission=args.include_admission)
        print(json.dumps(dict(status='PREPARED',requests=len(plan['trace']),candidates=len(plan['candidates']))))
    else:
        execute(args.plan,args.candidate,args.output,protocol_path=args.protocol,
            predictions_path=args.predictions,run_id=args.run_id)


if __name__=='__main__':
    main()
