"""Audit the four predeclared tiny-spillover development conditions.

Only terminal receipts are read. This compares measured conditions; it does
not select an algorithmic optimum, validate a predictor, or use reserved seeds.
"""
import argparse
from collections import Counter, defaultdict
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load('_fine_terminal_base', ROOT/'tools/spillover_background_20261022/analyze.py')
exposure = load('_fine_terminal_exposure', ROOT/'tools/spillover_capacity_20261022/exposure.py')


def api():
    result = exposure.frozen_api(SOURCE)
    prefix = result['trace'].__package__
    for name in ['config', 'evaluation', 'tasks', 'static_study', 'quality']:
        result[name] = importlib.import_module(prefix+'.'+name)
    return result


def stats(values):
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    values = sorted(values); point = (len(values)-1)*.95; lo = math.floor(point); hi = math.ceil(point)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=values[lo]+(values[hi]-values[lo])*(point-lo), maximum=values[-1])


def exact_replay(config, raw, samples, modules):
    trace = modules['trace'].make_trace(config.workload, config.horizon_seconds, config.random_seed, config.arrival_process)
    aliases, routes = modules['trace'].expand_routes(raw['plan']['deployment'], raw['plan']['routing'])
    router = modules['routing'].Router(routes)
    pool = modules['workload'].SamplePool(samples, list(config.workload), config.random_seed)
    window = raw['measurement_window']; rows = raw['request_records']
    assert len(rows) == len(trace)
    assert window['trace_sha256'] == modules['trace'].trace_digest(trace)
    cohort = []
    for i, ((offset, task), row) in enumerate(zip(trace, rows)):
        sample = pool.next(task); alias = router.choose(task)
        assert row['request_id'] == f'measurement:{i}' and row['arrival_index'] == i
        assert row['task'] == task and row['pool_id'] == sample['pool_id'] and row['model'] == alias
        assert row['scheduled_time_ns'] == window['start_time_ns']+round(offset*1e9)
        cohort.append((offset, task, sample['pool_id']))
    assert raw['replica_to_family'] == aliases
    return dict(trace_sha256=window['trace_sha256'], route_and_input_replay_verified=True,
                cohort_sha256=modules['quality'].quality_digest(cohort), requests=len(rows))


def native_quality(config, raw, modules):
    totals = {t: [0., 0., 0] for t in config.workload}; unknown = []; by_alias = defaultdict(lambda: [0.,0.,0])
    refs = config.catalog['quality']['references']
    for row in raw['request_records']:
        result = base.score_observed_native(row, row.get('worker_response') or {}, refs,
                                            modules['tasks'].score_result, modules['quality'].sample_totals)
        if result['status'] != 'AVAILABLE':
            unknown.append(dict(request_id=row['request_id'], reason=result.get('reason'))); continue
        for values in [totals[row['task']], by_alias[(row['model'],row['task'])]]:
            values[0] += result['error_numerator']; values[1] += result['reference_denominator']; values[2] += 1
    per_task = {t:dict(loss=a/b if b else None, error_numerator=a, reference_denominator=b, requests=n)
                for t,(a,b,n) in totals.items()}
    complete = not unknown and all(v[1] > 0 for v in totals.values())
    Q = math.fsum(config.workload[t]/math.fsum(config.workload.values())*v['loss'] for t,v in per_task.items()) if complete else None
    return dict(complete=complete, independently_rescored_Q_norm=Q, per_task=per_task, unknown=unknown,
        per_alias_task=[dict(alias=a,task=t,loss=x/y if y else None,requests=n) for (a,t),(x,y,n) in sorted(by_alias.items())])


def alias_summary(raw):
    groups = defaultdict(list)
    for row in raw['request_records']: groups[row['model']].append(row)
    summary = []
    for alias, rows in sorted(groups.items()):
        ledgers = [base.legacy.ledger(r, r.get('worker_response') or {}) for r in rows]
        good = [r['terms'] for r in ledgers if r['status']=='AVAILABLE']
        window = raw['measurement_window']; start = window['start_time_ns']; end = window['end_time_ns']
        native_services = [r['worker_response']['service_ms'] for r in rows if r.get('worker_response',{}).get('success') is True
                           and type(r['worker_response'].get('service_ms')) in (int,float)]
        tail_ledgers = [base.legacy.ledger(r,r.get('worker_response') or {}) for r in rows
                        if r['scheduled_time_ns'] >= start+round(.75*(end-start))]
        tail = [r['terms'] for r in tail_ledgers if r['status']=='AVAILABLE']
        service_seconds = math.fsum(native_services)/1000
        summary.append(dict(alias=alias, task_counts=dict(Counter(r['task'] for r in rows)),
            request_statuses=dict(Counter(r['status'] for r in rows)),
            text_input_counts=dict(Counter(r['pool_id'] for r in rows if r['task']=='text')),
            observed_service_sum_seconds=service_seconds,
            observed_service_sum_over_arrival_horizon=service_seconds/window['duration_seconds'],
            missing_service_requests=len(rows)-len(native_services),
            service_sum_scope='Actual completed cohort responses, including drain. Empirical sample work/H only; not formal rho or sustainable capacity.',
            end_of_arrival_window_backlog=sum(r['completion_time_ns'] is None or r['completion_time_ns']>end for r in rows),
            unfinished_after_drain=sum(r['status']!='completed' for r in rows),
            arrival_tail_fraction=.25,
            arrival_tail_timing_terms_ms={k:stats([r[k] for r in tail]) for k in
                ['host_enqueue_to_submit_ms','submit_to_worker_start_ms','observed_e2e_ms']} if tail else {},
            timing_terms_ms={k:stats([r[k] for r in good]) for k in good[0]} if good else {},
            unknown_timing_ledgers=len(ledgers)-len(good),
            negative_timing_flags=[flag for r in ledgers if r['status']=='AVAILABLE' for flag in r['negative_time_flags']],
            timing_scope='Host schedule-to-enqueue, host enqueue-to-submit, worker queue and handler retained separately; no clock residual correction.'))
    return summary


def verify_response_timing(row):
    response=row['worker_response']
    assert row['scheduled_time_ns'] <= row['enqueued_time_ns'] <= row['actual_submit_time_ns'] <= row['completion_time_ns']
    assert row['actual_submit_time_ns'] <= row['submit_write_completed_time_ns'] <= row['completion_time_ns']
    assert response['submit_ns']==row['actual_submit_unix_ns']
    assert response['submit_ns'] <= response['start_ns'] < response['end_ns'] <= row['completion_unix_ns']
    handler=(response['end_ns']-response['start_ns'])/1e6
    assert math.isclose(handler,response['worker_latency_ms'],abs_tol=1e-9)
    assert math.isclose((response['start_ns']-response['submit_ns'])/1e6,response['queue_wait_ms'],abs_tol=1e-9)
    assert 0 < response['service_ms'] <= handler


def specialist_imbalance(job):
    rows={r['alias']:r for r in job['per_alias'] if r['alias'].startswith('text#')}
    return dict(by_alias={a:{k:r[k] for k in ['task_counts','observed_service_sum_seconds',
        'observed_service_sum_over_arrival_horizon','missing_service_requests','end_of_arrival_window_backlog',
        'arrival_tail_timing_terms_ms']} for a,r in rows.items()},
        observed_service_work_gap_seconds=abs(rows['text#0']['observed_service_sum_seconds']-rows['text#1']['observed_service_sum_seconds']) if set(rows)=={'text#0','text#1'} else None,
        mechanism_limit='Changing tiny generalist weight remaps specialist input sequences under static SWRR. Input-mix redistribution and shared-GPU service changes may contribute; gains cannot be attributed solely to generalist capacity relief.')


def analyze_job(packet, job, raw, raw_path, modules):
    assert raw['experiment_job_id'] == job['id'] and raw['experiment_phase']=='calibration'
    assert raw['plan'] == job['plan'] and raw['backend']=='native'
    assert raw['status'] in {'completed','completed_with_failures','failed'}
    config = modules['config'].SystemConfig(**raw['config'])
    assert config.random_seed == job['seed'] and config.workload == packet['config']['workload']
    for key in ['weights','latency_reference_ms','gpu_memory_mib','drain_seconds','arrival_process',
                'energy_reference_watts','startup_amortization_seconds','utility_schema','horizon_seconds']:
        assert raw['config'][key] == packet['config'][key]
    for key in ['models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles']:
        assert config.catalog[key] == packet['config']['catalog'][key]
    expected = copy.deepcopy(job['identity'])
    expected['runtime_checkpoint_sha256'] = config.catalog['measurement_context']['runtime_checkpoint_sha256']
    expected['workers'] = {m: packet['workers'][m] for m in job['plan']['deployment']}
    expected['workers'] = copy.deepcopy(expected['workers'])
    binding_path=raw_path.parent/'native-binding/quality-evidence.json'
    binding = json.loads(binding_path.read_text())
    # Original bind_native_job expands OCR's directory inventory and adds this
    # exact allow-list to its worker environment before execution.
    if 'ocr' in expected['workers']:
        expected['workers']['ocr']['env']['GS_CHECKPOINT_FILES_JSON'] = json.dumps(sorted(binding['models']['ocr']['checkpoint']['files']))
    assert raw['executed_measurement_identity'] == expected
    assert raw['executed_measurement_id'] == modules['quality'].quality_digest(expected)
    assert raw['preflight']['gpu']['uuid'] == expected['hardware_contract']['device_uuid']
    assert {m:r['checkpoint']['sha256'] for m,r in binding['models'].items()} == expected['runtime_checkpoint_sha256']
    contract = json.loads((Path(packet['inputs'])/'study-contract.json').read_text())
    replay = exact_replay(config, raw, contract['runtime_input_pool'], modules)
    reevaluated = modules['evaluation'].evaluate_run(config, copy.deepcopy(raw))
    assert reevaluated == raw['evaluation'], 'Frozen evaluate_run differs from saved evaluation'
    observed = native_quality(config, raw, modules)
    utility = reevaluated['utility']; Q = observed['independently_rescored_Q_norm']
    if Q is not None:
        assert math.isclose(Q, utility['Q_norm'], rel_tol=1e-12, abs_tol=1e-12)
        for task,value in observed['per_task'].items():
            assert value == utility['quality_evidence']['per_task'][task]
    for row in raw['request_records']:
        response = row.get('worker_response') or {}
        if row['status']=='completed':
            verify_response_timing(row)
            assert response['request_id']==row['request_id'] and response['pool_id']==row['pool_id']
            assert response['arrival_index']==row['arrival_index'] and response['worker']==raw['replica_to_family'][row['model']]
            assert response['task_type']==row['task']
            family = raw['replica_to_family'][row['model']]
            worker_spec = json.loads(Path(packet['workers'][family]['spec']).read_text())
            base.independent.check_response_protocol(response, worker_spec, row['request_id'])
    w = raw['measurement_window']
    resource = base.resource_summary(raw, raw['config'], w['start_time_ns'], w['end_time_ns'], w['start_unix_ns']-w['start_time_ns'])
    shutdown_pass = set(raw['shutdown'])==set(raw['replica_to_family']) and all(
        r['exit_code']==0 and not r['forced'] and r['bye_received'] for r in raw['shutdown'].values())
    return dict(job_id=job['id'],label=job['labels'][0],seed=job['seed'],plan=raw['plan'],
        audit_status='VERIFIED_TERMINAL_EVIDENCE',raw_status=raw['status'],raw_errors=raw['errors'],
        **replay,**resource,shutdown_pass=shutdown_pass,original_evaluate_run_verified=True,native_quality=observed,
        utility=utility,stability=reevaluated['stability'],stability_evidence=reevaluated.get('stability_evidence'),
        per_alias=alias_summary(raw),per_task_metrics=raw.get('per_task_metrics'),system_metrics=raw.get('system_metrics'))


def pairs(jobs):
    result=[]
    for seed in [107,108]:
        rows=[r for r in jobs if r['seed']==seed and r.get('audit_status')=='VERIFIED_TERMINAL_EVIDENCE']
        if len(rows)!=2:
            result.append(dict(seed=seed,status='INCOMPLETE_OR_UNVERIFIED_PAIR',delta_U=None));continue
        left=next(r for r in rows if not r['plan']['routing']['text'].get('generalist_2'))
        right=next(r for r in rows if r['plan']['routing']['text'].get('generalist_2'))
        assert left['cohort_sha256']==right['cohort_sha256'], 'Paired route-independent input/arrival cohort changed'
        valid=all(r['utility'].get('eligible') and r['memory_pass'] and r['energy_pass'] and r['shutdown_pass'] for r in [left,right])
        delta={k:right['utility'].get(k)-left['utility'].get(k) if type(right['utility'].get(k)) in (int,float)
               and type(left['utility'].get(k)) in (int,float) else None for k in
               ['U','diagnostic_U']+[f'{t}_{kind}' for t in 'QTLPME' for kind in ['norm','contribution']]}
        if not valid:delta['U']=None
        result.append(dict(seed=seed,status='ELIGIBLE_PAIRED_COMPARISON' if valid else 'INELIGIBLE_DIAGNOSTIC_PAIR',
            no_split_job_id=left['job_id'],split_job_id=right['job_id'],route_independent_cohort_equal=True,
            delta_split_minus_no_split=delta,delta_U=delta['U'],stability_no_split=left['stability'],stability_split=right['stability'],
            specialist_imbalance_no_split=specialist_imbalance(left),specialist_imbalance_split=specialist_imbalance(right),
            scope='Two measured development routes only; no algorithmic optimum or independent confirmation claim'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['packet','queue','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite measured evidence')
    packet=json.loads(args.packet.read_text()); queue=json.loads(args.queue.read_text()); modules=api()
    assert modules['static_study'].validate_packet(packet)['valid']
    assert queue['packet']==exposure.receipt(args.packet) and queue['status']!='RUNNING'
    assert [(j['seed'],bool(j['plan']['routing']['text'].get('generalist_2'))) for j in packet['jobs']]==[(107,False),(107,True),(108,True),(108,False)]
    registrations={r['job_id']:r for r in queue['jobs']}; assert len(registrations)==len(queue['jobs'])
    jobs=[]; receipts=[]
    for job in packet['jobs']:
        registration=registrations.get(job['id'])
        if registration is None:
            jobs.append(dict(job_id=job['id'],label=job['labels'][0],seed=job['seed'],audit_status='DECLARED_NOT_EXECUTED',U=None));continue
        raw_path=Path(registration['raw']['path']); assert exposure.receipt(raw_path)==registration['raw']
        receipts.append(registration['raw']);raw=json.loads(raw_path.read_text())
        try: result=analyze_job(packet,job,raw,raw_path,modules)
        except (AssertionError,KeyError,ValueError,TypeError,IndexError,OSError) as exc:
            result=dict(job_id=job['id'],label=job['labels'][0],seed=job['seed'],audit_status='AUDIT_INCOMPLETE_OR_FAILED',
                raw_status=raw.get('status'),raw_errors=raw.get('errors'),audit_error=type(exc).__name__+': '+str(exc),
                audit_traceback=traceback.format_exc(),request_statuses=dict(Counter(r['status'] for r in raw.get('request_records',[]))),U=None)
        jobs.append(result)
    paired=pairs(jobs)
    report=dict(schema='gsdeploy.text185_fine_development.analysis.v1',jobs=jobs,paired=paired,
        status='ALL_FOUR_TERMINALS_AUDITED' if all(j['audit_status']=='VERIFIED_TERMINAL_EVIDENCE' for j in jobs) else 'INCOMPLETE_OR_FAILED_AUDIT',
        source_packet=exposure.receipt(args.packet),source_queue=exposure.receipt(args.queue),raw_receipts=receipts,
        predictor_activated=False,algorithm_selected=False,optimality_proven=False,
        limitations=['Development seeds107108 only; reserved seeds untouched.','All declared conditions retained, including ineligible, failed, or unexecuted.',
                     'Finite static SWRR exposure can repeat very few generalist text inputs; per-alias exact input counts included.'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    sources=[Path(__file__),Path(base.__file__),Path(exposure.__file__)]+[Path(v.__file__) for v in modules.values()]
    (args.output/'manifest.json').write_text(json.dumps(dict(sources=[exposure.receipt(p) for p in sources],raw_receipts=receipts,
        report=exposure.receipt(args.output/'report.json')),indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],jobs=len(jobs),paired=[{k:v for k,v in p.items() if k in ['seed','status','delta_U']} for p in paired])))


if __name__=='__main__':main()
