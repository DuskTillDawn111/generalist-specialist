"""Review the complete offered training pilot; never qualify predictor accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .device_progress_review_contract_v4 import V6, monitoring_boundaries, qualify_roles
from .device_progress_actual_review_v3 import valid_live_ownership
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt, save
from .paired_profile_campaign_v6 import audit
from .profile_training_supervisor_v3 import completed_execution
from .profiling import RESIDENTS


def require(value, message):
    if not value:
        raise ValueError(message)


def ns(value):
    return type(value) is int and value >= 0


def checked(ref):
    require(receipt(ref['path']) == ref, 'Immutable evidence changed')
    return json.loads(Path(ref['path']).read_text())


def inspect_request_clock(row):
    """Pipe write and handler start are independent branches after send begins."""
    keys = ('scheduled_time_ns', 'enqueued_time_ns', 'actual_submit_time_ns',
        'submit_write_completed_time_ns', 'observed_worker_start_time_ns',
        'completion_time_ns', 'response_reader_event_enqueue_time_ns',
        'response_collector_dequeued_time_ns', 'response_collector_processing_time_ns')
    require(all(ns(row.get(k)) for k in keys), 'Complete integer request chronology required')
    response = row['worker_response']
    begin, end = response.get('start_perf_ns'), response.get('end_perf_ns')
    require(ns(begin) and ns(end) and row['observed_worker_start_time_ns'] == begin,
        'Observed START must bind the worker response clock')
    scheduled, enqueued, submit = (row[k] for k in keys[:3])
    received = row['response_reader_event_enqueue_time_ns']
    require(scheduled <= enqueued <= submit <= begin <= end <= received
        and submit <= row['submit_write_completed_time_ns'] <= received
        and row['completion_time_ns'] == received
        and received <= row['response_collector_dequeued_time_ns']
            <= row['response_collector_processing_time_ns'],
        'Arrival, dispatch, worker and parent availability chronology required')


def inspect_cohort(plan, raw):
    """Independent counts/order/times; no fitted outcomes or quality labels."""
    require(plan['split'] == 'training' and len(plan['jobs']) == 26,
        'Original 26-job training pilot required')
    require(raw['status'] == 'completed' and not any(raw.get(k) for k in
        ('errors', 'runtime_errors', 'telemetry_errors')), 'Clean complete training raw required')
    jobs, rows, batches = plan['jobs'], raw['request_records'], raw['batches']
    expected = Counter((j['id'], a, p, j['tasks'][a], 'training', 'training')
        for j in jobs for a, p in j['inputs'].items())
    observed = Counter((r['profile_job_id'], r['model'], r['pool_id'], r['task'],
        r['split'], r['validation_view']) for r in rows)
    require(expected == observed and len(rows) == sum(expected.values()),
        'Complete exact offered job/owner/input/task cohort required')
    require(len({r['request_id'] for r in rows}) == len(rows), 'Unique measured request IDs required')
    require([b['job_id'] for b in batches] == [j['id'] for j in jobs],
        'Exact complete planned job order required')
    previous_end = None
    for job, batch in zip(jobs, batches):
        start, end = batch['start_ns'], batch['end_ns']
        require(ns(start) and ns(end) and start <= end and
            (previous_end is None or previous_end <= start), 'Finite nonoverlapping job boundaries required')
        selected = [r for r in rows if r['profile_job_id'] == job['id']]
        require(Counter(batch['request_ids']) == Counter(r['request_id'] for r in selected),
            'Batch must retain every exact request identity')
        for row in selected:
            inspect_request_clock(row)
            scheduled = start + round(job['start_offsets_ms'][row['model']] * 1e6)
            received = row.get('response_reader_event_enqueue_time_ns')
            require(row['status'] == 'completed' and row.get('worker_response', {}).get('success') is True
                and row['scheduled_time_ns'] == scheduled
                and row.get('completion_time_ns') == received and ns(received)
                and scheduled <= received <= end, 'Exact arrival and complete parent receipt within job required')
        previous_end = end
    expected_priming = []
    for alias, family in RESIDENTS.items():
        tasks = ['text', 'image', 'asr'] if family == 'generalist_2' else ['ocr' if family == 'ocr' else 'text']
        for task in tasks:
            selected = sorted((x for x in plan['inputs'] if x['task'] == task),
                key=lambda x: x['input_size'], reverse=True)[:3]
            expected_priming.extend((alias, task, x['pool_id']) for x in selected)
    priming = raw['priming_request_records']
    for row in priming:
        inspect_request_clock(row)
    require([(r['model'], r['task'], r['pool_id']) for r in priming] == expected_priming
        and all(r['status'] == 'completed' and r.get('worker_response', {}).get('success') is True
            for r in priming), 'Exact ordered successful priming required')
    require(len({r['request_id'] for r in priming}) == len(priming)
        and not {r['request_id'] for r in priming} & {r['request_id'] for r in rows},
        'Priming and measured request identities must remain disjoint')
    require(len(raw['ready']) == 4 and set(raw['ready']) == set(RESIDENTS)
        and len({v['pid'] for v in raw['ready'].values()}) == 4, 'Four distinct resident workers required')
    require(set(raw['shutdown']) == set(RESIDENTS) and all(v['exit_code'] == 0
        and v['forced'] is False and v['bye_received'] is True for v in raw['shutdown'].values()),
        'Clean exit of all four workers required')
    return dict(jobs=len(jobs), measured_requests=len(rows), priming_requests=len(priming),
        measured_requests_by_owner=dict(Counter(r['model'] for r in rows)))


def inspect_samples(raw, collector_pid):
    boundary = monitoring_boundaries(raw, V6)
    require(not boundary['issues'], 'Full startup-through-drain telemetry scope required')
    pids = {r['pid'] for r in raw['ready'].values()}
    ready_evidence = raw.get('ready_ownership_evidence', {})
    require(set(ready_evidence) == set(RESIDENTS), 'Every READY worker needs live ancestry evidence')
    pins = {}
    for alias, ready in raw['ready'].items():
        rows = ready_evidence[alias]
        require(len(rows) == 1 and rows[0]['alias'] == alias and rows[0]['pid'] == ready['pid']
            and valid_live_ownership(rows[0], collector_pid, collector_pid), 'Exact READY identity/ancestry required')
        pins[alias] = tuple(rows[0][k] for k in
            ('launcher_pid', 'launcher_start_ticks', 'collector_start_ticks'))
    last = -1
    for sample in raw['board_samples']:
        owned = {int(k): v for k, v in sample['owned_processes_mib'].items()}
        processes = {int(k): v for k, v in sample['processes_mib'].items()}
        evidence = sample['ownership_evidence']
        require(ns(sample['time_ns']) and last <= sample['time_ns'] and owned == processes
            and ns(sample['query_duration_ns']) and type(sample['watts']) in (int, float)
            and math.isfinite(sample['watts']) and sample['watts'] > 0
            and not sample['foreign_pids'] and not sample['unresolved_nvml_pids']
            and not sample['unresolved_ownership_reasons'], 'Ordered entirely owned NVML samples required')
        require({e['pid'] for e in evidence} == set(owned) and len(evidence) == len(owned)
            and all(valid_live_ownership(e, collector_pid, collector_pid) for e in evidence),
            'Every NVML process requires stable declared-launcher ancestry')
        require(len(sample['owned_aliases']) == len(set(sample['owned_aliases']))
            and set(sample['owned_aliases']) == {e['alias'] for e in evidence},
            'Every sample alias set must exactly match live ownership evidence')
        require(all(e['alias'] in pins and tuple(e[k] for k in
            ('launcher_pid', 'launcher_start_ticks', 'collector_start_ticks')) == pins[e['alias']]
            for e in evidence), 'Owned aliases must retain their exact READY launcher identities')
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in owned.values())
            and type(sample['owned_used_mib']) in (int, float) and math.isfinite(sample['owned_used_mib']),
            'Finite nonnegative sampled memory required')
        require(abs(sum(owned.values()) - sample['owned_used_mib']) < 1e-6,
            'Owned sampled peak arithmetic mismatch')
        if sample['phase'] in ('measurement', 'final_drain'):
            require(pids <= set(owned) and set(sample['owned_aliases']) == set(RESIDENTS),
                'Complete resident memory coverage throughout measurement/drain required')
            require(all(any(e['pid'] == ready['pid'] and e['alias'] == alias for e in evidence)
                for alias, ready in raw['ready'].items()), 'Sampled READY PID/alias assignment must remain exact')
        last = sample['time_ns']
    return dict(total_samples=len(raw['board_samples']),
        owned_sampled_peak_mib=max(s['owned_used_mib'] for s in raw['board_samples']),
        monitoring=boundary)


def review(run, execution_path, raw_review_path, output):
    run, out = Path(run).resolve(), Path(output).resolve()
    require(not out.exists(), 'Exclusive review output required')
    plan = audit(run)
    require(plan['split'] == 'training', 'Smoke/validation cannot enter the training evidence review')
    refs = {n: receipt(run/n) for n in ('plan.json', 'context.json', 'raw.json', 'executed-jobs.json', 'attempt.json')}
    raw, context, executed, attempt = (checked(refs[n]) for n in
        ('raw.json', 'context.json', 'executed-jobs.json', 'attempt.json'))
    role = qualify_roles(raw, V6)
    execution_ref, raw_review_ref = receipt(execution_path), receipt(raw_review_path)
    execution = completed_execution(execution_ref, run)
    require(raw['plan'] == refs['plan.json'] and raw['context'] == refs['context.json']
        and raw['executed_jobs'] == refs['executed-jobs.json'], 'Exact original raw bindings required')
    require(attempt['plan'] == refs['plan.json'] and attempt['mode'] == plan['mode'], 'Exact no-retry attempt required')
    require(executed == dict(split=None, contexts=None, limit=None,
        trace=(plan['mode'] == 'kernel'), jobs=plan['jobs']), 'No offered job filtering allowed')
    parsed = checked(raw_review_ref)
    require(parsed['schema'] == 'gsdeploy.device_progress_smoke_CPU_raw_contract_review.v4'
        and parsed['status'] == 'RAW_STRUCTURAL_CONTRACT_ACCEPTED_DEVICE_API_NOT_AUTHENTICATED'
        and parsed['issues'] == [] and parsed['raw'] == refs['raw.json']
        and parsed['qualified_role'] == json.loads(json.dumps(role)), 'Exact successful independent raw/protocol review required')
    cohort = inspect_cohort(plan, raw)
    telemetry = inspect_samples(raw, execution['process_group'])
    sources = strict_references([list(refs.values()), execution_ref, raw_review_ref,
        role['exact_pipeline_sources'], implementation_sources(__file__)])
    require(all(receipt(r['path']) == r for r in sources), 'Evidence/source changed during review')
    result = dict(schema='gsdeploy.profile_training_evidence_review.v1',
        status='PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION',
        role=role, inputs=refs, execution=execution_ref, raw_review=raw_review_ref,
        cohort=cohort, telemetry=telemetry, sources=sources,
        training_feature_extraction_eligible=True, fitted_model_qualified=False,
        independent_prediction_accuracy_validated=False, final_gain_eligible=False,
        GPU_started_by_review=False, validation_labels_read=False,
        limitations=['Training evidence only; no independent service/queue/P95/DeltaU validation.',
            'Natural host timing and profiler-perturbed kernel/stream timing remain distinct domains.',
            'Recorded stream elapsed/query values are source-bound worker measurements; independent native API numeric authentication is separate.',
            'Sampled owned peak is not instantaneous continuous peak. Shutdown memory is outside this explicitly declared monitoring window.',
            'No new global admission cap, changed utility weights or stability qualification.'])
    out.mkdir(parents=True, exist_ok=False)
    save(out/'report.json', result)
    return receipt(out/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--raw-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.run, args.execution, args.raw_review, args.output)))
