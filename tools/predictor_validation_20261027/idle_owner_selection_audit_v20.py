"""Read-only CPU trajectory and training-prefix diagnosis; never changes models."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def trajectory(scheduler_path, forecast_path):
    scheduler, forecast = read(scheduler_path), read(forecast_path)
    resources = forecast['resource_transitions']
    begins = [r['time_ns'] for r in resources]
    assert all(a['until_ns'] == b['time_ns'] for a, b in zip(resources, resources[1:]))
    end = resources[-1]['until_ns']
    first_arrival = {}
    for row in forecast['cohort'].values():
        alias = row['alias']
        first_arrival[alias] = min(first_arrival.get(alias, row['arrival_ns']), row['arrival_ns'])

    def handlers(clock):
        i = bisect_right(begins, clock) - 1
        if i < 0 or clock >= resources[i]['until_ns']:
            raise ValueError('Decision must have an exact half-open forecast resource interval')
        return resources[i]['handler_active_aliases']

    counts, targets, examples = Counter(), Counter(), []
    for query in scheduler['queries']:
        decision = query['decision']
        if decision['kind'] != 'switch':
            continue
        clock, target = query['clock_ns'], decision['target_owner']
        counts['switches'] += 1
        targets[target] += 1
        pending = {r[0] for r in query['pending_frontier']}
        counts['target_not_frontier'] += target not in pending
        counts['target_no_active_handler'] += target not in handlers(clock)
        counts['target_has_no_offered_request_in_whole_cohort'] += target not in first_arrival
        counts['target_has_no_arrival_by_cutoff'] += target not in first_arrival or first_arrival[target] > clock
        counts['host_active_target_not_frontier'] += target in handlers(clock) and target not in pending
        if target not in handlers(clock) and len(examples) < 4:
            examples.append(dict(clock_ns=clock, target=target, pending_frontier=query['pending_frontier'],
                handler_active_aliases=handlers(clock), decision=decision))

    # Exact modeled SERVICE_READY -> SAVE_END spans, intersected with resource
    # intervals. These are envelope hypotheses, never measured physical waste.
    service_spans, active = [], None
    for row in scheduler['transitions']:
        if row['kind'] == 'SERVICE_READY':
            assert active is None
            active = (row['clock_ns'], row['owner'])
        elif row['kind'] == 'SAVE_END' and active is not None:
            service_spans.append((*active, min(row['clock_ns'], end)))
            active = None
    if active is not None:
        service_spans.append((*active, end))
    absent_ns = Counter()
    for begin, owner, stop in service_spans:
        i = max(0, bisect_right(begins, begin) - 1)
        while i < len(resources) and resources[i]['time_ns'] < stop:
            row = resources[i]
            width = max(0, min(stop, row['until_ns']) - max(begin, row['time_ns']))
            if owner not in row['handler_active_aliases']:
                absent_ns[owner] += width
            i += 1

    # A query frontier is certified at that event only. It remains unchanged
    # until the very next engine resource boundary, never beyond it. Restrict
    # to HOLD queries with an already service-ready idle holder.
    transitions = scheduler['transitions']
    transition_times = [r['clock_ns'] for r in transitions]
    blocked_ns, blocked_queries = 0, 0
    for query in scheduler['queries']:
        clock, owner = query['clock_ns'], query['held_owner']
        if query['decision']['kind'] != 'hold' or not query['pending_frontier']:
            continue
        i = bisect_right(transition_times, clock) - 1
        if i < 0 or transitions[i]['kind'] != 'SERVICE_READY' or transitions[i]['owner'] != owner:
            continue
        resource = resources[bisect_right(begins, clock) - 1]
        if resource['time_ns'] != clock:
            raise ValueError('Query/resource event settlement mismatch')
        if owner not in resource['handler_active_aliases']:
            assert owner not in {r[0] for r in query['pending_frontier']}
            blocked_queries += 1
            blocked_ns += resource['until_ns'] - clock
    return dict(scheduler=receipt(scheduler_path), forecast=receipt(forecast_path),
        simulated_window_ns=end, queries=len(scheduler['queries']), starts=len(forecast['starts']),
        finishes=len(forecast['finishes']), first_offered_arrival_ns_by_owner=first_arrival,
        counts=dict(counts), switch_targets=dict(targets),
        modelled_service_ready_envelope_without_handler_ns_by_owner=dict(absent_ns),
        hold_query_witnesses_with_ready_GPU_frontier_and_idle_holder=blocked_queries,
        certified_next_resource_interval_zero_GPU_progress_ns=blocked_ns,
        zero_progress_interval_scope='Subset after HOLD queries only; frontier certified until next engine event; binary modeled rates, not physical GPU idle or wasted time',
        examples=examples)


def training_prefix(supervision_path):
    supervision = read(supervision_path)
    counts, examples, sources = Counter(), [], []
    for audit in supervision['training_audits']:
        for key in ('raw', 'report'):
            assert receipt(audit[key]['path']) == audit[key]
            sources.append(audit[key])
        raw, report = read(audit['raw']['path']), read(audit['report']['path'])
        known = {r['request_id']: r['model'] for r in [*raw.get('priming_request_records', []), *raw['request_records']]}
        sql_ref = next(r for r in report['sources'] if Path(r['path']).suffix == '.sqlite')
        assert receipt(sql_ref['path']) == sql_ref
        sources.append(sql_ref)
        phases = defaultdict(list)
        with sqlite3.connect(Path(sql_ref['path']).as_uri() + '?mode=ro', uri=True) as connection:
            for begin, end, text in connection.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'GS_STAGE:%'"):
                match = re.fullmatch(r'GS_STAGE:(.*):([^:]+):([^:]+)', text)
                if match and match[1] in known and end is not None:
                    phases[known[match[1]]].append((begin, end, match[1], match[2], match[3]))
        observation_ref = report['kernel_ready_observations']
        assert receipt(observation_ref['path']) == observation_ref
        sources.append(observation_ref)
        issued = defaultdict(list)
        for row in read(observation_ref['path'])['rows']:
            if row['issue_end_ns'] is not None:
                issued[known[row['request_id']]].append((row['issue_end_ns'], row['observed_end_ns']))
        rows = [r for r in supervision['transition_observations']
            if r['capture'] == audit['raw']['sha256'] and not r['next_owner_is_pending']]
        for row in rows:
            clock, owner = row['save_end_ns'], row['next_owner']
            observed_phases = [(rid, kind, phase) for begin, end, rid, kind, phase in phases[owner] if begin <= clock < end]
            issued_outstanding = any(begin <= clock < end for begin, end in issued[owner])
            counts['target_not_pending'] += 1
            counts['within_observed_target_NVTX_phase'] += bool(observed_phases)
            counts['issued_API_returned_without_observed_kernel_end_yet'] += issued_outstanding
            counts['neither_phase_nor_measured_issued_kernel_witness'] += not observed_phases and not issued_outstanding
            counts['context_crosses_request_start'] += row['context_crosses_request_start']
            if len(examples) < 6:
                examples.append(dict(capture=row['capture'], job_name=row['job_name'], save_end_ns=clock,
                    target=owner, observed_active_phases=observed_phases,
                    issued_returned_without_kernel_end=issued_outstanding,
                    context_crosses_request_start=row['context_crosses_request_start']))
    assert counts['target_not_pending'] == supervision['counts']['transition_target_not_pending']
    return dict(supervision=receipt(supervision_path), sources=sources, counts=dict(counts), examples=examples,
        semantics='NVTX START is visible by cutoff with no END yet; returned launch API with no kernel END yet is separate from ready frontier. No post-cutoff kernel START is used as a feature.',
        limits=['Absent phase or measured-kernel witnesses do not prove owner idle: handler gaps, copies/memsets, synchronization and other native work remain unclassified.',
            'Phase activity proves an open native call, not CPU-only preparation or physical SM demand.',
            'This audit does not reconstruct exact causal handler lifecycle or train a new predictor.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    base = Path('artifacts/predictor-static-compilation-v16-20261028-cpu01')
    supervision = Path('artifacts/predictor-context-initial-transition-v13-20261028-training02/report.json')
    inputs = [p for horizon in (1000000000, 3000000000) for p in (
        base / f'{horizon}-compiled-scheduler.json', base / f'{horizon}-compiled-forecast.json')]
    inputs.append(supervision)
    before = [receipt(p) for p in inputs]
    rows = [trajectory(inputs[i], inputs[i + 1]) for i in (0, 2)]
    training = training_prefix(supervision)
    assert before == [receipt(p) for p in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    value = dict(schema='gsdeploy.idle_owner_selection_read_only_audit.v20',
        source=receipt(__file__), inputs_before=before, inputs_after=[receipt(p) for p in inputs],
        trajectories=rows, training_prefix=training, GPU_started=False,
        predictor_modified=False, validation_labels_used=False, physical_accuracy_validated=False,
        final_gain_eligible=False)
    output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), trajectories=[{k:v for k,v in r.items() if k not in ('examples','scheduler','forecast')} for r in rows], training_counts=training['counts']), indent=2))


if __name__ == '__main__':
    main()
