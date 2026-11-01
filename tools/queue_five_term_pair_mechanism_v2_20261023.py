"""Read-only matched-run timing diagnosis; no counterfactual latency claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    p = .95 * (len(values) - 1)
    lo = int(p)
    hi = min(lo + 1, len(values) - 1)
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p95=values[lo] + (values[hi] - values[lo]) * (p - lo),
                maximum=max(values), minimum=min(values), total=sum(values))


def components(row):
    route = row['routing_decision']
    decision_ms = route['decision_compute_ns'] / 1e6
    worker = row['worker_response']
    handler = worker['worker_latency_ms']
    service = worker['service_ms']
    schedule = row['schedule_delay_ms']
    queue = row['worker_queue_wait_ms']
    return dict(
        e2e_ms=row['e2e_latency_ms'], schedule_delay_ms=schedule,
        predecision_delay_ms=(route['decision_time_ns'] - row['scheduled_time_ns']) / 1e6,
        decision_compute_ms=decision_ms,
        postdecision_to_enqueue_ms=(row['enqueued_time_ns'] - route['decision_time_ns']) / 1e6 - decision_ms,
        host_send_queue_ms=(row['actual_submit_time_ns'] - row['enqueued_time_ns']) / 1e6,
        worker_queue_ms=queue, handler_ms=handler, handler_service_ms=service,
        handler_wrapper_ms=handler - service,
        return_transport_clock_residual_ms=row['e2e_latency_ms'] - schedule - queue - handler,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite ' + str(a.output))
    snapshot = json.loads(a.snapshot.read_bytes())
    paired = next(p for p in snapshot['summary']['paired']
                  if p['strategy'] == 'coupled_spill' and p['both_evidence_valid_eligible']
                  and (a.seed is None or p['seed'] == a.seed))
    seed = paired['seed']
    observations = {o['role']: o for o in snapshot['observations'] if o['seed'] == seed}
    raws = {}
    for role in ('coupled_spill', 'matched_no_spill'):
        observation = observations[role]
        if not observation['verified'] or receipt(observation['raw']['path']) != observation['raw']:
            raise SystemExit('Unverified or changed raw: ' + role)
        raws[role] = json.loads(Path(observation['raw']['path']).read_bytes())
    on, off = raws['coupled_spill'], raws['matched_no_spill']
    on_rows = {r['request_id']: r for r in on['request_records']}
    off_rows = {r['request_id']: r for r in off['request_records']}
    cohort_fields = ('request_id', 'task', 'pool_id', 'sample_id', 'scheduled_offset_seconds')
    if set(on_rows) != set(off_rows) or any(
            [r[k] for k in cohort_fields] != [off_rows[rid][k] for k in cohort_fields]
            for rid, r in on_rows.items()):
        raise SystemExit('Cohort mismatch')
    spilled = {rid for rid, r in on_rows.items()
               if r['task'] == 'text' and on['replica_to_family'][r['model']].startswith('generalist')}
    groups = {
        'all': set(on_rows),
        'text_spilled_in_on': spilled,
        'text_specialist_in_on': {rid for rid, r in on_rows.items() if r['task'] == 'text'} - spilled,
        **{task: {rid for rid, r in on_rows.items() if r['task'] == task}
           for task in ('text', 'image', 'asr', 'ocr')},
    }
    timing = {}
    for name, ids in groups.items():
        ca, cb = [components(on_rows[i]) for i in ids], [components(off_rows[i]) for i in ids]
        timing[name] = dict(n=len(ids), on={k: summary([r[k] for r in ca]) for k in ca[0]},
                            off={k: summary([r[k] for r in cb]) for k in cb[0]},
                            matched_on_minus_off={k: summary([u[k] - v[k] for u, v in zip(ca, cb)]) for k in ca[0]})
    forecasts = {}
    spill_records = []
    for group in ('text_spilled_in_on', 'text_specialist_in_on'):
        entries = []
        for rid in sorted(groups[group]):
            r = on_rows[rid]
            decision = r['routing_decision']['decision']
            candidate = next((e for e in decision.get('audit', [])
                              if e['alias'] == r['model'] and e.get('score') is not None), None)
            if candidate is None:
                continue
            entry = dict(request_id=rid, actual_handler_service_ms=r['worker_response']['service_ms'],
                         actual_e2e_ms=r['e2e_latency_ms'],
                         decision_compute_ms=r['routing_decision']['decision_compute_ns'] / 1e6,
                         predicted={s: {k: e[k] for k in ('predicted_own_service_ms', 'predicted_completion_ms')}
                                    for s, e in candidate['scenarios'].items()},
                         scenario_delta_cost_vs_selected_specialist=candidate.get('scenario_delta_cost_vs_selected_specialist'),
                         later_arrivals_before_completion=sum(
                             x['scheduled_time_ns'] > r['routing_decision']['decision_time_ns']
                             and x['scheduled_time_ns'] < r['completion_time_ns'] for x in on_rows.values()))
            entries.append(entry)
            if rid in spilled:
                spill_records.append(entry)
        stats = dict(n=len(entries))
        for scenario in ('mean', 'p90'):
            stats[scenario] = dict(
                own_service_prediction_ms=summary([e['predicted'][scenario]['predicted_own_service_ms'] for e in entries]),
                service_actual_minus_prediction_ms=summary([e['actual_handler_service_ms'] - e['predicted'][scenario]['predicted_own_service_ms'] for e in entries]),
                service_actual_over_prediction=summary([e['actual_handler_service_ms'] / e['predicted'][scenario]['predicted_own_service_ms'] for e in entries]),
                completion_actual_minus_prediction_ms=summary([e['actual_e2e_ms'] - e['predicted'][scenario]['predicted_completion_ms'] for e in entries]),
                service_exceedance_count=sum(e['actual_handler_service_ms'] > e['predicted'][scenario]['predicted_own_service_ms'] for e in entries),
                completion_exceedance_count=sum(e['actual_e2e_ms'] > e['predicted'][scenario]['predicted_completion_ms'] for e in entries))
        forecasts[group] = stats
    intervals = [(r['worker_response']['start_ns'], r['worker_response']['end_ns']) for rid, r in on_rows.items() if rid in spilled]
    overlaps = {}
    for task in ('text', 'image', 'asr', 'ocr'):
        rows = [r for r in on_rows.values() if r['task'] == task and r['request_id'] not in spilled]
        for flag in (False, True):
            ids = [r['request_id'] for r in rows if any(
                min(r['worker_response']['end_ns'], end) > max(r['worker_response']['start_ns'], start)
                for start, end in intervals) == flag]
            overlaps[task + ('_overlaps_G2_text_handler' if flag else '_no_G2_text_handler_overlap')] = dict(
                n=len(ids), on_service_ms=summary([on_rows[rid]['worker_response']['service_ms'] for rid in ids]),
                off_matched_service_ms=summary([off_rows[rid]['worker_response']['service_ms'] for rid in ids]),
                service_matched_on_minus_off_ms=summary([on_rows[rid]['worker_response']['service_ms'] - off_rows[rid]['worker_response']['service_ms'] for rid in ids]))
    fields = ('predecision_delay_ms', 'decision_compute_ms', 'postdecision_to_enqueue_ms', 'host_send_queue_ms',
              'worker_queue_ms', 'handler_service_ms', 'handler_wrapper_ms', 'return_transport_clock_residual_ms')
    errors = [abs(components(r)['e2e_ms'] - sum(components(r)[k] for k in fields))
              for raw in raws.values() for r in raw['request_records']]
    result = dict(schema='gsdeploy.five_term_matched_timing_diagnosis.v1', seed=seed,
                  source_snapshot=receipt(a.snapshot), analysis_source=receipt(__file__),
                  source_raws={role: observations[role]['raw'] for role in raws}, same_cohort=True,
                  paired_utility=paired, timing=timing, selected_path_forecasts=forecasts,
                  spill_forecasts_and_outcomes=spill_records, handler_overlap_diagnostic=overlaps,
                  timing_decomposition_max_abs_error_ms=max(errors),
                  interpretation=dict(
                      matched_groups='Grouping uses realized on-arm routing; matched off observations are a separate system run, not same-execution unchosen-path counterfactuals.',
                      forecast='Forecast errors compare the chosen path prediction with that same chosen path outcome. Future arrivals and later system transitions can change outcomes.',
                      decision_timer='Includes queue snapshot, feedback collection/observation and choose under runtime.lock. It excludes waiting to acquire runtime.lock and excludes enqueue/send after the timer.',
                      queue='worker_queue_ms is submit-wall-clock to handler-start; includes pipe/serialization and worker waiting, not a pure physical GPU queue.',
                      handler='Handler service is host wall duration reported by the handler, not isolated GPU kernel time; service difference cannot uniquely identify GPU interference.',
                      residual='E2E minus submit delay, worker queue and handler; includes return transport and host collection plus wall/perf-clock drift.',
                      causality='Descriptive one-seed paired decomposition only. Random-order or intervention evidence is needed to attribute net utility loss to dispatcher overhead or interference.'),
                  GPU_started=False, frozen_sources_modified=False)
    a.output.mkdir(parents=True)
    (a.output / 'diagnosis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(a.output), timing=timing['all'], forecasts=forecasts), ensure_ascii=False))


if __name__ == '__main__':
    main()
