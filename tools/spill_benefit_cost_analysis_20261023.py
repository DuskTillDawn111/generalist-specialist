"""Read-only benefits and costs in the two frozen primary native pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/queue-five-term-study-20261023'


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def mean(values):
    return statistics.fmean(values) if values else None


def comparisons(on, off, ids, field):
    va = [on[i][field] if field != 'service_ms' else on[i]['worker_response'][field] for i in ids]
    vb = [off[i][field] if field != 'service_ms' else off[i]['worker_response'][field] for i in ids]
    delta = [a - b for a, b in zip(va, vb)]
    wins = [-d for d in delta if d < 0]
    losses = [d for d in delta if d > 0]
    return dict(n=len(ids), on_mean_ms=mean(va), off_mean_ms=mean(vb),
                on_minus_off_mean_ms=mean(delta), on_minus_off_total_ms=sum(delta),
                faster_count=len(wins), equal_count=sum(d == 0 for d in delta), slower_count=len(losses),
                faster_mean_gain_ms=mean(wins), faster_maximum_gain_ms=max(wins, default=None),
                faster_total_gain_ms=sum(wins), slower_mean_cost_ms=mean(losses),
                slower_maximum_cost_ms=max(losses, default=None), slower_total_cost_ms=sum(losses))


def main():
    out = BASE / 'spill-benefit-cost-analysis-v1'
    if out.exists():
        raise SystemExit('Refusing to overwrite ' + str(out))
    terminal_path = BASE / 'independent-runtime-audit-stream-v1/terminal.json'
    terminal = json.loads(terminal_path.read_bytes())
    assert terminal['verified']
    observations = {o['label']: o for o in terminal['observations']}
    results = []
    for pair in terminal['summary']['paired'][:2]:
        seed = pair['seed']
        assert pair['both_evidence_valid_eligible'] and pair['same_cohort']
        audited = [observations[f'{role}-seed{seed}'] for role in ('coupled_spill', 'matched_no_spill')]
        raws = []
        for o in audited:
            assert o['verified'] and receipt(o['raw']['path']) == o['raw']
            raws.append(json.loads(Path(o['raw']['path']).read_bytes()))
        a, b = raws
        on, off = [{r['request_id']: r for r in raw['request_records']} for raw in raws]
        assert set(on) == set(off)
        assert all([r[k] for k in ('task', 'pool_id', 'sample_id', 'scheduled_offset_seconds')]
                   == [off[i][k] for k in ('task', 'pool_id', 'sample_id', 'scheduled_offset_seconds')]
                   for i, r in on.items())
        text = {i for i, r in on.items() if r['task'] == 'text'}
        spill = {i for i in text if a['replica_to_family'][on[i]['model']].startswith('generalist')}
        resident = text - spill
        ids_by_group = dict(spilled_text=spill, remaining_specialist_text=resident,
                            remaining_specialist_text_same_alias={i for i in resident if on[i]['model'] == off[i]['model']},
                            all_measurement_requests=set(on),
                            **{t: {i for i, r in on.items() if r['task'] == t} for t in ('image', 'asr', 'ocr')})
        group_metrics = {g: {f: comparisons(on, off, sorted(ids), f)
                             for f in ('worker_queue_wait_ms', 'service_ms', 'e2e_latency_ms')}
                         for g, ids in ids_by_group.items()}
        replicas = {}
        for alias in sorted(k for k, family in a['replica_to_family'].items() if family == 'text'):
            aa = [r for i, r in on.items() if i in text and r['model'] == alias]
            bb = [r for i, r in off.items() if i in text and r['model'] == alias]
            replicas[alias] = dict(on_requests=len(aa), off_requests=len(bb), request_reduction=len(bb) - len(aa),
                                   on_sum_handler_service_ms=sum(r['worker_response']['service_ms'] for r in aa),
                                   off_sum_handler_service_ms=sum(r['worker_response']['service_ms'] for r in bb))
        ea, eb = a['energy_evidence'], b['energy_evidence']
        config = a['config']
        e_run_delta = (ea['run_joules'] - eb['run_joules']) / config['horizon_seconds'] / config['energy_reference_watts']
        e_start_delta = (ea['startup_joules'] - eb['startup_joules']) / config['startup_amortization_seconds'] / config['energy_reference_watts']
        energy = dict(on_window_joules=ea['run_joules'], off_window_joules=eb['run_joules'],
                      window_joule_change=ea['run_joules'] - eb['run_joules'],
                      on_startup_joules=ea['startup_joules'], off_startup_joules=eb['startup_joules'],
                      startup_joule_change=ea['startup_joules'] - eb['startup_joules'],
                      normalized_E_window_component_change=e_run_delta,
                      normalized_E_amortized_startup_component_change=e_start_delta,
                      normalized_E_total_change=e_run_delta + e_start_delta,
                      contribution_to_delta_U=-.2 * (e_run_delta + e_start_delta),
                      on_full_cohort_joules=audited[0]['diagnostics']['energy_full_cohort_diagnostic_joules'],
                      off_full_cohort_joules=audited[1]['diagnostics']['energy_full_cohort_diagnostic_joules'])
        assert abs(energy['normalized_E_total_change'] - pair['term_deltas_spill_minus_control']['E']) < 1e-12
        completion = dict(offered_requests=len(on),
                          on_completed_within_window=audited[0]['diagnostics']['completed_within_window'],
                          off_completed_within_window=audited[1]['diagnostics']['completed_within_window'],
                          on_completed_during_drain=audited[0]['diagnostics']['completed_during_drain'],
                          off_completed_during_drain=audited[1]['diagnostics']['completed_during_drain'],
                          on_drain_seconds=audited[0]['diagnostics']['drain_seconds'],
                          off_drain_seconds=audited[1]['diagnostics']['drain_seconds'],
                          T_change=pair['term_deltas_spill_minus_control']['T'])
        examples = []
        for i in sorted(spill, key=lambda i: on[i]['e2e_latency_ms'] - off[i]['e2e_latency_ms'])[:5]:
            examples.append(dict(request_id=i, on_alias=on[i]['model'], off_alias=off[i]['model'],
                                 on_e2e_ms=on[i]['e2e_latency_ms'], off_e2e_ms=off[i]['e2e_latency_ms'],
                                 faster_by_ms=off[i]['e2e_latency_ms'] - on[i]['e2e_latency_ms'],
                                 on_worker_queue_ms=on[i]['worker_queue_wait_ms'], off_worker_queue_ms=off[i]['worker_queue_wait_ms']))
        results.append(dict(seed=seed, raw_receipts=[o['raw'] for o in audited], same_cohort=True,
                            actual_text_spills=len(spill), text_requests=len(text), specialist_replicas=replicas,
                            grouped_metrics=group_metrics, window_completion=completion, energy=energy,
                            formal_paired_utility=pair, faster_spill_examples=examples))
    result = dict(schema='gsdeploy.spill_benefit_cost_analysis.v1', source=receipt(__file__),
                  audit_terminal=receipt(terminal_path), results=results,
                  scope=dict(grouped_metrics='Descriptive comparison of matched inputs/arrival positions across two complete system runs. Groups are selected by on-arm realized routing; they do not identify same-execution unchosen-path causal effects.',
                             unspilled_faster='Some matched unspilled requests improve, but noise, changed alias routing and changed execution history prevent attributing those individual gains to spill without intervention evidence.',
                             queue='Submit to worker handler start includes transport/worker serialization; not a physical GPU queue.',
                             empty_generalist='Host-visible empty generalist queue does not certify an idle physical GPU.',
                             energy='Formal E includes 600s window mean power plus startup amortized over3600s. Full-cohort energy includes drain but excludes startup and is diagnostic only.',
                             utility='Only workload-weighted normalized full-run term contributions are formal utility gains/costs. Unweighted conditional-group means must not be presented as utility contributions.'),
                  frozen_sources_modified=False, GPU_started=False)
    out.mkdir()
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
