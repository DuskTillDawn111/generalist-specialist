"""CPU counterexamples for a proposed contract, NOT a resume implementation.

Uses the current main engine, real numeric SAVE controller, stream observer,
and original scorer. Test fixtures are explicit; nothing here measures GPU
accuracy or introduces a scheduler policy or a device-progress cursor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.metrics import percentile, score_forecast
from gsdeploy.forecasting.observed_state import (
    advance_observed_state, feedback_from_runtime_records, new_observed_state)
from tests.test_device_progress_observer_v1 import DeviceObserverTests, TensorShape
from tests.test_learned_lifecycle_scheduler_v49_independent_review import IndependentReviewTests
from tests.test_owner_lifecycle_v21 import EngineTests


ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve()
        if path.suffix == '.py' and path.is_relative_to(ROOT):
            paths.add(path)
    return [{'path': str(p), 'size_bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]


def run_audit():
    before = sources()
    checks = []
    numeric = IndependentReviewTests()
    numeric.setUp()
    engine = EngineTests()
    engine.setUp()
    try:
        # Equal present lifecycle/frontier/age does not identify past risk.
        high = numeric.scheduler(lower_rate_on_peer=True)
        low = numeric.scheduler(lower_rate_on_peer=True)
        high.decide(numeric.view(0, handlers=('text#0',)))
        low.decide(numeric.view(0, handlers=('text#0', 'text#1')))
        now = 1000
        for scheduler in (high, low):
            scheduler.decide(numeric.view(now, handlers=('text#0', 'text#1')))
        a, b = high.oracle.hazard, low.oracle.hazard
        assert a['hazard_budget'] == b['hazard_budget']
        assert a['consumed_hazard'] > b['consumed_hazard']
        assert a['absolute_save_candidate_ns'] < b['absolute_save_candidate_ns']
        checks.append(dict(id='same_present_different_SAVE_history', status='PASS',
            cutoff_ns=now, same_current_lifecycle=True, same_empty_frontier=True,
            same_restore_origin_ns=0, hazard_budget=a['hazard_budget'],
            consumed_hazard=[a['consumed_hazard'], b['consumed_hazard']],
            SAVE_candidate_ns=[a['absolute_save_candidate_ns'], b['absolute_save_candidate_ns']],
            consequence='Current queue/lifecycle is insufficient; preserve sealed cumulative budget state'))

        scheduler = numeric.scheduler()
        crossing = scheduler.decide(numeric.view(0)).next_transition_ns
        scheduler.decide(numeric.view(crossing))
        action = copy.deepcopy(scheduler.pending_switch)
        events = copy.deepcopy(scheduler.context_events)
        try:
            scheduler._accept(action, scheduler.snapshot(numeric.view(crossing)))
        except UnknownEvidence as exc:
            assert 'PENDING_CONTEXT_SWITCH_ALREADY_EXISTS' in str(exc)
            duplicate_error = str(exc)
        else:
            raise AssertionError('A second handoff was silently accepted')
        scheduler.decide(numeric.view(crossing + 1))
        assert scheduler.pending_switch == action and scheduler.context_events == events
        checks.append(dict(id='pending_handoff_is_absolute_and_immutable', status='PASS',
            SAVE_ns=crossing, cutoff_ns=crossing + 1,
            pending_action=action.__dict__, remaining_absolute_events=events,
            duplicate_rejected=duplicate_error,
            consequence='Do not recreate full handoff duration at the rolling cutoff'))

        rows = engine.rows()
        result = engine.engine().forecast(rows, 0, 30, drain_deadline_ns=1000)
        first, second = result['finishes']['b-0'], result['finishes']['b-4']
        assert first['handler_end_ns'] < first['finish_ns'] < first['worker_next_ready_ns']
        assert second['start_ns'] >= first['worker_next_ready_ns']
        assert len(result['cohort']) == len(rows)
        assert any(row['arrival_ns'] == 17 for row in result['cohort'].values())
        checks.append(dict(id='handler_END_receipt_READY_are_separate', status='PASS',
            first_request=first, next_START_ns=second['start_ns'], original_cohort_size=len(rows),
            consequence='A parent completion contributes to latency while its owner can remain unavailable'))

        for option in ({'checkpoint': {}}, {'pause_after_event_count': 1}, {'pause_at_or_after_ns': 1}):
            try:
                engine.engine().forecast(rows, 0, 30, **option)
            except ValueError as exc:
                assert 'checkpoint state is not implemented' in str(exc)
            else:
                raise AssertionError('Current main engine unexpectedly accepted a checkpoint')
        checks.append(dict(id='main_v21_does_not_implement_resume', status='PASS',
            all_three_checkpoint_pause_entrypoints_reject=True,
            consequence='This audit and design must not be reported as implemented continuation'))

        observer_fixture = DeviceObserverTests()
        observer_fixture.setUp()
        with observer_fixture.observe() as observer:
            observer_fixture.owner(TensorShape(1, 20))
            unfinished = copy.deepcopy(observer.queries[-1])
            observer_fixture.cuda.complete = True
            observer.poll()
            finished = copy.deepcopy(observer.queries[-1])
        assert not unfinished['end_event_complete'] and finished['end_event_complete']
        for query in (unfinished, finished):
            assert query['in_flight_remaining_fraction'] is None
            assert query['physical_SM_service_ns'] is None
            assert query['completion_time_ns'] is None
            assert query['all_request_GPU_work_complete'] is False
        records = [dict(request_id='req', model='owner', execution_events=[
            dict(request_id='req', perf_ns=1, received_time_ns=2),
            dict(finished, received_time_ns=finished['perf_ns'] + 10)])]
        cutoff = finished['perf_ns'] + 10
        feed = feedback_from_runtime_records(records, 'CPU-contract-fixture', cutoff)
        assert feed[-1]['kind'] == 'PHASE'
        try:
            advance_observed_state(new_observed_state({'owner': {}}, 'CPU-contract-fixture'),
                                   cutoff, feedback=feed)
        except ValueError as exc:
            assert 'Unknown phase edge' in str(exc)
            adapter_error = str(exc)
        else:
            raise AssertionError('Legacy PHASE adapter accepted device query as phase progress')
        early = feedback_from_runtime_records(records, 'CPU-contract-fixture', cutoff - 1)
        assert len(early) == 1
        checks.append(dict(id='stream_event_scope_and_legacy_adapter_gap', status='PASS',
            unfinished_query=unfinished, completed_query=finished,
            legacy_adapter_error=adapter_error, query_not_visible_before_reader_receipt=True,
            consequence='Add a typed evidence ledger; do not infer all-DAG completion, holder, or work fraction'))

        # Invoke the actual six-term scorer. Prefix P95 is not additive; missing
        # prefix energy may support a common-prefix comparison, never absolute U.
        aliases = {'owner': {'family': 'f', 'supported_tasks': ['text']}}
        cohort = {f'r{i}': dict(request_id=f'r{i}', alias='owner', task='text', pool_id='p',
                                arrival_ns=0) for i in range(4)}
        history = [dict(cohort['r0'], completion_ns=1_000_000),
                   dict(cohort['r1'], completion_ns=100_000_000)]
        forecast = dict(cohort={rid: row for rid, row in cohort.items() if rid not in {'r0', 'r1'}},
            finishes={rid: dict(cohort[rid], finish_ns=latency*1_000_000)
                      for rid, latency in [('r2', 200), ('r3', 300)]},
            peak_memory_mib=20., board_energy_joules=3.)
        kwargs = dict(phase_start_ns=0, now_ns=150_000_000, deadline_ns=400_000_000,
            memory_budget_mib=100., energy_reference_watts=10.,
            observed_peak_memory_mib=40., memory_safe_factor=1.1)
        complete = score_forecast(forecast, history, aliases, {'text': 10.}, {'text': 100.},
            {'f': {'text': .2}}, observed_board_energy_joules=1., **kwargs)
        unknown = score_forecast(forecast, history, aliases, {'text': 10.}, {'text': 100.},
            {'f': {'text': .2}}, observed_board_energy_joules=None, **kwargs)
        assert complete['terms']['T'] == 0.
        assert abs(complete['terms']['L'] - 1.5025) < 1e-12
        assert abs(complete['terms']['P'] - 2.85) < 1e-12
        assert abs(complete['terms']['M'] - .44) < 1e-12
        assert complete['terms']['E'] == 1. and complete['full_U'] is not None
        assert unknown['full_U'] is None
        assert percentile([1., 100., 200., 300.]) != (
            percentile([1., 100.]) + percentile([200., 300.])) / 2
        checks.append(dict(id='whole_cohort_six_term_prefix_accounting', status='PASS',
            six_terms=complete['terms'], whole_cohort_p95_ms=285.,
            average_of_prefix_suffix_p95_ms=(percentile([1., 100.]) + percentile([200., 300.])) / 2,
            full_U=complete['full_U'], missing_prefix_energy_full_U=unknown['full_U'],
            consequence='Merge original identities/times, sum nonoverlapping energy, take whole-window memory max'))
    finally:
        engine.doCleanups()
        numeric.doCleanups()
    after = sources()
    assert before == after, 'Audit source closure changed while running'
    return dict(schema='gsdeploy.lifecycle_resume_contract_audit.v1', status='PASS',
        CPU_fixture_only=True, GPU_started=False, main_resume_implemented=False,
        real_observation_assimilation_implemented=False, physical_accuracy_validated=False,
        test_count=len(checks), checks=checks, source_closure=before,
        source_bytes_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], CPU_checks=report['test_count'],
        report=str(args.output.resolve()), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
