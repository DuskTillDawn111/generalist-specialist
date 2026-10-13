"""CPU-only acceptance correction for the native frozen input sample schema.

The original frozen verify.py and failed log remain unchanged. No GPU work,
request edit, predictor fit, or experimental re-selection takes place here.
"""
import ast
from .common import *
from module21.trajectory import choose


def main():
    frozen = read(OUT / 'discovery_protocol.json')
    amendment = read(OUT / 'hardware_identity_amendment.json')
    campaign = read(OUT / 'campaign.json')
    gate = read(OUT / 'module27_gate.json')
    sources = [ref(p) for p in sorted((ROOT / 'module27').glob('*.py'))]
    raw_refs = [ref(OUT / 'raw_runs' / (n + '.json'))
                for n in campaign['executed_B'] + campaign['executed_C']]
    save('analysis_acceptance_amendment.json', dict(
        at_utc=now(), scope='POST_GPU_CPU_VERIFICATION_ONLY',
        cause='Original verifier used item.sample_id; native frozen trajectory uses item.sample.id',
        original_verifier=ref(ROOT / 'module27/verify.py'),
        failed_verification_log=ref(OUT / 'final_verification.log'),
        failed_pipeline_log=ref(OUT / 'recovery_pipeline.log'),
        corrected_verifier=ref(ROOT / 'module27/post_run_verify.py'),
        raw_runs_before_verification=raw_refs,
        GPU_reruns=0, experimental_configuration_unchanged=True,
        utility_or_gate_selection_changed=False))
    for p in (ROOT / 'module27').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    verify(frozen['implementation'] + [frozen[k] for k in
           ['catalog', 'utility_definition', 'utility_reference', 'quality_policy']])
    verify(amendment['implementation'])
    verify(read(OUT / 'consumed_historical_sources.json'))
    assert campaign['B_attempts'] <= 5 and campaign['C_attempts'] <= 6
    assert campaign['B_budget_checks_used'] <= 6 <= 12
    ledger = read(OUT / 'attempt_ledger.json')
    assert len({r['run_id'] for r in ledger}) == len(ledger)
    assert sum(r['stage'] == 'B' for r in ledger) == campaign['B_budget_checks_used']
    assert campaign['initial_pre_inference_identity_check_count'] == 1
    assert ledger[0]['GPU_workers_started'] is False
    checks = ['Original frozen sources, hardware amendment, utility/quality, and strict run budget unchanged']
    verified_requests = 0
    for name in campaign['executed_B'] + campaign['executed_C']:
        raw = read(OUT / 'raw_runs' / (name + '.json'))
        value, observation = arithmetic(raw)
        slot = raw['slot']
        phase = raw.get('phases', {}).get('measurement')
        if not phase:
            assert not observation['eligible'] and observation['U'] is None
            continue
        assert phase['window']['planned_duration_seconds'] == slot['measurement_seconds']
        trace = slot['trajectories']['measurement']
        verify([trace])
        events = read(trace['path'])['events']
        records = phase['request_records']
        assert len(events) == len(records)
        for event, record in zip(events, records):
            assert event['task'] == record['task']
            assert event['item']['pool_id'] == record['pool_id']
            assert event['item']['sample']['id'] == record['sample_id']
            assert abs(event['offset_seconds'] - record['frozen_offset_seconds']) < 1e-12
            assert event['routing_uniform'] == record['routing_uniform']
            assert event['replica_uniform'] == record['replica_uniform']
            assert record['model'] == choose(event, slot['routing'], slot['deployment'])
            if record['status'] == 'completed':
                assert abs(record['e2e_latency_ms'] - (record['completion_time_ns'] - record['scheduled_time_ns']) / 1e6) < 1e-8
            verified_requests += 1
        if observation['eligible']:
            assert raw['valid_session'] and raw['gpu_release']['verified']
            assert len(records) == observation['completed']
            assert observation['safe_memory_mib'] <= BUDGET
            assert observation['sampled_peak_mib'] <= BUDGET
            assert not observation['hard_unstable']
            assert abs(observation['U'] - sum(value[k + '_contribution'] for k in ['T', 'L', 'P', 'M'])) < 1e-12
            assert max(r['completion_time_ns'] for r in raw['warmup']['request_records']) < phase['window']['start_time_ns']
        else:
            assert observation['U'] is None
    checks.append('Every frozen task/input/arrival/route and E2E timestamp, complete cohort U, safe memory and stability')
    development = read(OUT / 'exploratory_result.json')
    independent = read(OUT / 'independent_result.json') if (OUT / 'independent_result.json').exists() else None
    assert bool(independent) == bool(development['Stage_C_permitted']) or gate.get('failure')
    if independent:
        assert set(read(OUT / 'frozen_positive_candidate.json')['rates']) == set(frozen['refs'])
        assert set(r['seed'] for r in ledger if r['stage'] == 'C') <= set(frozen['independent_protocol']['seeds'])
        assert not (set(r['seed'] for r in ledger if r['stage'] == 'C') & set(r['seed'] for r in ledger if r['stage'] == 'B'))
    assert (OUT / 'optimizer_autonomous_decision.json').exists() == bool(independent and independent['Stage_D_permitted'])
    checks.append('Development excluded from independent evidence, conditional optimizer gate, and no additional seeds')
    before = read(OUT / 'historical_metadata_before.json')
    after = protected_metadata()
    assert all(after.get(path) == value for path, value in before.items())
    verify(raw_refs)
    checks.append('Historical metadata/scientific SHA256 and GPU raw run SHA256 unchanged')
    required = ['stage_a_evidence_audit.json', 'candidate_screening.csv', 'historical_service_evidence.csv',
                'discovery_protocol.json', 'frozen_positive_candidate.json', 'gpu_discovery_runs.csv',
                'exploratory_utility_comparison.csv', 'per_task_latency.csv', 'worker_queue_stability.csv',
                'utility_decomposition.csv', 'GPU_thermal_state.csv', 'module27_gate.json',
                'module27_summary.md', 'historical_integrity.json', 'analysis_acceptance_amendment.json',
                'negative_split_attribution.json', 'post_run_evidence_report.log']
    assert all((OUT / name).exists() for name in required)
    gate['CPU_acceptance_passed'] = True
    gate['CPU_acceptance_amendment'] = ref(OUT / 'analysis_acceptance_amendment.json')
    save('module27_gate.json', gate)
    save('execution_source_manifest.json', dict(sources=sources, results=[ref(OUT / name) for name in required]))
    save('final_verification.json', dict(passed=True, checks=checks, verified_offered_requests=verified_requests,
         history_files=len(before), all_negative_results_retained=True, no_next_Module=True, remaining_work=False,
         original_failed_verification_preserved=True, amendment=ref(OUT / 'analysis_acceptance_amendment.json')))
    save('pipeline_completed.json', dict(at_utc=now(), status='COMPLETED_AND_CPU_VERIFIED', gate=gate['status'],
         final_verification=ref(OUT / 'final_verification.json'), GPU_stopped=True, no_next_Module=True))
    print('MODULE27_VERIFIED_AND_STOPPED', gate['status'], 'requests', verified_requests, flush=True)


if __name__ == '__main__':
    main()
