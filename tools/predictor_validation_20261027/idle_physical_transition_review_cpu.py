"""Audit frozen spill_high idle-target failure and v51 explicit extrapolation."""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from . import projected_handler_bridge_v44 as bridge
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .learned_availability_forecast_v20 import verify_sources
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import OwnerLifecycle
from .projected_handler_context_v42 import ObservableContextCandidate, OWNERS
from .unknown_preflight_archive_v1 import save_synced


def tally_stream(reference, kind):
    verify_sources([reference])
    counts = Counter()
    holders = defaultdict(Counter)
    all_rows = 0
    with gzip.open(reference['path'], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            all_rows += 1
            features = row['shared_left_features' if kind == 'risk' else 'features']
            mask = features['handler_active']
            if set(mask) != set(OWNERS) or any(type(v) is not bool for v in mask.values()):
                raise ValueError('Complete projected four-owner mask required')
            if any(mask.values()):
                continue
            count = Counter(rows=1)
            if kind == 'risk':
                width = row['end_ns']-row['start_ns']
                if (width <= 0 or width != row['age_exit_ns']-row['age_entry_ns']
                        or row['held_owner'] != features['held_owner']
                        or row['exposure_qualified'] is not True):
                    raise ValueError('Qualified consistent positive attained-age risk required')
                count.update(exposure_ns=width, physical_SAVE=int(row['physical_SAVE_END_observed']))
                count['boundary_near_any_handler_END_or_BEGIN'] = int(any(
                    v is True for v in row['handler_boundary_proximity_diagnostic'].values()))
                count['exact_native_handler_projection_claimed'] = int(row['handler_projection_is_exact_native_fact'])
            else:
                count.update(physical_mark=1, same_owner=int(row['same_owner']))
            counts.update(count)
            holders[features['held_owner']].update(count)
    verify_sources([reference])
    return dict(total_stream_rows=all_rows, all_idle=dict(counts),
        by_held_owner={owner:dict(holders[owner]) for owner in OWNERS})


def run(output):
    root = Path(__file__).resolve().parents[2]
    source = root/'tools/predictor_validation_20261027'
    out = Path(output).absolute()
    out.mkdir(parents=True, exist_ok=False)
    forecast_path = root/'artifacts/predictor-projected-handler-v44-20261028-preflight01/shared_spill_high-forecast.json.gz'
    forecast_ref = receipt(forecast_path)
    if forecast_ref['sha256'] != 'f611d35a6dbde3e9c2ce0ee5315db7a19143bc535659ebfc067fd7e6fd886101':
        raise ValueError('Original frozen spill_high forecast required')
    with gzip.open(forecast_path, 'rb') as stream:
        original_bytes = stream.read()
    if (len(original_bytes) != 1007558 or hashlib.sha256(original_bytes).hexdigest() !=
            '847d825ea885b0471a761fb94edeb23e64b9ca368ff2dc69d1d9a089727c8a4b'):
        raise ValueError('Full gzip CRC, length and digest required')
    forecast = json.loads(original_bytes)
    scheduler = forecast['scheduler_diagnostic']
    oracle = scheduler['oracle']
    hazard = oracle['latest_cumulative_hazard']
    now = oracle['current_lifecycle_clock_ns']
    if forecast['reason'] != 'NO_VISIBLE_ACTIVE_REFERENCE_TARGET' or now != hazard['absolute_save_candidate_ns']:
        raise ValueError('Exact-time original idle SAVE rejection required')
    states = {owner:OwnerLifecycle(**values) for owner,values in oracle['current_lifecycle'].items()}
    if any(state.active_service_eligible is not False for state in states.values()):
        raise ValueError('Complete known idle engine-owned-work domain required')
    model_ref = forecast['configuration']['observable_context_model']
    verify_sources([model_ref])
    model = json.loads(Path(model_ref['path']).read_text())
    candidate = ObservableContextCandidate(model)
    previous = oracle['decisions'][-1]['held_owner']
    features = bridge.feature_projection(scheduler['held_owner'], previous, hazard['restore_age_ns'], states)
    physical = candidate.next_owner(features)
    # Independently rebuild the feature vector and stable softmax from saved
    # coefficients, without calling the production vector/prediction helpers.
    log_age = math.log1p(features['context_age_ns']/1e6)
    values = dict(intercept=1., log_age_ms=log_age, log_age_ms_squared=log_age**2)
    values['held:'+features['held_owner']] = 1.
    values['held_log_age:'+features['held_owner']] = log_age
    values['previous:'+str(previous)] = 1.
    values['held_handler:'+features['held_owner']+':0000'] = 1.
    fitted = model['fitted']['mark']
    vector = [(values.get(name,0.)-center)/scale for name,center,scale in
        zip(fitted['feature_names'],fitted['center'],fitted['scale'])]
    logits = [sum(x*row[index] for x,row in zip(vector,fitted['coefficients'])) for index in range(4)]
    weights = [math.exp(value-max(logits)) for value in logits]
    probabilities = {owner:weight/sum(weights) for owner,weight in zip(OWNERS,weights)}
    if any(abs(probabilities[owner]-physical['physical_owner_probabilities'][owner])>1e-12 for owner in OWNERS):
        raise ValueError('Independent physical softmax reconstruction failed')
    mark = object.__new__(bridge.ObservableReferenceMark)
    mark.candidate, mark.states, mark.clock = candidate, states, now
    try:
        mark.predict_given_save_now(held_owner=features['held_owner'], previous_held_owner=previous,
            context_age_ns=features['context_age_ns'], pending_frontier=[])
    except UnknownEvidence as exc:
        if str(exc) != forecast['reason']:
            raise
    else:
        raise ValueError('Frozen bridge must reproduce exact idle rejection')
    training = []
    refs = [forecast_ref, model_ref]
    combined = Counter()
    by_holder = defaultdict(Counter)
    for capture, entry in model['captures'].items():
        verify_sources([entry['report']])
        report = json.loads(Path(entry['report']['path']).read_text())
        risk, marks = tally_stream(report['risk_chronology'],'risk'), tally_stream(report['mark_records'],'mark')
        if risk['total_stream_rows'] != report['counts']['all_paired_intervals']:
            raise ValueError('Complete source risk population required')
        training.append(dict(route=entry['route'], capture=capture, report=entry['report'], risk=risk, mark=marks))
        combined.update(risk['all_idle'])
        combined['physical_mark'] += marks['all_idle'].get('physical_mark',0)
        for owner,count in risk['by_held_owner'].items():
            by_holder[owner].update(count)
        refs.extend([entry['report'],report['risk_chronology'],report['mark_records']])
    if combined['physical_SAVE'] != 0 or combined['physical_mark'] != 0:
        raise ValueError('Reviewed all-idle source count changed')
    modules = ['tests.test_projected_handler_idle_bridge_diagnostic',
        'tests.test_idle_physical_mark_bridge_v51_independent_review']
    refs = strict_references([refs, implementation_sources(source/'projected_handler_preflight_v52.py'),
        receipt(__file__), [receipt(root.joinpath(*module.split('.')).with_suffix('.py')) for module in modules]])
    verify_sources(refs)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestLoader().loadTestsFromNames(modules))
    (out/'tests.txt').write_text(stream.getvalue())
    verify_sources(refs)
    next_arrival = min(row['arrival_ns'] for row in forecast['full_public_cohort'] if row['arrival_ns']>now)
    report = dict(schema='gsdeploy.idle_physical_transition_independent_CPU_review.v1',
        status='PASS' if result.wasSuccessful() else 'FAIL', sources=refs,
        original_forecast=forecast_ref, full_gzip_CRC_hash_length_verified=True,
        original_uncompressed_bytes=len(original_bytes),
        actual_diagnostic=dict(reason=forecast['reason'], clock_ns=now,
            SAVE_candidate_ns=hazard['absolute_save_candidate_ns'], late_by_ns=0,
            held_owner=scheduler['held_owner'], previous_held_owner=previous,
            restore_start_ns=scheduler['restore_start_ns'], context_age_ns=features['context_age_ns'],
            lifecycle=oracle['current_lifecycle'], frontier=hazard['state']['frontier'],
            eligible_physical_probability_mass=0., features=features,
            physical_mark=physical, independent_softmax_probabilities=probabilities,
            last_accepted_query=scheduler['queries'][-1], next_public_arrival_ns=next_arrival,
            until_next_arrival_ns=next_arrival-now, original_transitions=scheduler['transitions'],
            registered_public_requests=forecast['queue_diagnostic_sidecar']['diagnostics']['registered_request_count'],
            mechanism='The settled simulator owns no handler or active activity. Its physical SAVE-target hazard reaches budget exactly on time, but the active-reference bridge cannot condition a physical mark on an empty work domain.',
            lifecycle_mapping_gap_indicated=False,
            native_idle_or_physical_SAVE_directly_observed=False),
        training_all_idle=training, training_all_idle_totals=dict(combined),
        training_all_idle_by_holder={owner:dict(by_holder[owner]) for owner in OWNERS},
        training_definition='All four projected worker-handler booleans at each risk left endpoint are False; sum end-start and terminal physical_SAVE_END_observed. Mark selection uses the same all-False mask.',
        scientific_limitations=[
            'Forecast SAVE is a statistical simulator event, not an independently measured physical SAVE.',
            'Training all-idle means projected handler mask, not observed 12-field engine idle or certified CUDA-empty state.',
            '91 all-idle risk intervals contain zero physical SAVE and zero idle mark labels; v51 categorical idle transitions are explicit extrapolation.',
            'Zero observed events does not establish a zero hazard. Conditional hazard and mark support are separate.',
            'Idle handoff SERVICE_READY remains a reference hypothesis, not evidence of actual GPU service.'],
        new_contract_review=dict(v51_v52_CPU_contract_checks_passed=result.wasSuccessful(),
            tests_run=result.testsRun, test_log=receipt(out/'tests.txt'),
            all_12_fields_false_and_frontier_empty_required=True,
            unknown_fields_rejected_by_original_validator_or_UNKNOWN_guard=True,
            raw_all_class_probabilities_and_same_holder_guard_preserved=True,
            idle_reference_target_and_distribution_are_None=True,
            idle_reference_progress_is_zero_without_current_nodes=True,
            pending_arrival_cannot_reselect_or_extend_transition=True,
            old_budget_integrated_without_redraw_and_new_budget_only_after_new_restore=True,
            no_rewind_guard_preserved=True,
            future_public_workload_does_not_enter_mark=True,
            recommended_status='CPU-reviewed explicit extrapolation hypothesis; no physical accuracy certification or production seal'),
        proposed_counting_hazard_contract=dict(training_only=True,
            sufficient_statistics='N=terminal physical SAVE count, T=positive all-idle exposure seconds; preserve holder/capture support.',
            predeclared_model='Homogeneous Poisson counting-process hazard with Jeffreys posterior Gamma(shape=N+0.5, rate=T_seconds); posterior mean or median fixed before forecast.',
            no_exposure='UNKNOWN unless an explicit predeclared pooling rule supplies supported exposure.',
            interval_semantics='Bayesian credible interval lower bound is positive. Separate frequentist zero-event one-sided bound may use -log(alpha)/T.',
            transfer='Projected handler-idle to complete engine-idle remains explicit; constant hazard, independent increments and source clock projection are assumptions.',
            budget_semantics='Integrate old state before changes; change rate only, never reset surviving restore age or redraw its hazard budget.',
            mark_identification='A lower idle hazard does not identify the physical idle next-owner mark; keep explicit extrapolation/UNKNOWN and same-holder/no-rewind guards.'),
        model_fitted=False, full_forecast_executed=False, GPU_started=False,
        validation_labels_read=False, frozen_sources_modified=False, source_bytes_unchanged=True)
    save_synced(out/'report.json',report)
    if not result.wasSuccessful():
        raise RuntimeError('CPU fixture failed')
    return receipt(out/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(run(parser.parse_args().output)))
