"""Measured six-term training labels, kept out of forecasting inputs.

The caller must authenticate the native source and persist predictions first.
This pure label adapter does not qualify quality, instantaneous memory peaks,
sampling assumptions, prediction accuracy, stability or routing benefit.
"""
from __future__ import annotations

from fractions import Fraction
import copy
import math

from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_observed_resources_v2 as observed
from . import twoFIFO_source_rebound_execute_v1 as executor
from .macro_six_term_bridge_v1 import whole_score
from .macro_response_queue_reference_v1 import encode

SCHEMA = 'gsdeploy.twoFIFO_source_rebound_observed_six_score.v1'
require = generated.require


def score_case(raw, plan, case, epoch, scoring_plan, quality, *, source_evidence):
    """Measure fixed-window E and startup-through-fixed-drain sampled M.

    These labels may use the heldout future. They must never be passed to the
    folded provider, native seed, donor law or candidate-selection procedure.
    """
    require(raw.get('schema') == executor.RAW_SCHEMA and raw.get('backend') == 'native'
            and raw.get('status') == 'completed', 'Complete rebound native raw required')
    require(all(raw.get(k) == plan[k] for k in ('capture_id', 'attempt_id', 'execution_id')),
            'Exact raw/plan identity required')
    require(type(source_evidence) is dict and source_evidence, 'Explicit native label provenance required')
    planned = [c for c in plan['cases'] if c['case_id'] == case['case_id']]
    require(len(planned) == 1 and planned[0] == case, 'Exact public case required')
    native_case = [c for c in raw['cases'] if c['case_id'] == case['case_id']]
    require(len(native_case) == 1 and native_case[0]['observation_start_ns'] == epoch,
            'Original native case epoch required')
    measured = observed.observed(raw)
    rows = [r for r in measured['cases'] if r['case_id'] == case['case_id']]
    require(len(rows) == 1, 'Exact measured case resource row required')
    row = rows[0]
    start, end, drain = epoch, epoch + generated.MEASUREMENT_NS, epoch + generated.MEASUREMENT_NS + generated.DRAIN_NS
    require([row[k] for k in ('measurement_start_ns', 'measurement_end_ns', 'fixed_drain_deadline_ns')]
            == [start, end, drain], 'Original fixed measurement and drain boundaries required')
    require(set(scoring_plan['aliases']) == set(plan['resident_owners']), 'All resident owners must be scored')
    startup = measured['startup']['start_ns']
    before = [s['owned_used_mib'] for s in raw['board_samples'] if startup <= s['time_ns'] <= start]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in before),
            'Finite nonnegative sampled owned memory required')
    # Existing measured adapter uses all inactive residents and excludes worker
    # close/shutdown. Keep prior history separate for the original scorer.
    resources = dict(original_window_ns=[start, end], original_drain_end_ns=drain,
        resident_owners=list(plan['resident_owners']),
        startup_energy_joules=measured['startup']['energy_joules'],
        startup_through_phase_start_owned_peak_mib=max(before) if before else None,
        phase_start_through_drain_owned_peak_mib=row['local_measurement_through_drain_owned_sampled_peak_mib'],
        measurement_board_energy_joules=row['measurement_energy_joules'],
        measurement_energy_includes_drain=False, memory_safety_factor_already_applied=False,
        startup_energy_already_amortized=False, source_evidence=copy.deepcopy(source_evidence))
    labels = generated.native_labels(raw, case, epoch, scoring_plan, quality)
    requests = generated.public_case(case, epoch)
    arrivals = {r['request_id']: r['arrival_ns'] for r in requests}
    receipts = {r['request_id']: Fraction(arrivals[r['request_id']] + r['reader_E2E_ns'])
                for r in labels['requests'] if r['completed_within_fixed_drain']}
    frame = dict(original_window=encode([Fraction(start), Fraction(end)]),
        original_drain_end_ns=encode(Fraction(drain)), cutoff_ns=encode(Fraction(drain)),
        RESULT_receipts=encode(receipts), offered_requests=len(requests),
        unfinished_request_ids=[r['request_id'] for r in requests if r['request_id'] not in receipts])
    scored = whole_score(frame, requests, scoring_plan, quality, resources)
    require(all(scored['score']['terms'][k] == labels['original_score']['terms'][k] for k in 'QTLP'),
            'Observed resources must not change original Q/T/L/P')
    return dict(schema=SCHEMA, status='MEASURED_TRAINING_SCORE_LABEL_NOT_PREDICTION',
        case_id=case['case_id'], attempt_id=plan['attempt_id'], block_id=plan['capture_id'],
        reader_labels=labels, measured_resources=resources, measured_original_score=scored['score'],
        original_score_bridge=scored, observed_case_resources=row, offered_requests=len(requests),
        complete_offered_cohort_preserved=True, heldout_future_used_for_evaluation_only=True,
        quality_from_frozen_training_expectations_not_native_outcome=True,
        actual_native_quality_validated=False, sampled_memory_is_continuous_capacity_bound=False,
        original_stability_evaluated=False, predictor_accuracy_qualified=False,
        native_evidence_authenticated_by_this_pure_helper=False,
        final_gain_eligible=False, validation_labels_read=False, GPU_started=False)
