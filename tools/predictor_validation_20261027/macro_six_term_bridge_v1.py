"""Explicit CPU-world adapter to the unchanged six-item forecast arithmetic.

No quality or resource values are inferred here. Rational simulator timestamps
are rounded upward by less than one ns, declared as a numerical approximation.
This never authenticates native timestamps, stability, physical accuracy or gain.
"""
from __future__ import annotations

import math

from .flow_validation import _score
from .macro_resource_timeline_v1 import clock

RESOURCE_FIELDS={'original_window_ns','original_drain_end_ns','resident_owners',
    'startup_energy_joules','startup_through_phase_start_owned_peak_mib',
    'phase_start_through_drain_owned_peak_mib','measurement_board_energy_joules',
    'measurement_energy_includes_drain','memory_safety_factor_already_applied',
    'startup_energy_already_amortized','source_evidence'}


def whole_score(reference, public_requests, scoring_plan, quality, resources=None):
    start,end=map(clock,reference['original_window']);drain=clock(reference['original_drain_end_ns'])
    if start.denominator!=1 or end.denominator!=1 or drain.denominator!=1:
        raise ValueError('Original public scoring boundaries must be integer ns')
    if clock(reference['cutoff_ns'])!=drain:
        raise ValueError('Whole-horizon scoring requires the original fixed drain cutoff')
    plan=dict(scoring_plan)
    if (plan.get('startup_amortization_seconds')!=3600
            or plan.get('weights')!={k:1/6 for k in 'QTLPME'}
            or round(plan['window_seconds']*1e9)!=end-start):
        raise ValueError('Original equal six weights, 3600s startup amortization and window required')
    residents=set(plan['aliases']);cohort={}
    for row in public_requests:
        if (type(row) is not dict or set(row)!={'request_id','owner','task','pool_id','arrival_ns'}
                or type(row['arrival_ns']) is not int or row['request_id'] in cohort
                or not start<=row['arrival_ns']<end or row['owner'] not in residents):
            raise ValueError('Complete original public cohort and arrivals required')
        cohort[row['request_id']]=dict(request_id=row['request_id'],alias=row['owner'],
            task=row['task'],pool_id=row['pool_id'],arrival_ns=row['arrival_ns']-int(start))
    expected=set(reference['RESULT_receipts'])|set(reference['unfinished_request_ids'])
    if len(cohort)!=reference['offered_requests'] or set(cohort)!=expected:
        raise ValueError('Scoring must retain every offered request including unfinished')
    finishes={};rounding=[]
    for request_id,stamp in reference['RESULT_receipts'].items():
        at=clock(stamp)
        if not start+cohort[request_id]['arrival_ns']<=at<=drain:
            raise ValueError('RESULT reader receipt outside its original cohort/drain')
        numeric=math.ceil(at)-int(start)
        finishes[request_id]=dict(cohort[request_id],finish_ns=numeric)
        rounding.append(float(math.ceil(at)-at))
    future_peak=energy=startup=historical_peak=None
    if resources is not None:
        if type(resources) is not dict or set(resources)!=RESOURCE_FIELDS or not resources['source_evidence']:
            raise ValueError('Source-bound complete resource scope declaration required')
        if (resources['original_window_ns']!=[int(start),int(end)]
                or resources['original_drain_end_ns']!=int(drain)
                or set(resources['resident_owners'])!=residents
                or resources['measurement_energy_includes_drain'] is not False
                or resources['memory_safety_factor_already_applied'] is not False
                or resources['startup_energy_already_amortized'] is not False):
            raise ValueError('Resource domain/window/residents or single-application scope mismatch')
        for field in ('startup_energy_joules','startup_through_phase_start_owned_peak_mib',
                'phase_start_through_drain_owned_peak_mib','measurement_board_energy_joules'):
            value=resources[field]
            if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
                raise ValueError('Nonnegative raw resource estimates or explicit UNKNOWN required')
        future_peak=resources['phase_start_through_drain_owned_peak_mib']
        historical_peak=resources['startup_through_phase_start_owned_peak_mib']
        energy=resources['measurement_board_energy_joules'];startup=resources['startup_energy_joules']
    # The original scorer also supports comparisons that omit a common startup
    # term. This absolute whole-world adapter has no such omission contract:
    # an unobserved startup contribution cannot silently become zero.
    missing_resources=[]
    if resources is not None:
        missing_resources=[field for field in (
            'startup_energy_joules','measurement_board_energy_joules',
            'startup_through_phase_start_owned_peak_mib',
            'phase_start_through_drain_owned_peak_mib') if resources[field] is None]
    else:
        missing_resources=['startup_energy_joules','measurement_board_energy_joules',
            'startup_through_phase_start_owned_peak_mib',
            'phase_start_through_drain_owned_peak_mib']
    forecast=dict(cohort=cohort,finishes=finishes,peak_memory_mib=future_peak,
        board_energy_joules=energy if startup is not None else None)
    score=_score(plan,forecast,quality,startup_joules=startup,historical_peak=historical_peak)
    return dict(schema='gsdeploy.macro_world_original_six_term_bridge.v1',
        forecast=forecast,score=score,original_scorer_reused=True,weights_changed=False,
        numeric_timestamp_conversion='ceil_exact_rational_ns_for_original_integer_scorer',
        maximum_timestamp_roundup_ns=max(rounding,default=0),
        E2E_endpoint='scheduled_arrival_to_RESULT_reader_receipt',
        resources_were_inferred=False,unknown_resource_fields=missing_resources,
        original_stability_evaluated=False,
        original_dispatch_qualified=(reference.get('dispatch_contract_matches_original_sender') is True
            and reference.get('busy_transport_transfer_qualified') is True
            and reference.get('internal_work_template_qualified') is True),
        main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False)
