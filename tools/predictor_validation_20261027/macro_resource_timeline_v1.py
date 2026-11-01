"""Global resource-state slices from an explicit macro reference world.

This aligns one board clock, including idle and drain intervals. It supplies no
power/memory estimates and transfers no old execution-domain resource model.
"""
from __future__ import annotations

from fractions import Fraction

from .macro_response_queue_reference_v1 import SCHEMA, encode


def clock(value):
    if (type(value) is not dict or set(value)!={'numerator','denominator'}
            or any(type(v) is not int for v in value.values()) or value['denominator']<=0):
        raise ValueError('Exact rational reference clock required')
    result=Fraction(value['numerator'],value['denominator'])
    if result<0:raise ValueError('Nonnegative reference clock required')
    return result


def timeline(reference, public_requests, *, resident_owners):
    if reference.get('schema')!=SCHEMA:
        raise ValueError('Explicit macro reference world required')
    if (type(resident_owners) is not tuple or not resident_owners
            or len(set(resident_owners))!=len(resident_owners)
            or any(type(owner) is not str or not owner for owner in resident_owners)):
        raise ValueError('Explicit complete resident owner roster required')
    public={}
    for row in public_requests:
        if (type(row) is not dict or set(row)!={'request_id','owner','task','pool_id','arrival_ns'}
                or row['request_id'] in public or row['owner'] not in resident_owners):
            raise ValueError('Distinct public input and resident route identities required')
        public[row['request_id']]=row
    expected_ids=set(reference['RESULT_receipts'])|set(reference['unfinished_request_ids'])
    if len(public)!=reference['offered_requests'] or set(public)!=expected_ids:
        raise ValueError('Complete offered cohort required')
    start,end=map(clock,reference['original_window']);cutoff=clock(reference['cutoff_ns'])
    cuts={start,cutoff}
    if start<end<cutoff:cuts.add(end)
    pieces=[]
    for row in reference['response_exposure']:
        left,right=clock(row['left_ns']),clock(row['right_ns'])
        if not start<=left<right<=cutoff or row['request_id'] not in public:
            raise ValueError('Complete bounded macro interval identity required')
        request=public[row['request_id']]
        if request['owner']!=row['owner']:
            raise ValueError('Macro owner differs from public route')
        pieces.append((left,right,row));cuts.update((left,right))
    slices=[];ordered=sorted(cuts)
    for left,right in zip(ordered,ordered[1:]):
        active=[];seen=set()
        for first,last,row in pieces:
            if not first<=left<last:continue
            if row['owner'] in seen:
                raise ValueError('Overlapping actual macro intervals for one owner')
            seen.add(row['owner']);request=public[row['request_id']]
            active.append(dict(owner=row['owner'],request_id=row['request_id'],task=request['task'],
                pool_id=request['pool_id'],macro_id=row['macro_id'],
                response_phase=row['evidence_key'][0],actual_SM_work_known=False))
        slices.append(dict(left_ns=left,right_ns=right,duration_ns=right-left,
            measurement_overlap_ns=max(Fraction(0),min(end,right)-max(start,left)),
            resident_owners=list(resident_owners),actual_START_to_READY_handlers=sorted(active,key=lambda r:r['owner']),
            inactive_resident_memory_is_zero=False,power_watts=None,owned_memory_mib=None))
    return encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',
        status='GLOBAL_CLOCK_RESOURCE_INPUTS_NO_FITTED_RESOURCE_PROVIDER',
        slices=slices,total_wall_ns=cutoff-start,
        total_measurement_overlap_ns=max(Fraction(0),min(end,cutoff)-start),
        original_window=reference['original_window'],original_drain_end_ns=reference['original_drain_end_ns'],
        resource_clock='one_global_board_clock_not_sum_of_owner_exposures',
        memory_scope='startup_to_original_drain_owned_sampled_peak_requires_separate_startup_prefix',
        energy_scope='original_measurement_window_only_plus_startup_amortization_once_in_existing_scorer',
        old_resource_execution_identity_transferred=False,
        peak_memory_mib=None,board_energy_joules=None,startup_energy_joules=None,
        resource_prediction_qualified=False,main_queue_provider_eligible=False))
