"""Joint HELD / IN_TRANSITION / unresolved initial scheduler-state prior.

Future source RESTOREs are producer training targets only. Prospective predict
and realization use frozen joint scenarios; they never query future events.
An observed scheduler gap does not establish physical nonexecution.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict,Counter
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import time

from gsdeploy.forecast_errors import UnknownEvidence
from .context_envelope_diagnostic_v8 import context_intervals
from .context_initial_prior_v14 import (CORE_SETUP_FIELDS,OBSERVER_SETUP_FIELDS,SETUP_FIELDS,
    frontier_key,scenario_from_observation,public_setup)
from .context_initial_transition_v13 import next_kernel_bound
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .profiling import digest

SCHEMA='gsdeploy.joint_initial_held_or_transition_prior.v15'
KINDS=('HELD_CONTEXT','IN_TRANSITION','UNKNOWN')
POLICY=dict(original_capture_and_job_weights_preserved=True,original_rows_preserved=True,
    source_gap_is_explicit_joint_state=True,unknown_mass_not_renormalized=True,
    conditioning='First pending frontier and public setup only; same exact/coarse/setup tiers as frozen v14',
    next_restore_and_service_events_are_training_targets_only=True,
    scheduler_gap_is_physical_nonexecution=False)


def _unknown(original,reason):
    return dict(copy.deepcopy(original),state_kind='UNKNOWN',state_identified=False,
                state_evidence=dict(reason=reason))


def annotate_gap(original,observation,*,intervals,issues,all_contexts_mapped,kernel_starts):
    """Partial identification in the recorded scheduler namespace only."""
    clock=observation['clock_ns'];starts=[r[0] for r in intervals]
    at=bisect_right(starts,clock)
    if not all_contexts_mapped:return _unknown(original,'UNMAPPED_RECORDED_CONTEXTS')
    if any(right[0]<left[1] for left,right in zip(intervals,intervals[1:])):
        return _unknown(original,'OVERLAPPING_RECORDED_ENVELOPES')
    if any(a<=clock<b for a,b,owner in intervals):return _unknown(original,'QUERY_NOT_IN_OBSERVED_GAP')
    if at==0 or at==len(intervals):return _unknown(original,'MISSING_TWO_SIDED_ENVELOPE_WITNESS')
    pa,pb,previous=intervals[at-1];na,nb,target=intervals[at]
    if not pb<=clock<na:return _unknown(original,'INVALID_SOURCE_GAP_ORDER')
    if any(pa<=r['timestamp']<=nb for r in issues):return _unknown(original,'UNRESOLVED_EVENT_ACROSS_GAP_WITNESSES')
    bounds=next_kernel_bound(kernel_starts.get(target,[]),na,nb)
    upper=bounds['upper_ns']
    return dict(copy.deepcopy(original),state_kind='IN_TRANSITION',state_identified=True,
        previous_held_owner=previous,elapsed_gap_ns=clock-pb,next_owner=target,
        residual_restore_gap_ns=na-clock,complete_restore_gap_ns=na-pb,
        next_restore_age_ns=0,
        service_ready_delay_bounds_ns=dict(lower_ns=na-clock,upper_ns=None if upper is None else upper-clock),
        service_ready_delay_after_restore_bounds_ns=dict(lower_ns=0,upper_ns=None if upper is None else upper-na),
        service_ready_point_identified=False,readiness_status='INTERVAL_CENSORED_AFTER_RESTORE' if upper is not None else 'NO_SERVICE_UPPER_WITNESS_AFTER_RESTORE',
        holder_identified=False,held_owner=None,context_age_ns=None,
        state_evidence=dict(query_clock_ns=clock,previous_restore_ns=pa,previous_SAVE_ns=pb,
            next_RESTORE_ns=na,next_SAVE_ns=nb,previous_held_owner=previous,next_owner=target,
            service_upper_witness_ns=upper,all_recorded_contexts_mapped=True,
            unpaired_events_between_witnesses=False,recorded_envelope_gap=True,
            physical_holder_or_service_identified=False))


def train(original_prior_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    original_ref=receipt(original_prior_path);original=checked(original_ref)
    from .context_initial_prior_v14 import InitialContextPrior
    InitialContextPrior(original)
    source_ref=original['source_supervision'];source=checked(source_ref)
    # Full producer replay, not just one selected unknown row.
    from .context_initial_transition_v13 import extract
    regenerated=extract([a['report'] for a in source['training_audits']])
    if regenerated!=source:raise ValueError('Complete original initial/transition supervision differs from source rederivation')
    del regenerated
    sources=[receipt(__file__),original_ref,source_ref,*source['sources']]
    caches={};evidence=[];scenarios=[]
    old_by_id={r['scenario_id']:r for r in original['scenarios']}
    for audit in source['training_audits']:
        report=checked(audit['report']);raw=checked(audit['raw']);context=checked(audit['context'])
        sql=next(r for r in report['sources'] if r['path'].endswith('.sqlite'))
        if receipt(sql['path'])!=sql:raise ValueError('Original context SQLite changed')
        c=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        events,brackets,issues=context_intervals(c);c.close()
        mapping={tuple(r['context']):r['alias'] for r in report['controlled_temporal_attribution']['creation_windows']}
        intervals=sorted((a,b,mapping[k]) for k,pieces in brackets.items() if k in mapping for a,b in pieces)
        observed=checked(report['kernel_ready_observations'])['rows'];records={r['request_id']:r for r in raw['request_records']}
        kernels=defaultdict(list)
        for row in observed:kernels[records[row['request_id']]['model']].append(row['observed_start_ns'])
        caches[audit['raw']['sha256']]=dict(intervals=intervals,issues=issues,
            all_contexts_mapped=set(events)==set(mapping),kernel_starts={k:sorted(v) for k,v in kernels.items()},
            public_setup=public_setup(context))
    for observation in source['initial_observations']:
        data=caches[observation['capture']]
        reconstructed=scenario_from_observation(observation,data['public_setup'])
        old=old_by_id.get(reconstructed['scenario_id'])
        if old!=reconstructed:raise ValueError('Original initial scenario or joint metadata changed')
        if old['holder_identified']:
            annotated=dict(copy.deepcopy(old),state_kind='HELD_CONTEXT',state_identified=True,
                state_evidence=dict(original_source_observation_status=observation['status']))
        elif observation['status']=='UNKNOWN_INITIAL_CONTEXT_GAP':
            annotated=annotate_gap(old,observation,**{k:data[k] for k in ('intervals','issues','all_contexts_mapped','kernel_starts')})
        else:annotated=_unknown(old,'ORIGINAL_UNPAIRED_OR_UNRESOLVED_INITIAL_STATE')
        scenarios.append(annotated)
    # Preserve original order too: fixed scenario quantiles must not be changed
    # by sorting the newly identified gap state ahead of existing observations.
    by_id={r['scenario_id']:r for r in scenarios};scenarios=[by_id[r['scenario_id']] for r in original['scenarios']]
    if len(scenarios)!=len(original['scenarios']) or len(by_id)!=len(scenarios):
        raise ValueError('Every original joint initial row must appear exactly once')
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Initial prior producer source changed')
    model=dict(schema=SCHEMA,training_only=True,policy=POLICY,scenarios=scenarios,
        original_prior=original_ref,source_supervision=source_ref,implementation=receipt(__file__),
        source_receipts=sources,original_source_rederivation_complete=True,
        original_row_count=len(original['scenarios']),state_counts=dict(Counter(r['state_kind'] for r in scenarios)),
        prospective_queries_read_future_events=False,physical_nonexecution_identified=False,
        independent_accuracy_validated=False,GPU_started=False)
    model['model_sha256']=digest(model);save(out/'model.json',model)
    report=dict(status='SOURCE_DERIVED_JOINT_INITIAL_TRANSITION_PRIOR_FITTED',model=receipt(out/'model.json'),
        sources=sources,state_counts=model['state_counts'],all_original_rows_preserved=True,
        original_scenario_order_preserved=True,original_capture_weights_preserved=True,
        full_original_source_rederivation=True,future_RESTOREs_are_training_labels_only=True,
        unknown_rows_dropped=False,missing_readiness_imputed=False,prospective_future_event_lookup=False,
        main_v14_files_modified=False,GPU_started=False,elapsed_seconds=time.perf_counter()-began)
    save(out/'report.json',report);return report


class JointInitialContextPrior:
    def __init__(self,model,*,allow_cpu_fixture=False):
        if (model.get('schema')!=SCHEMA or model.get('training_only') is not True
                or model.get('policy')!=POLICY
                or model.get('model_sha256')!=digest({k:v for k,v in model.items() if k!='model_sha256'})):
            raise ValueError('Source-bound v15 joint initial scheduler-state prior required')
        if model.get('CPU_fixture'):
            if allow_cpu_fixture is not True:raise ValueError('Explicit CPU fixture opt-in required')
        else:
            # Source closure verification happens once. Prediction below has
            # no file I/O, event reader, query clock, or future-arrival input.
            for ref in model['source_receipts']:
                if receipt(ref['path'])!=ref:raise ValueError('Joint initial prior source changed')
        if any(r.get('state_kind') not in KINDS for r in model['scenarios']):
            raise ValueError('Explicit held/transition/unknown scenario required')
        self.rows=copy.deepcopy(model['scenarios']);self.model_sha256=model['model_sha256']

    def predict(self,*,first_pending_frontier,public_execution_setup,
                allow_unvalidated_frontier_pooling=False,allow_unvalidated_observer_transfer=False):
        if set(public_execution_setup)!=set(SETUP_FIELDS):raise ValueError('Only explicit public setup fields required')
        exact=frontier_key(first_pending_frontier);coarse=frontier_key(first_pending_frontier,coarse=True)
        candidates=[r for r in self.rows if all(r['public_setup'][k]==public_execution_setup[k] for k in CORE_SETUP_FIELDS)]
        if not candidates:raise UnknownEvidence('INITIAL_PRIOR_PUBLIC_EXECUTION_SETUP_UNSUPPORTED')
        transfer=any(any(r['public_setup'][k]!=public_execution_setup[k] for k in OBSERVER_SETUP_FIELDS) for r in candidates)
        if transfer and allow_unvalidated_observer_transfer is not True:
            raise UnknownEvidence('INITIAL_PRIOR_OBSERVER_TRANSFER_REQUIRES_EXPLICIT_HYPOTHESIS')
        if not transfer:candidates=[r for r in candidates if all(r['public_setup'][k]==public_execution_setup[k] for k in OBSERVER_SETUP_FIELDS)]
        matching=[r for r in candidates if frontier_key(r['first_pending_frontier'])==exact];tier='EXACT_FIRST_PENDING_FRONTIER'
        if not matching and allow_unvalidated_frontier_pooling is True:
            matching=[r for r in candidates if frontier_key(r['first_pending_frontier'],coarse=True)==coarse];tier='POOLED_INPUT_AND_PHASE_HYPOTHESIS'
            if not matching:matching=candidates;tier='POOLED_FRONTIER_HYPOTHESIS'
        if not matching:raise UnknownEvidence('INITIAL_PRIOR_FIRST_FRONTIER_UNSEEN_EXPLICIT_POOLING_HYPOTHESIS_REQUIRED')
        captures=defaultdict(list)
        for row in matching:captures[row['capture']].append(row)
        scenarios=[]
        for capture,rows in sorted(captures.items()):
            scenarios.extend(dict(copy.deepcopy(r),probability=1/(len(captures)*len(rows))) for r in rows)
        masses={kind:sum(r['probability'] for r in scenarios if r['state_kind']==kind) for kind in KINDS}
        return dict(status='UNCALIBRATED_JOINT_INITIAL_HELD_OR_TRANSITION_PRIOR',scenarios=scenarios,
            probability_sum=sum(r['probability'] for r in scenarios),state_probability_masses=masses,
            unknown_state_mass=masses['UNKNOWN'],unknown_holder_mass=masses['UNKNOWN'],
            in_transition_mass=masses['IN_TRANSITION'],holder_not_applicable_mass=masses['IN_TRANSITION'],
            unresolved_point_readiness_mass=sum(r['probability'] for r in scenarios if not r['service_ready_point_identified']),
            supporting_capture_ids=sorted(captures),conditioning_tier=tier,observer_transfer_hypothesis=transfer,
            model_sha256=self.model_sha256,missing_mass_renormalized_away=False,physical_nonexecution_claimed=False,
            prospective_future_event_lookup=False,query_clock_used_as_feature=False,independent_validation=False)


def realize_joint_initial_scenario(scenario,*,query_clock_ns,readiness_rule='bounds_only'):
    if type(query_clock_ns) is not int or query_clock_ns<0:raise ValueError('Nonnegative integer prospective query clock required')
    kind=scenario['state_kind']
    if kind=='UNKNOWN':raise UnknownEvidence('INITIAL_STATE_REMAINS_UNIDENTIFIED')
    if readiness_rule not in ('bounds_only','lower_bound_hypothesis','upper_bound_hypothesis','witnessed_point_only'):
        raise ValueError('Explicit readiness bounds or hypothesis required')
    bounds=copy.deepcopy(scenario['service_ready_delay_bounds_ns']);lower,upper=bounds['lower_ns'],bounds['upper_ns']
    if readiness_rule=='bounds_only':service=None
    elif readiness_rule=='lower_bound_hypothesis':service=query_clock_ns+lower
    elif readiness_rule=='upper_bound_hypothesis':
        if upper is None:raise UnknownEvidence('INITIAL_SERVICE_UPPER_WITNESS_UNAVAILABLE')
        service=query_clock_ns+upper
    else:
        if upper is None or lower!=upper:raise UnknownEvidence('INITIAL_SERVICE_POINT_UNIDENTIFIED')
        service=query_clock_ns+lower
    result=dict(state_kind=kind,source_scenario_id=scenario['scenario_id'],query_clock_ns=query_clock_ns,
        held_owner=scenario['held_owner'],context_age_ns=scenario['context_age_ns'],
        service_ready_delay_bounds_ns=bounds,service_ready_ns=service,readiness_rule=readiness_rule,
        numerical_readiness_is_hypothesis=service is not None and not scenario['service_ready_point_identified'],
        prospective_future_event_lookup=False,physical_nonexecution_claimed=False)
    if kind=='HELD_CONTEXT':result['restore_start_ns']=query_clock_ns-scenario['context_age_ns']
    else:result.update(previous_held_owner=scenario['previous_held_owner'],elapsed_gap_ns=scenario['elapsed_gap_ns'],
        previous_SAVE_ns=query_clock_ns-scenario['elapsed_gap_ns'],next_owner=scenario['next_owner'],
        residual_restore_gap_ns=scenario['residual_restore_gap_ns'],
        next_RESTORE_ns=query_clock_ns+scenario['residual_restore_gap_ns'],
        service_ready_delay_after_restore_bounds_ns=copy.deepcopy(scenario['service_ready_delay_after_restore_bounds_ns']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-prior',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=train(a.original_prior,a.output);print(json.dumps({k:r[k] for k in ('status','state_counts','elapsed_seconds')}))
