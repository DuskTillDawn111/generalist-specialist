"""Exact conditional transport identification without choosing a latent line.

These are feasible-set calculations, not a fitted latent probability law. Any
transfer of paired line/continuation components to a changed schedule is an
explicit additional assumption. v2 raw statistics are contamination-HOLD forensic
evidence only, never donors for a qualified production training law.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path

from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .worker_stdin_fifo_contract_v1 import encode

SCHEMA='gsdeploy.twoFIFO_transport_partial_identification.v2'
HOLD='COLLECTOR_MEASUREMENT_CHECKPOINT_CONTAMINATION_HOLD'


def require(value,message):
    if not value:raise ValueError(message)


def ns(value):
    require(type(value) in (int,Fraction) and value>=0,'Exact nonnegative rational nanoseconds required')
    return Fraction(value)


def identify_source(submit_ns,write_record_ns,START_ns,previous_READY_ns):
    s,w,b,r=map(ns,(submit_ns,write_record_ns,START_ns,previous_READY_ns))
    require(s<=w and s<=b and r<=b,'Observed sender/START/READY chronology required')
    W,H,q=w-s,b-s,r-s;upper=min(W,H)
    xlow=H-max(q,upper);xhigh=H-max(q,Fraction(0))
    require(0<=xlow<=xhigh,'Empty joint source transport feasible set')
    if r>=w:regime='PREVIOUS_READY_DOMINATES_WRITE_RECORD'
    elif r<=s:regime='PREVIOUS_READY_PRECEDES_SUBMIT'
    else:regime='PREVIOUS_READY_INSIDE_SUBMIT_WRITE_INTERVAL'
    return dict(schema=SCHEMA,source_regime=regime,submit_to_write_record_ns=W,
        observed_submit_to_START_ns=H,previous_READY_minus_submit_ns=q,
        stdin_line_extra_ns_bounds=[Fraction(0),upper],
        worker_continuation_extra_ns_bounds=[xlow,xhigh],
        worker_continuation_extra_identified=xlow==xhigh,
        identified_worker_continuation_extra_ns=xlow if xlow==xhigh else None,
        joint_feasible_set='0<=u<=min(W,H); continuation=H-max(q_source,u)',
        line_continuation_independent=False,latent_probability_law_identified=False,
        source_clock_origin_used_for_lookup=False)


def checked_source(source):
    """Reconstruct the closed canonical feasible set using exact clocks only."""
    require(type(source) is dict and source.get('schema')==SCHEMA,'Exact source feasible set required')
    W=ns(source.get('submit_to_write_record_ns'));H=ns(source.get('observed_submit_to_START_ns'))
    q=source.get('previous_READY_minus_submit_ns')
    require(type(q) in (int,Fraction),'Exact signed source READY offset required')
    q=Fraction(q);anchor=max(Fraction(0),-q)
    canonical=identify_source(anchor,anchor+W,anchor+H,anchor+q)
    def exact(actual,wanted):
        if isinstance(wanted,Fraction):return type(actual) in (int,Fraction) and actual==wanted
        if type(actual) is not type(wanted):return False
        if type(wanted) is dict:return set(actual)==set(wanted) and all(exact(actual[k],v) for k,v in wanted.items())
        if type(wanted) is list:return len(actual)==len(wanted) and all(exact(a,b) for a,b in zip(actual,wanted))
        return actual==wanted
    require(exact(source,canonical),'Noncanonical/extra/missing source feasible-set fields')
    return canonical


def project_target(source,submit_ns,previous_READY_ns):
    """Exact image of every feasible paired source decomposition.

    u is the complete-line delay from source submit, paired with continuation
    H-max(q_source,u). The same vector carries source write-record duration W.
    Target START is s_target + max(q_target,u) + H-max(q_source,u).
    The envelope needs only endpoints and the two piecewise-linear corners.
    """
    source=checked_source(source)
    s,r=map(ns,(submit_ns,previous_READY_ns));qtarget=r-s
    W=source['submit_to_write_record_ns'];H=source['observed_submit_to_START_ns'];q=source['previous_READY_minus_submit_ns']
    lower,upper=source['stdin_line_extra_ns_bounds']
    require(lower==0 and upper==min(W,H) and H>=max(q,0),'Exact complete paired source constraints required')
    points={lower,upper}
    for corner in (q,qtarget):
        if lower<=corner<=upper:points.add(corner)
    values=[s+max(qtarget,u)+H-max(q,u) for u in points]
    lo,hi=min(values),max(values)
    if lo==hi:
        if source['source_regime']=='PREVIOUS_READY_DOMINATES_WRITE_RECORD' and qtarget>=W:
            reason='SOURCE_AND_TARGET_READY_DOMINATE_ALL_FEASIBLE_LINE_TIMES'
        elif source['source_regime']=='PREVIOUS_READY_PRECEDES_SUBMIT' and qtarget<=0:
            reason='SOURCE_AND_TARGET_READY_PRECEDE_ALL_FEASIBLE_LINE_TIMES'
        else:reason='ALL_JOINTLY_FEASIBLE_SOURCE_DECOMPOSITIONS_AGREE'
    else:reason='CHANGED_SCHEDULE_DEPENDS_ON_UNOBSERVED_PAIRED_LINE_CONTINUATION'
    return dict(status='CONDITIONALLY_POINT_IDENTIFIED_START' if lo==hi else 'PARTIALLY_IDENTIFIED_START_ENVELOPE',
        START_ns_bounds=[lo,hi],identified_START_ns=lo if lo==hi else None,
        target_submit_ns=s,target_previous_READY_ns=r,target_write_record_ns=s+W,
        reason=reason,transfer_assumption='paired_source_line_extra_continuation_extra_and_write_record_duration_transfer_unchanged',
        latent_point_selected=False,latent_probability_law_identified=False,
        changed_schedule_transport_transfer_qualified=False,main_queue_provider_eligible=False)


def observed_dominance(raw,plan):
    """Whole-attempt history, including priming; every offered row retained."""
    require(raw['execution_id']==plan['execution_id'],'Forensic raw/plan execution mismatch')
    measured={r['request_id']:r for r in raw['request_records']}
    offered=[a['request_id'] for c in plan['cases'] for a in c['arrivals']]
    require(set(offered)==set(measured) and len(offered)==len(measured),'Exact complete offered identity set required')
    public_order=raw['priming_request_records']+[measured[a['request_id']] for c in plan['cases']
        for a in sorted(c['arrivals'],key=lambda a:(a['arrival_ns'],a['arrival_index']))]
    previous={};rows=[];counts=defaultdict(Counter)
    priming_ids={r['request_id'] for r in raw['priming_request_records']}
    for row in public_order:
        owner=row['model'];prior=previous.get(owner);role='priming' if row['request_id'] in priming_ids else 'measured'
        record=dict(request_id=row['request_id'],owner=owner,role=role,case_id=row.get('case_id'))
        values=[row.get(k) for k in ('actual_submit_time_ns','submit_write_completed_time_ns','observed_worker_start_time_ns')]
        ready=prior.get('previous_handler_next_request_ready',{}).get('perf_ns') if prior else None
        if any(type(v) is not int for v in values+[ready]):
            record.update(status='UNKNOWN_MISSING_ACTUAL_PREVIOUS_READY_OR_SEND_START_CLOCKS',request_status=row.get('status'))
            counts[role]['unknown']+=1
        else:
            source=identify_source(*values,ready)
            # Reconstructing the same source schedule must point-identify START
            # for every feasible source line, even when its split is unknown.
            same=project_target(source,values[0],ready)
            require(same['identified_START_ns']==values[2],'Same-schedule source START failed exact feasible-set reconstruction')
            record.update(status='OBSERVED_SOURCE_CONSTRAINTS',constraints=source)
            counts[role][source['source_regime']]+=1
            counts[role]['continuation_identified' if source['worker_continuation_extra_identified'] else 'continuation_not_identified']+=1
        rows.append(record)
        # Whole public owner history is retained even for incomplete requests.
        # Never jump past an unresolved offered predecessor to an older READY.
        previous[owner]=row
    require(sum(r['role']=='measured' for r in rows)==len(offered),'Offered denominator lost')
    return dict(rows=rows,counts={k:dict(v) for k,v in counts.items()},
        measured_offered=len(offered),priming_observed=len(raw['priming_request_records']),
        priming_planned=len(plan['priming']),whole_attempt_previous_READY_history=True,
        missing_predecessor_substituted_with_zero=False,source_split_probability_fitted=False)


def run(raw_path,plan_path,output):
    raw_ref,plan_ref=receipt(raw_path),receipt(plan_path);plan=checked(plan_ref);raw=checked(raw_ref)
    require(raw['plan']==plan_ref,'Exact raw-bound plan receipt required')
    require(raw.get('split')==plan.get('split')=='training','Training-only forensic role required before label inspection')
    require(raw.get('schema') in ('gsdeploy.original_twoFIFO_owned_training_capture.v2','gsdeploy.original_twoFIFO_owned_training_capture.v3'),
        'Explicit v2/v3 forensic capture schema required')
    require(raw.get('schema').rsplit('.',1)[-1]==plan['schema'].split('.')[-2],'Matching raw/plan collector version required')
    summary=observed_dominance(raw,plan)
    v2=raw['schema'].endswith('.v2')
    result=dict(schema=SCHEMA+'.observed_dominance',status=HOLD if v2 else 'FORENSIC_OBSERVED_FEASIBLE_SET_ONLY',
        raw=raw_ref,plan=plan_ref,summary=summary,source_training_domain_qualified=False,
        counts_are_production_training_support=False,fitted_transport_probability_law=False,
        transfer_to_changed_schedule_qualified=False,main_queue_provider_eligible=False,
        validation_labels_read=False,GPU_started=False,
        sources=strict_references([raw_ref,plan_ref,implementation_sources(__file__)]))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'report.json').open('x') as stream:json.dump(encode(result),stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.raw,args.plan,args.output)))
