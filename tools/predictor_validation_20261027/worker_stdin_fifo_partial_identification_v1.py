"""Paired idle-donor partial identification; no learned busy transport law."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import worker_stdin_fifo_contract_v1 as fifo
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import receipt


def paired_idle_components(donor_submit_to_START_ns, donor_submit_to_write_record_ns,
                           line_extra_ns):
    """One joint draw: x and pre=d-x must never be sampled independently."""
    d,w,x=map(fifo.ns,(donor_submit_to_START_ns,donor_submit_to_write_record_ns,line_extra_ns))
    fifo.require(x<=min(w,d),'Latent line extra must belong to this donor pair')
    return dict(line_extra_ns=x,worker_pre_START_ns=d-x,
                donor_submit_to_START_ns=d,donor_submit_to_write_record_ns=w)


def transferred_START_bounds(donor_submit_to_START_ns,donor_submit_to_write_record_ns,
                             submit_ns,previous_READY_occurrence_ns,*,
                             assume_idle_pair_transfers):
    """Conditional sharp interval at a supplied submit and actual READY.

    This transfers one idle pair to a hypothetical busy worker only under an
    explicit assumption. It does not identify sender delay or pipe backpressure
    under a new schedule, and cannot generate a sender submit time.
    """
    fifo.require(assume_idle_pair_transfers is True,
                 'UNKNOWN_BUSY_TRANSFER: explicit idle-pair transfer assumption required')
    d,w,s,r=map(fifo.ns,(donor_submit_to_START_ns,donor_submit_to_write_record_ns,
                        submit_ns,previous_READY_occurrence_ns))
    u=min(w,d)
    return dict(START_ns_bounds=[max(r,s+u)+d-u,max(r,s)+d],
        joint_components=dict(line_extra_ns_bounds=[0,u],
                              worker_pre_START_ns_equals='donor_d_minus_same_draw_line_extra'),
        conditional_on_explicit_idle_pair_transfer_assumption=True,
        submit_time_supplied_not_predicted=True,
        busy_sender_pipe_backpressure_identified=False,
        unique_transport_decomposition_identified=False)


def counterexamples():
    requests=[dict(request_id='r'+str(i),owner='owner',task='text',pool_id='input',arrival_ns=i)
              for i in range(2)]
    keys=[('owner','text','input',i) for i in range(2)]
    handlers={key:dict(START_to_RESULT_mark_ns=8,START_to_READY_occurrence_ns=10,
        RESULT_mark_to_reader_ns=10,RESULT_reader_to_processing_ns=100,
        READY_occurrence_to_reader_ns=10,READY_reader_to_processing_ns=100) for key in keys}
    def world(first_write):
        draws={key:fifo.TransportDraw(0,0,first_write if i==0 else 1,0,1)
               for i,key in enumerate(keys)}
        return fifo.compose_explicit_world(requests,draws,handlers,
            initial_worker_read_release_ns={'owner':0},initial_sender_available_ns={'owner':0})
    positive=world(1);blocked=world(20)
    # Two equally feasible 2 ms idle decompositions, reused coherently across
    # queued requests. Fixed 10 ms handlers isolate transport uncertainty.
    scenarios=[]
    for x in (0,2_000_000):
        component=paired_idle_components(2_000_000,2_000_000,x)
        r=10_000_000;starts=[]
        for _ in range(2):
            start=max(r,x)+component['worker_pre_START_ns'];starts.append(start)
            r=start+10_000_000
        scenarios.append(dict(joint_component_draw=component,
                              supplied_submits_ns=[0,0],START_ns=starts))
    interval=transferred_START_bounds(2_000_000,2_000_000,0,10_000_000,
                                     assume_idle_pair_transfers=True)
    return fifo.encode(dict(
        all_examples_are_synthetic_CPU_worlds_not_observed_busy_donors=True,
        queued_START_before_parent_notices=positive,
        sender_blocking_separate_from_worker_FIFO=blocked,
        START_before_parent_write_record=dict(submit_ns=0,line_available_ns=2,
            worker_pre_START_ns=1,START_ns=3,parent_write_record_ns=5),
        same_idle_observation_different_busy_START=dict(
            idle_submit_ns=0,idle_write_record_ns=5,idle_START_ns=5,
            new_previous_READY_ns=10,new_submit_ns=0,
            worlds=[dict(line_extra_ns=5,worker_pre_START_ns=0,busy_START_ns=10),
                    dict(line_extra_ns=0,worker_pre_START_ns=5,busy_START_ns=15)]),
        two_millisecond_uncertainty_propagation=dict(
            conditional_on_idle_pair_transfer=True,first_START_interval=interval,
            coherent_worlds=scenarios,first_START_width_ns=2_000_000,
            second_START_width_ns=4_000_000,
            second_width_is_not_an_independent_marginal_draw=True),
        unqualified='Busy sender backpressure and changed-schedule transport remain UNKNOWN.'))


def build_report(audit_path):
    audit_path=Path(audit_path);audit_ref=receipt(audit_path)
    audit=json.loads(audit_path.read_text())
    fifo.require(audit['schema']==fifo.SCHEMA and audit['measured_donors']==150,
                 'Exact three-capture FIFO audit required')
    fifo.require(all(receipt(r['path'])==r for r in audit['sources']),
                 'Frozen FIFO audit closure changed')
    rows=[]
    for capture in audit['captures']:
        for row in capture['records']:
            fifo.require(not row['previous_handler_busy_at_submit'],
                         'Idle-pair bounds cannot be estimated from a busy donor')
            d=row['observed_submit_to_START_ns'];w=row['submit_to_write_record_ns'];u=min(w,d)
            rows.append(dict(capture_sha256=capture['capture_sha256'],
                request_id=row['request_id'],owner=row['owner'],
                pair=dict(d_ns=d,w_ns=w),line_extra_ns_bounds=[0,u],
                worker_pre_START_ns_bounds=[d-u,d],
                joint_constraint='worker_pre_START_ns = d_ns - same_draw_line_extra_ns',
                observed_busy_transfer_qualified=False))
    sources=strict_references([audit['sources'],[audit_ref],implementation_sources(__file__)])
    fifo.require(all(receipt(r['path'])==r for r in sources),'Source changed during partial-identification audit')
    return dict(schema='gsdeploy.worker_stdin_FIFO_partial_identification.v1',
        status='PAIRED_IDLE_COMPONENT_BOUNDS_ONLY_BUSY_TRANSFER_REQUIRES_EXPLICIT_ASSUMPTION',
        sources=sources,donor_pairs=rows,counterexamples=counterexamples(),
        learned_busy_sender_law=False,unknown_components_imputed=False,
        main_queue_provider_eligible=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        validation_labels_read=False,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fifo-audit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=build_report(args.fifo_audit)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],donor_pairs=len(result['donor_pairs']))))
