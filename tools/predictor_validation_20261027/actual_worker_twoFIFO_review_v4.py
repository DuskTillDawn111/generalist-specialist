"""Source-bound CPU counterexamples for the isolated actual-worker v4 engine."""
import argparse
import json
from pathlib import Path

from . import actual_worker_twoFIFO_reference_v4 as reference
from .macro_response_queue_reference_v1 import Mark,Macro,RequestPlan
from .worker_stdin_fifo_contract_v1 import TransportDraw
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt


def explicit_world(*,count=2,write=1,line=0,pre=1,work=10,
                   result_reader_slacks=(10,9),result_processing_slacks=(100,0),
                   ready_reader_slack=2,cutoff=150):
    requests=[dict(request_id='synthetic:'+str(i),owner='owner',task='text',pool_id='public-input',arrival_ns=i)
              for i in range(count)]
    plans=tuple(RequestPlan((Macro('macro:'+str(i),'supplied_work',work,
        (Mark('RESULT:'+str(i),work*8//10,0,'result','owner'),)),),0,'owner') for i in range(count))
    def slack(reader,processing=0):
        return dict(reader_slack_ns=reader,collector_dequeue_slack_ns=0,collector_processing_slack_ns=processing)
    kwargs=dict(requests=requests,plans={('owner','text','public-input'):plans},
        rates={('supplied_work',0,0):1},window_start_ns=0,window_end_ns=count,
        drain_end_ns=cutoff,protocol_slacks={plan:{'RESULT:'+str(i):slack(result_reader_slacks[i],result_processing_slacks[i]),
            'worker_READY':slack(ready_reader_slack)} for i,plan in enumerate(plans)},
        transport_draws={('owner','text','public-input',i):TransportDraw(0,0,write,line,pre) for i in range(count)},
        initial_sender_available_ns={'owner':0},initial_worker_read_release_ns={'owner':0},
        initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=()))
    return reference.simulate(**kwargs),requests


def build_report(v3_report,fifo_audit,partial_report):
    paths=[Path(path) for path in (v3_report,fifo_audit,partial_report)]
    docs=[json.loads(path.read_text()) for path in paths]
    closure=[]
    for path,doc in zip(paths,docs):
        closure.extend(doc['sources']);closure.append(receipt(path))
    if not all(receipt(row['path'])==row for row in closure):
        raise ValueError('Frozen training/FIFO closure changed')
    v3_source=Path(__file__).with_name('macro_protocol_timing_reference_v3.py')
    if receipt(v3_source) not in docs[0]['sources']:
        raise ValueError('Exact frozen v3 executor must be in training05 closure')
    positive,public=explicit_world()
    blocked,_=explicit_world(write=20)
    early_cutoff,_=explicit_world(cutoff=5)
    donors=[row for capture in docs[1]['captures'] for row in capture['records']]
    donor=max(donors,key=lambda row:min(row['observed_submit_to_START_ns'],row['submit_to_write_record_ns']))
    d=donor['observed_submit_to_START_ns'];w=donor['submit_to_write_record_ns'];u=min(d,w)
    work=max(10_000_000,3*w+d);worlds=[]
    for x in (0,u):
        world,_=explicit_world(count=3,write=w,line=x,pre=d-x,work=work,
            result_reader_slacks=(0,0,0),result_processing_slacks=(0,0,0),ready_reader_slack=0,
            cutoff=10*(work+d+w))
        worlds.append(dict(conditional_on_idle_pair_transfer=True,paired_draw=dict(d_ns=d,w_ns=w,x_ns=x,pre_ns=d-x),
                           world=world))
    sources=strict_references([closure,implementation_sources(__file__)])
    if not all(receipt(row['path'])==row for row in sources):
        raise ValueError('Source changed during v4 review')
    return dict(schema='gsdeploy.actual_worker_twoFIFO_CPU_review.v4',
        status='SOURCE_FIFO_ORDER_AND_EXPLICIT_WORLD_COUNTEREXAMPLES_ONLY',sources=sources,
        copied_frozen_v3_source=receipt(v3_source),
        positive_continuation_world=positive,sender_blocking_world=blocked,
        cutoff_world=early_cutoff,public_synthetic_requests=public,
        source_bound_idle_pair_endpoint_sensitivity=dict(
            donor_capture_sha256=donor['capture_sha256'],donor_request_id=donor['request_id'],
            role='training_donor_pair_provenance_not_target_request_lookup',
            same_idle_first_START_ns=d,second_START_width_ns=u,third_START_width_ns=2*u,
            worlds=worlds,busy_sender_transfer_qualified=False,
            explicit_assumption='Reuse idle d,w and joint x,d-x under changed schedule; not a fitted busy law'),
        resource_adapter='actual_worker_twoFIFO_reference_v4.resource_timeline_view',
        original_scorer_modified=False,original_dispatch_order_source_backed=True,
        original_transport_mechanism_numerically_qualified=False,
        protocol_initial_empty_pending_is_explicit_supplied_world_assumption=True,
        main_queue_provider_eligible=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        independent_prediction_accuracy_validated=False,validation_labels_read=False,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-report',required=True);parser.add_argument('--fifo-audit',required=True)
    parser.add_argument('--partial-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=build_report(args.v3_report,args.fifo_audit,args.partial_report)
    with Path(args.output).open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],sources=len(report['sources']))))
