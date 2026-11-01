"""Read-only natural request/phase/reader/worker-ready wall-time inventory."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from .owner_lifecycle_mark_v21 import receipt
from .unknown_preflight_archive_v1 import save_synced

ROUTES=('balanced','skewed','spill_low','spill_high')


def checked(ref):
    assert receipt(ref['path'])==ref
    return json.loads(Path(ref['path']).read_text())


def distribution(values):
    return dict(n=len(values),minimum_ns=min(values),median_ns=float(np.median(values)),
        p95_ns=float(np.percentile(values,95)),maximum_ns=max(values)) if values else dict(n=0)


def run(base,output):
    output=Path(output);assert not output.exists();output.mkdir()
    summaries=[];sources=[];ledger=[]
    for route in ROUTES:
        review_ref=receipt(Path(base)/route/'report.json');review=checked(review_ref)
        assert review['status']=='PASS' and review['issues']==[] and review['trace'] is None
        raw_ref=review['evidence']['raw.json'];raw=checked(raw_ref)
        plan=checked(review['evidence']['plan.json']);context=checked(review['evidence']['context.json'])
        host=checked(review['host_lifecycle']);assert host['source_raw']==raw_ref
        assert len(raw['request_records'])==57 and len(raw['priming_request_records'])==12
        assert raw['training_only'] and raw['route_id']==route and plan['route_id']==route
        actual_sources=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith(
            ('/generalist-specialist-project/gsdeploy/runtime/protocol.py','/generalist-specialist-project/gsdeploy/runtime/deployment.py'))]
        assert len(actual_sources)==2
        for ref in actual_sources:assert receipt(ref['path'])==ref
        sources.extend([review_ref,raw_ref,review['evidence']['plan.json'],review['evidence']['context.json'],review['host_lifecycle'],*actual_sources])
        allrows=raw['priming_request_records']+raw['request_records'];counts=Counter();byowner=defaultdict(list)
        result_ready=[];end_ready=[];end_result=[];phase_delivery=[];next_start_minus_ready=[];next_before_receipt=[]
        phase_examples=[];missing=[];host_rows={r['request_id']:r for r in host['rows']}
        for row in allrows:
            request=row['request_id'];owner=row['model'];worker=row['worker_response'];ready=row['previous_handler_next_request_ready']
            assert row['status']=='completed' and worker['request_id']==request and ready['request_id']==request
            start,end=worker['start_perf_ns'],worker['end_perf_ns'];receipt_clock=row['response_reader_event_enqueue_time_ns'];ready_clock=ready['perf_ns']
            assert start<end<=ready_clock<=ready['received_time_ns']
            assert row['completion_time_ns']==receipt_clock<=row['response_collector_dequeued_time_ns']<=row['response_collector_processing_time_ns']
            assert end<=receipt_clock
            assert ready['clock']=='host_perf_counter_ns' and ready['boundary']=='previous_handler_next_request_ready'
            assert ready['scope']=='after_response_flush_before_next_read' and ready['pid']==raw['ready'][owner]['pid']
            counts['worker_ready_before_result_reader']+=ready_clock<receipt_clock
            counts['worker_ready_after_result_reader']+=ready_clock>receipt_clock
            counts['worker_ready_equals_result_reader']+=ready_clock==receipt_clock
            if row in raw['request_records']:
                observed=host_rows[request]['offsets_ns'];origin=raw['phase_start_ns']
                assert observed['handler_end_ns']==end-origin and observed['parent_receipt_ns']==receipt_clock-origin
                assert observed['worker_next_ready_ns']==ready_clock-origin
                counts['measured']+=1
            else:counts['priming']+=1
            result_ready.append(receipt_clock-ready_clock);end_ready.append(ready_clock-end);end_result.append(receipt_clock-end)
            events=row['execution_events'];starts=[e for e in events if 'event' not in e]
            assert len(starts)==1 and starts[0]['perf_ns']==row['observed_worker_start_time_ns']
            assert start<=starts[0]['perf_ns']<end
            pairs=defaultdict(dict)
            for event in events:
                assert event['request_id']==request and start<=event['perf_ns']<=end
                assert event['perf_ns']<=event['received_time_ns']<=event['collector_dequeued_time_ns']<=event['collector_processing_time_ns']
                phase_delivery.append(event['received_time_ns']-event['perf_ns'])
                kind='START' if 'event' not in event else 'PHASE_'+event['event'].upper()
                ledger.append(dict(capture=raw_ref['sha256'],route=route,request_id=request,owner=owner,
                    event_kind=kind,segment_id=event.get('segment_id'),semantic_phase=event.get('semantic_phase'),
                    worker_event_ns=event['perf_ns'],reader_observation_ns=event['received_time_ns'],
                    collector_dequeued_ns=event['collector_dequeued_time_ns'],
                    collector_processing_ns=event['collector_processing_time_ns']))
                if 'event' in event:
                    assert event['event'] in ('start','end')
                    assert event['event'] not in pairs[event['segment_id']]
                    pairs[event['segment_id']][event['event']]=event
            for segment,pair in pairs.items():
                if set(pair)!=set(('start','end')):missing.append([request,segment,sorted(pair)]);continue
                assert pair['start']['perf_ns']<=pair['end']['perf_ns']
                counts['paired_Python_wall_segments']+=1
                counts['paired_semantic_wall_segments']+=pair['start'].get('semantic_phase') is not None
            for kind,event_clock,delivered in [('HANDLER_END',end,receipt_clock),('RESULT_READER',receipt_clock,receipt_clock),
                ('WORKER_NEXT_READY',ready_clock,ready['received_time_ns'])]:
                ledger.append(dict(capture=raw_ref['sha256'],route=route,request_id=request,owner=owner,
                    event_kind=kind,worker_event_ns=event_clock,reader_observation_ns=delivered))
            byowner[owner].append(row)
        for owner,rows in byowner.items():
            rows.sort(key=lambda r:r['worker_response']['start_perf_ns'])
            for previous,current in zip(rows,rows[1:]):
                ready_clock=previous['previous_handler_next_request_ready']['perf_ns'];start=current['worker_response']['start_perf_ns']
                assert ready_clock<=start
                next_start_minus_ready.append(start-ready_clock)
                next_before_receipt.append(start<previous['response_reader_event_enqueue_time_ns'])
                counts['next_request_dispatched_before_previous_ready']+=current['actual_submit_time_ns']<ready_clock
        summaries.append(dict(route=route,capture=raw_ref,counts=dict(counts),
            missing_phase_pairs=missing,
            result_reader_minus_worker_next_ready=distribution(result_ready),
            worker_next_ready_minus_handler_end=distribution(end_ready),
            result_reader_minus_handler_end=distribution(end_result),
            START_PHASE_reader_delivery_delay=distribution(phase_delivery),
            next_same_owner_handler_start_minus_previous_worker_ready=distribution(next_start_minus_ready),
            next_same_owner_handler_started_before_previous_result_reader_count=sum(next_before_receipt),
            observed_same_owner_serial_handler_order_verified=True,
            common_host_clock_contract=raw['clock_contract'],all_timestamps_positive_int=True))
    path=output/'observed-wall-event-ledger.jsonl.gz'
    with gzip.open(path,'wt') as handle:
        for event in sorted(ledger,key=lambda e:(e['capture'],e['reader_observation_ns'],e['request_id'],e['event_kind'])):
            handle.write(json.dumps(event,separators=(',',':'))+'\n')
    result=dict(schema='gsdeploy.natural_host_walltime_inventory_independent_CPU_review.v1',
        status='EXISTING_NATURAL_HOST_EVENTS_SUPPORT_DELIVERED_PREFIX_TRAINING_CANDIDATE',
        captures=summaries,sources=sources,review_implementation=receipt(__file__),event_ledger=receipt(path),
        request_completion_target='Parent RESULT reader event enqueue, equal to original completion_time_ns.',
        worker_availability_target='Separate WORKER_READY perf_ns after RESULT flush and before next stdin read; not RESULT reader receipt.',
        phase_target='Paired Python phase wall START/END; not kernel END, CUDA-ready, or SM work.',
        rolling_prefix_rule='At parent cutoff t use only events whose reader_observation_ns<=t (or collector_processing_ns<=t for collector-driven state). Worker perf timestamps from delayed events are not retroactively visible.',
        request_queue_rule='Per-owner stdin execution is serial. Parent dispatch/write completion can occur while prior handler runs. Preserve transport queue plus worker-ready and request result as separate events.',
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        data_sufficient_for_physical_SM_progress=False,calibrated_prediction_accuracy=False,
        limitations=['Only four independent route captures, reused input family; many phases per request are correlated, not independent replicates.',
            'Natural captures have no CUPTI/context trace. They identify request/phase wall outcomes and host observations, not kernel ready or effective service.',
            'Do not use final generated token counts, future END, later responses or future observed peer states as cutoff features.',
            'Closed-loop prediction requires models for not-yet-observed phase/request/worker-ready events and observation delay; replay of actual future events is diagnostic only.',
            'No counterfactual layout/backend/arrival support or statistical coverage is certified by this inventory.'])
    for ref in sources:assert receipt(ref['path'])==ref
    save_synced(output/'report.json',result)
    print(json.dumps(dict(report=receipt(output/'report.json'),captures=summaries)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.base,a.output)
