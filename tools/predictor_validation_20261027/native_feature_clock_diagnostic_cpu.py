"""Read-only native phase/queue observability and clock-order audit; no fit."""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sqlite3

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .native_collect import receipt
from .save_hazard_v13 import EDGES_NS
from .trace_export import linux_pid
from .unknown_preflight_archive_v1 import save_synced

OWNERS=('generalist_2#0','ocr#0','text#0','text#1')


def read(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Frozen receipt differs')
    return json.loads(Path(ref['path']).read_text())


def run(model_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    modelref=receipt(model_path);model=read(modelref)
    if model['training_only'] is not True or model['validation_labels_used'] is not False:raise ValueError('Original training-only model required')
    sources=[modelref];results=[];feature_counts=[]
    for capture,entry in model['captures'].items():
        projected=read(entry['report']);native=read(projected['source_adapter']);review=read(native['review'])
        if review['training_only'] is not True or review['status']!='PASS':raise ValueError('Training role before source read required')
        rawref=review['evidence']['raw.json'];raw=read(rawref);sqlref=review['trace']['sqlite']
        if receipt(sqlref['path'])!=sqlref:raise ValueError('Original sqlite drift')
        sources.extend([entry['report'],projected['source_adapter'],native['review'],rawref,sqlref,native['native_kernel_witnesses'],projected['risk_chronology']])
        records=raw['priming_request_records']+raw['request_records'];known={r['request_id']:r for r in records};measured={r['request_id'] for r in raw['request_records']}
        owners={row['pid']:owner for owner,row in raw['ready'].items()};meta={};anchor={}
        for record in records:
            worker=record['worker_response'];pid=raw['ready'][record['model']]['pid']
            for stage in worker['observed_stages']+worker.get('semantic_forward_observations',{}).get('records',[]):
                label=f"GS_STAGE:{record['request_id']}:{stage['kind']}:{stage['id']}"
                anchor[pid,label]=(stage['start_ns'],stage['end_ns']);meta[pid,label]=dict(stage,request_id=record['request_id'],owner=record['model'])
        conn=sqlite3.connect(Path(sqlref['path']).as_uri()+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
        clocks={pid:dict(owner=owner,matched_ranges=0,lower_ns=None,upper_ns=None) for pid,owner in owners.items()}
        phases={owner:[] for owner in OWNERS};semantic_stats=Counter();seen=set()
        for row in conn.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL'):
            pid=linux_pid(row['globalTid']);key=pid,row['label']
            if key not in anchor:continue
            if key in seen:raise ValueError('Duplicate native/worker stage mapping')
            seen.add(key);pstart,pend=anchor[key];lower=pend-row['end'];upper=pstart-row['start'];clock=clocks[pid]
            clock['matched_ranges']+=1;clock['lower_ns']=max(clock['lower_ns'],lower) if clock['lower_ns'] is not None else lower
            clock['upper_ns']=min(clock['upper_ns'],upper) if clock['upper_ns'] is not None else upper
            stage=meta[key]
            if stage['request_id'] in measured and 'semantic_phase' in stage:
                phases[stage['owner']].append((row['start'],row['end'],stage['semantic_phase']))
                semantic_stats['forward_ranges']+=1;semantic_stats[stage['semantic_phase']]+=1
                semantic_stats['known_input_sequence_length']+=int(type(stage.get('input_sequence_length')) is int)
                semantic_stats['known_forward_index']+=int(type(stage.get('forward_index')) is int)
        if seen!=set(anchor):raise ValueError('Every original stage anchor must match native NVTX')
        for pid,clock in clocks.items():
            clock['conditional_constant_offset_interval_nonempty']=clock['lower_ns']<=clock['upper_ns']
            clock['interval_width_ns']=clock['upper_ns']-clock['lower_ns']
            clock['old_median_offset_ns']=native['original_handler_projection_by_pid'][str(pid)]['offset_ns']
            clock['old_median_inside_order_interval']=clock['lower_ns']<=clock['old_median_offset_ns']<=clock['upper_ns']
        syncs=dict(Counter(row['name'] for row in conn.execute('SELECT s.value name FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION x JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON x.correlationId=r.correlationId AND (x.globalPid >> 24)=(r.globalTid >> 24) JOIN StringIds s ON s.id=r.nameId')))
        inventory={table:conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CUDA_EVENT','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')}
        event_times=conn.execute('SELECT COUNT(*) total,SUM(timestamp=0) zero FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT').fetchone()
        conn.close()
        for owner,values in phases.items():
            values.sort()
            if any(a[1]>b[0] for a,b in zip(values,values[1:])):raise ValueError('Expected serial language forwards per owner')
        starts={owner:[x[0] for x in values] for owner,values in phases.items()}
        queues={owner:dict(returned=[],started=[],ended=[]) for owner in OWNERS};kernel_stats=Counter()
        with gzip.open(native['native_kernel_witnesses']['path'],'rt') as stream:
            for line in stream:
                k=json.loads(line);kernel_stats['measured_kernels']+=1
                kernel_stats['no_certified_queued_ready_bound']+=int(k['certified_queued_ready_bound_ns'] is None)
                kernel_stats['all_dependencies_not_identified']+=int(not k['all_native_dependencies_identified'])
                kernel_stats['API_return_before_kernel_START']+=int(k['API_return_ns']<k['observed_start_ns'])
                q=queues[k['owner']];q['returned'].append(k['API_return_ns']);q['started'].append(k['observed_start_ns']);q['ended'].append(k['observed_end_ns'])
        for q in queues.values():
            for values in q.values():values.sort()
        counter=defaultdict(lambda:dict(exposure_ns=0,events=0));save_features=Counter();risk_ref=projected['risk_chronology']
        if receipt(risk_ref['path'])!=risk_ref:raise ValueError('Risk changed')
        with gzip.open(risk_ref['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if not row['exposure_qualified']:continue
                owner=row['held_owner'];mask=''.join('1' if row['shared_left_features']['handler_active'][o] else '0' for o in OWNERS)
                lo,hi=row['start_ns'],row['end_ns'];restore=row['restore_start_ns'];cuts={lo,hi}
                values=phases[owner];i=max(0,bisect_right(starts[owner],lo)-1)
                while i<len(values) and values[i][0]<hi:
                    a,b,_=values[i]
                    if lo<a<hi:cuts.add(a)
                    if lo<b<hi:cuts.add(b)
                    i+=1
                cuts.update(restore+edge for edge in EDGES_NS if lo<restore+edge<hi);cuts=sorted(cuts)
                for left,right in zip(cuts,cuts[1:]):
                    j=bisect_right(starts[owner],left)-1
                    phase=values[j][2] if j>=0 and values[j][0]<=left<values[j][1] else 'outside_language_forward'
                    agebin=bisect_right(EDGES_NS,left-restore)-1;cell=counter[owner,mask,agebin,phase]
                    cell['exposure_ns']+=right-left;cell['events']+=int(row['physical_SAVE_END_observed'] and right==hi)
                if row['physical_SAVE_END_observed']:
                    for other,q in queues.items():
                        # Trace-time truth only. Endpoints are revealed when
                        # their event occurs, but actual online delivery lag is
                        # not recorded/certified by these archived witnesses.
                        at=row['end_ns'];started=bisect_right(q['started'],at);ended=bisect_right(q['ended'],at);returned=bisect_right(q['returned'],at)
                        save_features[('held' if other==owner else 'peer')+'_started_unfinished_positive']+=int(started>ended)
                        save_features[('held' if other==owner else 'peer')+'_returned_minus_completed_positive']+=int(returned>ended)
        for (owner,mask,agebin,phase),value in sorted(counter.items()):feature_counts.append(dict(capture=capture,route=entry['route'],holder=owner,mask=mask,agebin=agebin,native_current_host_phase=phase,**value))
        if sum(v['exposure_ns'] for v in counter.values())!=sum(v['exposure_ns'] for v in model['by_capture'][capture].values()):raise ValueError('Native phase exposure conservation failed')
        if sum(v['events'] for v in counter.values())!=sum(v['events'] for v in model['by_capture'][capture].values()):raise ValueError('SAVE conservation failed')
        results.append(dict(route=entry['route'],capture=capture,clock_order_intervals_by_pid=clocks,semantic_metadata=dict(semantic_stats),
            native_inventory=inventory,sync_api_inventory=syncs,cuda_event_timestamp_zero=int(event_times['zero'] or 0),
            native_kernel_observability=dict(kernel_stats),trace_time_SAVE_state_associations=dict(save_features)))
    sources=strict_references([sources,implementation_sources(__file__)])
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen source changed')
    out.mkdir(parents=True,exist_ok=False);save_synced(out/'native-phase-age-risk-counts.json',feature_counts)
    result=dict(schema='gsdeploy.native_feature_and_clock_order_readonly_diagnostic.v1',status='FEATURE_CONTRACT_IDENTIFIED_NO_FIT',routes=results,
        native_phase_age_risk_counts=receipt(out/'native-phase-age-risk-counts.json'),original_age_edges_ns=EDGES_NS,
        clock_contract=dict(offset_definition='perf = native + delta',same_rate_constant_offset_hypothesis_interval='max(perf_END-native_POP) <= delta <= min(perf_START-native_PUSH)',
            source_order='NVTX push returns before perf START; perf END sampled before NVTX pop call',
            equal_rates_proven=False,monotonic_only='Only within-thread PUSH before perf START and perf END before POP order is certified; monotonicity alone gives no global additive offset or drift bound.',
            handler_projection='Request handler boundaries are not bracketed directly by request NVTX; any projection still transfers a clock model beyond stage anchors.'),
        minimum_training_features=['Held owner and attained context age; do not reset on feature updates.',
            'Per-owner current observed language forward phase, forward index, input sequence length and batch, from BEGIN metadata; ended forwards move to outside/none without erasing queued GPU work.',
            'Per-owner issued API-return count minus confirmed completed activity count; keep source timestamps and notification/delivery times distinct.',
            'Same-stream known predecessors plus exact (process,context,eventId,eventSyncId) record/wait identity; only completed predecessors certify queued readiness.',
            'Known/unknown masks for issued, completion and dependency status; no future kernel end or future template node promoted to current fact.'],
        minimum_collection_changes=['New observer variant emits perf-before-push/perf-after-push and perf-before-pop/perf-after-pop plus PID/thread/sequence, using no new CUDA synchronize.',
            'Add request BEGIN/END NVTX brackets with perf-side brackets; preserve READY/handler/RESULT/next-ready as separate events.',
            'Transmit forward BEGIN shape/index metadata immediately, with worker event time and parent received/dequeued/processed timestamps.',
            'To expose real queued readiness, collect launch/stream/event dependency identity and nonblocking completion confirmations plus their availability time; absence is UNKNOWN.',
            'At runtime engine expose issued node cursor, completed predecessor set and current semantic segment; unreleased nodes and future cohort remain excluded.'],
        explanation_limit='Native semantic phase and workload shape are available candidate conditioning features. Counts alone do not establish that they explain route effects. Queue/ready observability remains incomplete, so no new calibrated hazard or physical service claim is made.',
        sources=sources,source_bytes_unchanged=True,fit_performed=False,model_parameters_changed=False,GPU_started=False,validation_labels_read=False)
    save_synced(out/'report.json',result);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.model,args.output)))
