"""Read-only training host gaps and existing SQLite launch evidence; no GPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .stage_response_features_v1 import checked, receipt
from .stage_host_partition_v1 import extract as reviewed_partition


def gaps(row):
    semantic = row['worker_response'].get('semantic_forward_observations', {})
    forwards = sorted(semantic.get('records', []), key=lambda r: r['forward_index'])
    if not forwards:
        return []
    native = next(s for s in row['worker_response']['observed_stages'] if s['id'] == 'native_inference')
    bounds = [('entry', None, forwards[0]['id'], native['start_ns'], forwards[0]['start_ns'])]
    bounds.extend(('transition', a['id'], b['id'], a['end_ns'], b['start_ns'])
                  for a, b in zip(forwards, forwards[1:]))
    bounds.append(('exit', forwards[-1]['id'], None, forwards[-1]['end_ns'], native['end_ns']))
    result = []
    for ordinal, (kind, previous, following, left, right) in enumerate(bounds):
        if not left <= right:
            raise ValueError('Nonmonotone source gap')
        queries = [dict(e) for e in row['execution_events']
                   if e.get('kind') == 'device_event_query' and left <= e['perf_ns'] < right]
        result.append(dict(request_id=row['request_id'], owner=row['model'], pool_id=row['pool_id'],
            job_id=row['profile_job_id'], gap_ordinal=ordinal, kind=kind,
            previous_phase=previous, following_phase=following, left_ns=left, right_ns=right,
            duration_ns=right-left, queries=queries,
            false_queries=sum(q['end_event_complete'] is False for q in queries),
            true_queries=sum(q['end_event_complete'] is True for q in queries),
            is_pure_CPU_service=False, implies_SM_idle=False))
    return result


def union_ns(intervals):
    end, total = None, 0
    for left, right in sorted(intervals):
        if left >= right:
            continue
        if end is None or left > end:
            total += right-left
        elif right > end:
            total += right-end
        end = right if end is None else max(end, right)
    return total


def trace_owner(export_path, rows, raw_ref, ready_pid):
    """Use one exact NVTX trace clock; never sum nested parent/child kernels."""
    export = checked(receipt(export_path))
    if export['raw'] != raw_ref or {row['model'] for row in rows} != {export['owner']}:
        raise ValueError('Exact original kernel raw and owner export binding required')
    for key in ('source', 'output'):
        ref = export['export'][key]
        if receipt(ref['path']) != ref:
            raise ValueError('Changed existing trace export')
    conn = sqlite3.connect('file:' + export['export']['output']['path'] + '?mode=ro', uri=True)
    try:
        names = dict(conn.execute('SELECT id,value FROM StringIds'))
        nvtx = {}
        for left, right, text, tid, textid in conn.execute(
                'SELECT start,end,text,globalTid,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
            name = text if text is not None else names.get(textid)
            if name and name.startswith('GS_STAGE:profile:'):
                nvtx.setdefault(name, []).append((left, right, tid))
        requests = []
        for row in rows:
            source_gaps = gaps(row)
            if not source_gaps:
                continue
            rid = row['request_id']

            def interval(phase):
                matches = nvtx.get('GS_STAGE:' + rid + ':gpu:' + phase, [])
                if len(matches) != 1:
                    raise ValueError('Exact unique request/phase NVTX identity required')
                return matches[0]

            left, right, tid = interval('native_inference')
            if (tid >> 24) & 0xffffff != ready_pid:
                raise ValueError('NVTX owner PID differs from actual READY worker')
            pid = tid & ~0xffffff
            records = row['worker_response']['semantic_forward_observations']['records']
            ranges = {record['id']: interval(record['id']) for record in records}
            if any(r[2] != tid for r in ranges.values()):
                raise ValueError('Native and forward thread identity mismatch')
            runtime = [dict(start=a, end=b, correlation=c, name=names[n]) for a,b,c,n in conn.execute(
                'SELECT start,end,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME '
                'WHERE globalTid=? AND start>=? AND end<=? ORDER BY start', (tid, left, right))]
            by_correlation = defaultdict(list)
            for api in runtime:
                by_correlation[api['correlation']].append(api)
            kernels = [dict(identity=identity,start=a,end=b,correlation=c,name=names[n],stream=stream)
                       for identity,a,b,c,n,stream in conn.execute(
                'SELECT rowid,start,end,correlationId,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL '
                'WHERE globalPid=? AND ((end>=? AND start<=?) OR correlationId IN '
                '(SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME '
                'WHERE globalTid=? AND start>=? AND end<=?))', (pid, left, right, tid, left, right))]
            copies = [dict(identity=identity,start=a,end=b,correlation=c,bytes=size,kind=kind,stream=stream)
                      for identity,a,b,c,size,kind,stream in conn.execute(
                'SELECT rowid,start,end,correlationId,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE globalPid=? AND ((end>=? AND start<=?) OR correlationId IN '
                '(SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME '
                'WHERE globalTid=? AND start>=? AND end<=?))', (pid, left, right, tid, left, right))]
            gap_results = []
            launched_ids = set()
            for source_gap in source_gaps:
                previous, following = source_gap['previous_phase'], source_gap['following_phase']
                a = ranges[previous][1] if previous else left
                b = ranges[following][0] if following else right
                gap_apis = [api for api in runtime if a <= api['start'] and api['end'] <= b]
                gap_correlations = {api['correlation'] for api in gap_apis}
                launched = [k for k in kernels if k['correlation'] in gap_correlations
                            and len(by_correlation[k['correlation']]) == 1]
                launched_copies = [c for c in copies if c['correlation'] in gap_correlations
                                  and len(by_correlation[c['correlation']]) == 1]
                if launched_ids & {k['identity'] for k in launched}:
                    raise ValueError('A native kernel launch was attributed to two disjoint gaps')
                launched_ids.update(k['identity'] for k in launched)
                overlapping = [k for k in kernels if k['start'] < b and a < k['end']]
                carry = [k for k in overlapping if k['correlation'] not in gap_correlations]
                gap_results.append(dict(kind=source_gap['kind'],ordinal=source_gap['gap_ordinal'],
                    previous_phase=previous,following_phase=following,
                    trace_NVTX_interval_ns=[a,b],trace_counterpart_wall_ns=b-a,
                    runtime_API_counts=dict(Counter(api['name'] for api in gap_apis)),
                    runtime_API_wall_union_ns=union_ns((api['start'],api['end']) for api in gap_apis),
                    gap_launched_unique_kernels=len(launched),
                    gap_launched_kernel_rowids=[k['identity'] for k in launched],
                    gap_launched_kernel_names=dict(Counter(k['name'] for k in launched)),
                    gap_launched_kernels_ending_after_trace_gap=sum(k['end']>b for k in launched),
                    owner_kernel_execution_overlap_union_ns=union_ns((max(a,k['start']),min(b,k['end'])) for k in overlapping),
                    overlap_from_other_launch_intervals=len(carry),
                    gap_launched_copy_count=len(launched_copies),
                    gap_launched_copy_kinds=dict(Counter(c['kind'] for c in launched_copies)),
                    gap_launched_copy_bytes_by_kind={str(kind):sum(c['bytes'] for c in launched_copies if c['kind']==kind)
                                                   for kind in sorted({c['kind'] for c in launched_copies})},
                    gap_launched_copy_bytes=sum(c['bytes'] for c in launched_copies),
                    gap_launched_copy_records=launched_copies))
            requests.append(dict(request_id=rid,owner=row['model'],pid=row['worker_response']['pid']
                if 'pid' in row['worker_response'] else (tid>>24)&0xffffff,
                trace_globalTid=tid,native_NVTX_interval_ns=[left,right],gaps=gap_results))
        return dict(owner=export['owner'],export_receipt=receipt(export_path),
            sqlite=export['export']['output'],source_rep=export['export']['source'],requests=requests,
            interval_scope='Exact same-trace NVTX counterparts, not worker occurrence endpoints or natural timing.',
            launch_scope='Unique SQLite kernel row identity via one same-PID runtime correlation; no parent/child summation.',
            activity_scope='Owner-local kernel execution overlap only; not continuous SM service, not global GPU idle.',
            query_numerical_authentication=False)
    finally:
        conn.close()


def summarize(rows):
    result = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['owner'],row['kind'],row['previous_phase'],row['following_phase'])].append(row)
    for key, members in sorted(grouped.items(),key=lambda x:str(x[0])):
        result.append(dict(owner=key[0],kind=key[1],previous_phase=key[2],following_phase=key[3],
            rows=len(members),duration_ns=sum(x['duration_ns'] for x in members),
            false_queries=sum(x['false_queries'] for x in members),
            true_queries=sum(x['true_queries'] for x in members)))
    return result


def review(natural_review, kernel_review, trace_directory):
    natural = reviewed_partition(natural_review)
    kernel = reviewed_partition(kernel_review)
    refs = [receipt(natural_review),receipt(kernel_review)]
    documents = [checked(ref) for ref in refs]
    if [doc['role']['mode'] for doc in documents] != ['natural','kernel']:
        raise ValueError('Separate exact natural and kernel timing domains required')
    raws = [checked(doc['inputs']['raw.json']) for doc in documents]
    all_gaps = [sum((gaps(row) for row in raw['request_records']),[]) for raw in raws]
    for partition, cohort_gaps in zip((natural,kernel),all_gaps):
        expected = sum(piece['duration_ns'] for row in partition['requests'] for piece in row['pieces']
                       if piece['label']=='forward_external_gap_in_native_inference')
        if sum(g['duration_ns'] for g in cohort_gaps) != expected:
            raise ValueError('Gap decomposition must exactly match the reviewed disjoint partition')
    trace = []
    for owner in ('text#0','text#1','generalist_2#0'):
        path = Path(trace_directory)/(owner.replace('#','-')+'-export.json')
        trace.append(trace_owner(path,[r for r in raws[1]['request_records'] if r['model']==owner],
                                 documents[1]['inputs']['raw.json'],raws[1]['ready'][owner]['pid']))
    return dict(schema='gsdeploy.native_inference_gap_evidence_review.v1',
        status='TRAINING_GAP_TOPOLOGY_AND_PARTIAL_NATIVE_LAUNCH_EVIDENCE_ONLY',
        sources=[*refs,*[doc['inputs']['raw.json'] for doc in documents],receipt(__file__),
                 receipt(Path(__file__).with_name('stage_host_partition_v1.py')),
                 receipt(Path(__file__).with_name('stage_response_features_v1.py'))],
        natural=dict(gaps=all_gaps[0],summary=summarize(all_gaps[0]),
            total_gap_ns=sum(g['duration_ns'] for g in all_gaps[0]),
            total_worker_occupancy_ns=sum(r['worker_handler_to_ready_ns'] for r in natural['requests'])),
        kernel=dict(gaps=all_gaps[1],summary=summarize(all_gaps[1]),traces=trace),
        kernel_natural_durations_pooled=False,fit_performed=False,GPU_started=False,
        validation_labels_read=False,native_CUDA_API_numerical_authentication=False,
        main_queue_provider_eligible=False,final_gain_eligible=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-review',required=True)
    parser.add_argument('--kernel-review',required=True)
    parser.add_argument('--trace-directory',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = review(args.natural_review,args.kernel_review,args.trace_directory)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt(args.output)))
