"""CPU-only common-clock context-envelope and launch-queue decomposition.

Envelope coverage is an availability surrogate, never SM service. Missing
RESTORE/SAVE events stay unresolved. CUDA queue intervals retain same-stream
predecessor exposure and are not labelled wholly as context wait.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt, save
from .trace_export import linux_pid
from .unified_corun_review_v8_1 import load_review, LIGHT


class IntervalUnion:
    def __init__(self, intervals):
        merged = []
        for lo, hi in sorted(intervals):
            if type(lo) is not int or type(hi) is not int or hi < lo:
                raise ValueError('Ordered integer observed intervals required')
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        self.intervals = merged
        self.starts = [a for a, _ in merged]
        self.ends = [b for _, b in merged]
        self.prefix = [0]
        for a, b in merged:
            self.prefix.append(self.prefix[-1] + b - a)

    def covered_before(self, t):
        i = bisect_right(self.ends, t)
        total = self.prefix[i]
        if i < len(self.starts):
            total += max(0, t - self.starts[i])
        return total

    def overlap(self, lo, hi):
        if hi < lo:
            raise ValueError('Reversed query')
        return self.covered_before(hi) - self.covered_before(lo)

    def intersection(self, other):
        result = []
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i]; c, d = other.intervals[j]
            if max(a, c) < min(b, d):
                result.append((max(a, c), min(b, d)))
            if b <= d:
                i += 1
            else:
                j += 1
        return IntervalUnion(result)


def partition(lo, hi, own, others, both):
    own_ns = own.overlap(lo, hi); other_ns = others.overlap(lo, hi)
    both_ns = both.overlap(lo, hi)
    result = dict(own_only_ns=own_ns-both_ns, other_only_ns=other_ns-both_ns,
                  overlapping_envelopes_ns=both_ns,
                  unresolved_ns=hi-lo-own_ns-other_ns+both_ns)
    if min(result.values()) < 0 or sum(result.values()) != hi-lo:
        raise ValueError('Envelope partition arithmetic failed')
    return result


def context_intervals(connection):
    tags = {r['id']: r['name'] for r in connection.execute('select * from ENUM_GPU_CTX_SWITCH')}
    events = defaultdict(list)
    for r in connection.execute('select * from GPU_CONTEXT_SWITCH_EVENTS order by timestamp,seqNo'):
        events[(r['vmId'], r['gpuId'], r['contextId'], r['globalPid'])].append(dict(r))
    brackets = {}; issues = []
    for key, rows in events.items():
        opened = None; intervals = []
        for r in rows:
            tag = tags[r['tag']]
            if tag == 'RESTORE_START':
                if opened is not None:
                    issues.append(dict(context=list(key), reason='repeated_restore_unresolved', timestamp=opened))
                opened = r['timestamp']
            elif tag == 'SAVE_END':
                if opened is None:
                    issues.append(dict(context=list(key), reason='unpaired_save_unresolved', timestamp=r['timestamp']))
                else:
                    intervals.append((opened, r['timestamp'])); opened = None
            else:
                raise ValueError('Unverified scheduler tag: '+tag)
        if opened is not None:
            issues.append(dict(context=list(key), reason='unpaired_restore_unresolved', timestamp=opened))
        brackets[key] = intervals
    return events, brackets, issues


def controlled_creation(connection, raw, events, brackets):
    """Creation window correspondence, cross-checked independently by priming."""
    aliases = set(raw['ready']); identity = raw['live_identity_events']
    baseline = identity[0]
    if baseline['phase'] != 'pre_startup' or baseline['launchers'] or baseline['ready'] or baseline['nvml_processes_mib']:
        raise ValueError('Observed empty startup baseline required')
    startup = [e for e in identity if e['phase']=='startup_begin' and set(e['ready']) != aliases]
    all_ready = next(e for e in identity if set(e['ready']) == aliases)
    sessions = list(connection.execute('select * from TARGET_INFO_SESSION_START_TIME'))
    if len(sessions) != 1 or len(startup) != 4 or len(events) != 4:
        raise ValueError('Single session with four controlled creation windows required')
    base = sessions[0]['systemClockNs']; first = {k:min(r['timestamp'] for r in v) for k,v in events.items()}
    previous = set(); mapping = {}; windows = []
    for i, row in enumerate(startup):
        current = set(row['launchers'])
        if not current >= previous or len(current-previous) != 1 or set(row['ready']) != previous:
            raise ValueError('Single newly launched owner per window required')
        alias = next(iter(current-previous)); launcher = row['launchers'][alias]
        if (launcher['pid'] != raw['ready'][alias]['pid']
                or launcher['start_ticks'] != raw['live_owned_processes'][alias]['start_ticks']
                or launcher['execution_nonce'] != raw['controller_nonce']):
            raise ValueError('Owner nonce/start identity changed')
        begin = row['time_ns']; end = (startup[i+1] if i+1<len(startup) else all_ready)['time_ns']
        matches = [k for k,t in first.items() if begin < base+t < end]
        if len(matches) != 1:
            raise ValueError('Unique first-context creation correspondence required')
        mapping[alias] = matches[0]
        windows.append(dict(alias=alias, context=list(matches[0]), begin_perf_ns=begin, end_perf_ns=end,
                            first_context_margin_ns=min(base+first[matches[0]]-begin,end-base-first[matches[0]])))
        previous = current
    if len(set(mapping.values())) != 4:
        raise ValueError('Distinct controlled context correspondence required')
    strings = {r['id']:r['value'] for r in connection.execute('select id,value from StringIds')}
    ranges = defaultdict(list)
    for r in connection.execute('select globalTid,start,end,text,textId from NVTX_EVENTS where end is not null'):
        ranges[linux_pid(r['globalTid'])].append((r['start'],r['end'],r['text'] or strings.get(r['textId'],'')))
    kernels = defaultdict(list)
    for r in connection.execute('select globalPid,start,end from CUPTI_ACTIVITY_KIND_KERNEL'):
        kernels[linux_pid(r['globalPid'])].append((r['start'],r['end']))
    indexes = {k:IntervalUnion(v) for k,v in brackets.items()}; cross_checks=[]
    for row in raw['priming_request_records']:
        alias=row['model']; pid=raw['ready'][alias]['pid']; prefix='GS_STAGE:'+row['request_id']+':'
        selected=[(a,b) for a,b,text in ranges[pid] if text.startswith(prefix)]
        if not selected:raise ValueError('Missing original priming NVTX')
        lo=min(a for a,b in selected); hi=max(b for a,b in selected)
        points=[t for a,b in kernels[pid] if lo<=a and b<=hi for t in (a,b)]
        if not points:raise ValueError('Missing independent priming kernels')
        hits={k:sum(any(a<=t<=b for a,b in idx.intervals[max(0,bisect_right(idx.starts,t)-1):bisect_right(idx.starts,t)]) for t in points)
              for k,idx in indexes.items()}
        if hits[mapping[alias]] != len(points) or any(v for k,v in hits.items() if k!=mapping[alias]):
            raise ValueError('Priming endpoint correspondence not uniquely supported')
        cross_checks.append(dict(alias=alias,request_id=row['request_id'],endpoints=len(points),own_hits=len(points)))
    if len(cross_checks)!=12:raise ValueError('Original twelve priming requests required')
    return mapping, dict(status='UNIQUE_CONTROLLED_TEMPORAL_CORRESPONDENCE',creation_windows=windows,
                         priming_cross_checks=cross_checks,direct_OS_namespace_identity=False,
                         bracket_equals_SM_service=False)


def analyze(review_path, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report,plan,raw=load_review(review_path,LIGHT)
    sql=report['trace']['sqlite']; parsed_ref=report['trace']['parsed']
    parsed=json.loads(Path(parsed_ref['path']).read_text())
    sources=[receipt(review_path),*report['evidence'].values(),sql,parsed_ref,receipt(__file__)]
    connection=sqlite3.connect('file:'+sql['path']+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    events,brackets,issues=context_intervals(connection)
    mapping,attribution=controlled_creation(connection,raw,events,brackets)
    owners={v['pid']:a for a,v in raw['ready'].items()}
    unions={a:IntervalUnion(brackets[k]) for a,k in mapping.items()}
    others={a:IntervalUnion(v for other,k in mapping.items() if other!=a for v in brackets[k]) for a in mapping}
    both={a:unions[a].intersection(others[a]) for a in mapping}
    launches={}
    for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        key=linux_pid(r['globalTid']),r['correlationId']
        if key in launches:raise ValueError('Ambiguous runtime correlation')
        launches[key]=dict(r)
    stream_previous={};predecessors={}
    for r in connection.execute('select globalPid,correlationId,streamId,start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start,end'):
        pid=linux_pid(r['globalPid']);key=pid,r['streamId']
        predecessors[pid,r['correlationId'],r['start'],r['end']]=stream_previous.get(key)
        stream_previous[key]=max(stream_previous.get(key,0),r['end'])
    rows={r['request_id']:r for r in raw['request_records']};summaries=defaultdict(Counter);examples=[];seen=set()
    semantic={r['request_id']:{x['id'] for x in r['worker_response'].get('semantic_forward_observations',{}).get('records',[])} for r in rows.values()}
    forwards=defaultdict(list)
    for p in parsed['phases']:
        if p['request_id'] in rows and p['segment_id'] in semantic[p['request_id']]:forwards[p['request_id']].append(p)
    for phase in parsed['phases']:
        rid=phase['request_id']
        if rid not in rows:continue
        row=rows[rid];alias=row['model'];pid=raw['ready'][alias]['pid']
        for kernel in phase['kernels']:
            lo,hi=kernel['nsys_start_ns'],kernel['nsys_end_ns'];corr=kernel['correlation_id']
            key=pid,corr,lo,hi
            if key in seen:raise ValueError('Duplicate canonical measured kernel')
            seen.add(key);api=launches[pid,corr]
            matching=[p for p in forwards[rid] if p['nsys_start_ns']<=api['start'] and api['end']<=p['nsys_end_ns']]
            if len(matching)>1:raise ValueError('Ambiguous launch forward membership')
            group=matching[0]['segment_id'] if matching else '__outside_language_forwards__'
            summary=summaries[rid,alias,row['task'],row['pool_id'],group]
            lifetime=partition(lo,hi,unions[alias],others[alias],both[alias])
            summary.update({'lifetime_'+k:v for k,v in lifetime.items()});summary['kernel_count']+=1;summary['kernel_lifetime_sum_ns']+=hi-lo
            signed=lo-api['end'];summary['launch_exit_to_start_signed_sum_ns']+=signed
            summary['negative_launch_exit_to_start_count']+=signed<0
            summary['launch_entry_to_start_sum_ns']+=lo-api['start']
            if signed>=0:
                queue=partition(api['end'],lo,unions[alias],others[alias],both[alias])
                summary.update({'launch_queue_'+k:v for k,v in queue.items()});summary['nonnegative_launch_queue_sum_ns']+=signed
                prev=predecessors[key]
                pred_overlap=max(0,min(prev,lo)-api['end']) if prev is not None else 0
                summary['previous_same_stream_kernel_lifetime_overlap_ns']+=pred_overlap
            else:queue=None;pred_overlap=None
            examples.append(dict(request_id=rid,alias=alias,phase=group,kernel_correlation_id=corr,
                kernel_name=kernel['name'],start_ns=lo,end_ns=hi,lifetime_partition=lifetime,
                launch_exit_to_start_signed_ns=signed,launch_queue_partition=queue,
                previous_same_stream_kernel_lifetime_overlap_ns=pred_overlap))
    connection.close()
    summary_rows=[dict(request_id=k[0],alias=k[1],task=k[2],pool_id=k[3],phase=k[4],**v) for k,v in summaries.items()]
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Diagnostic source changed')
    result=dict(schema='gsdeploy.context_envelope_decomposition.v8',status='DESCRIPTIVE_CPU_REVIEW_COMPLETE',sources=sources,
        controlled_temporal_attribution=attribution,pairing_issues=issues,measured_kernel_count=len(seen),
        measured_request_count=len(rows),phase_summaries=summary_rows,
        largest_other_envelope_lifetime_examples=sorted(examples,key=lambda r:r['lifetime_partition']['other_only_ns'],reverse=True)[:40],
        kernel_lifetime_sum_includes_stream_overlap=True,queue_sums_include_overlapping_pending_kernels=True,
        partition_is_unique_wall_time_within_each_kernel=True,same_stream_predecessor_is_kernel_only=True,
        exact_device_eligibility_identified=False,missing_events_filled=False,physical_SM_service_identified=False,
        fit_performed=False,validation_labels_used=False,GPU_started=False)
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=analyze(args.review,args.output)
    print(json.dumps({k:r[k] for k in ('status','measured_kernel_count','measured_request_count','pairing_issues')}))


if __name__=='__main__':main()
