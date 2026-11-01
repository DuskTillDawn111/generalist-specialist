"""Training-only owned CUDA endpoint/queue evidence; no SM service inference."""
from __future__ import annotations

import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import context_intervals,IntervalUnion
from .owner_lifecycle_mark_v21 import receipt
from .idle_owner_selection_audit_v20 import read
from .profiling import digest
from .trace_export import linux_pid

SCHEMA='gsdeploy.physical_progress_endpoint_evidence.v26'


class ContextIndex:
    def __init__(self,envelopes,issue_times=()):
        self.rows=sorted(envelopes)
        if any(a[1]>b[0] for a,b in zip(self.rows,self.rows[1:])):
            raise ValueError('Nonoverlapping paired context envelopes required')
        self.starts=[row[0] for row in self.rows]
        self.boundaries={clock for begin,end,_ in self.rows for clock in (begin,end)}
        self.issue_times=sorted(issue_times)
        self.by_owner={owner:IntervalUnion([(begin,end) for begin,end,alias in self.rows if alias==owner])
            for owner in sorted({row[2] for row in self.rows})}

    def at(self,clock):
        i=bisect_right(self.starts,clock)-1
        if i<0 or clock>=self.rows[i][1]:
            return dict(owner=None,context_age_ns=None,status='UNRESOLVED_CONTEXT_GAP',boundary=clock in self.boundaries)
        begin,end,owner=self.rows[i]
        dirty=bisect_right(self.issue_times,end)>bisect_left(self.issue_times,begin)
        return dict(owner=owner,context_age_ns=clock-begin,
            status='UNPAIRED_EVENT_IN_ENVELOPE' if dirty else 'QUALIFIED_PAIRED_CONTEXT',
            boundary=clock in self.boundaries)

    def partition(self,begin,end,owner):
        if end<begin:raise ValueError('Ordered observed interval required')
        own=self.by_owner[owner].overlap(begin,end)
        others={alias:index.overlap(begin,end) for alias,index in self.by_owner.items() if alias!=owner}
        unresolved=end-begin-own-sum(others.values())
        if unresolved<0:raise ValueError('Context partition overlap')
        return dict(own_envelope_ns=own,other_envelope_ns=sum(others.values()),
            other_envelope_ns_by_owner={alias:value for alias,value in others.items() if value},
            gap_ns=unresolved)


class KernelPrefix:
    def __init__(self,by_owner):
        self.starts={owner:sorted(begin for begin,end in rows) for owner,rows in by_owner.items()}
        self.ends={owner:sorted(end for begin,end in rows) for owner,rows in by_owner.items()}
        self.unions={owner:IntervalUnion(rows) for owner,rows in by_owner.items()}

    def outstanding_started_by_owner(self,clock):
        return {owner:bisect_right(starts,clock)-bisect_right(self.ends[owner],clock)
            for owner,starts in self.starts.items()}


class ContextPrefix:
    """Raw RESTORE/SAVE markers consumed through cutoff, no future pairing."""
    def __init__(self,events):
        by_owner=defaultdict(list)
        for clock,sequence,owner,kind in sorted(events):
            if kind not in ('RESTORE_START','SAVE_END'):raise ValueError('Known context event kind required')
            by_owner[owner].append((clock,sequence,kind))
        self.rows=dict(by_owner)
        self.clocks={owner:[row[0] for row in rows] for owner,rows in self.rows.items()}

    def at(self,clock):
        active=[];last=[]
        for owner,rows in self.rows.items():
            i=bisect_right(self.clocks[owner],clock)-1
            if i<0:continue
            event_clock,_,kind=rows[i]
            last.append((owner,event_clock,kind))
            if kind=='RESTORE_START':active.append((owner,event_clock))
        if len(active)==1:
            owner,restore=active[0]
            return dict(owner=owner,context_age_ns=clock-restore,
                status='UNIQUE_OPEN_CONTEXT_PREFIX_NOT_SM_SERVICE',open_contexts=active)
        return dict(owner=None,context_age_ns=None,
            status='NO_OPEN_CONTEXT_PREFIX' if not active else 'MULTIPLE_OPEN_CONTEXT_PREFIXES',
            open_contexts=active)


def evidence(kernel,api,index,prefix,owner,context_prefix=None):
    begin,end=kernel['start'],kernel['end']
    if begin>=end:raise ValueError('Positive CUPTI kernel lifetime required')
    entered,returned=api['start'],api['end']
    if entered>returned:raise ValueError('Ordered source API interval required')
    start_state,end_state=index.at(begin),index.at(end)
    interval=index.partition(begin,end,owner)
    before_start=index.partition(returned,begin,owner) if returned<=begin else None
    overlap={other:union.overlap(begin,end) for other,union in prefix.unions.items() if other!=owner}
    features=dict(cutoff_ns=returned,owner=owner,source_launch_API_entry_ns=entered,
        source_launch_API_return_ns=returned,context_at_cutoff=index.at(returned),
        already_observed_kernel_start_at_cutoff=begin<=returned,
        already_observed_kernel_end_at_cutoff=end<=returned,
        observed_started_not_ended_kernel_count_by_owner=prefix.outstanding_started_by_owner(returned))
    # Paired window qualification can use a future SAVE or pairing issue. It is
    # diagnostic annotation, never a causal covariate. Only raw event prefix is.
    paired_cutoff=features.pop('context_at_cutoff')
    features['context_event_prefix_at_cutoff']=(context_prefix.at(returned) if context_prefix is not None else
        dict(owner=None,context_age_ns=None,status='CONTEXT_PREFIX_NOT_SUPPLIED',open_contexts=[]))
    labels=dict(observed_kernel_start_ns=begin,observed_kernel_end_ns=end,
        launch_API_return_to_kernel_start_signed_ns=begin-returned,
        kernel_start_to_end_lifetime_ns=end-begin,
        launch_API_return_to_observed_end_signed_ns=end-returned,
        context_at_kernel_start=start_state,context_at_kernel_end=end_state,
        paired_context_at_launch_return_diagnostic=paired_cutoff,
        observed_lifetime_context_partition=interval,
        nonnegative_return_to_start_context_partition=before_start,
        other_owner_kernel_lifetime_union_overlap_ns={alias:value for alias,value in overlap.items() if value})
    return dict(prefix_features=features,observation_labels=labels)


def run(supervision_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    supervision=read(supervision_path)
    if supervision.get('training_only') is not True or supervision.get('validation_labels_used') is not False:
        raise ValueError('Training-only source supervision required')
    implementation=[receipt(__file__)]
    from . import context_envelope_diagnostic_v8,owner_lifecycle_mark_v21
    implementation += [receipt(context_envelope_diagnostic_v8.__file__),receipt(owner_lifecycle_mark_v21.__file__)]
    before=[receipt(supervision_path),*implementation]
    output.mkdir(parents=True)
    all_counts=Counter();strata=defaultdict(Counter);pairs=defaultdict(Counter);examples=[];vocabulary={};captures=[]
    row_path=output/'owned-kernel-prefix-labels.jsonl.gz'
    with gzip.open(row_path,'wt',encoding='utf-8') as stream:
        for audit in supervision['training_audits']:
            for key in ('raw','report'):
                if receipt(audit[key]['path'])!=audit[key]:raise ValueError('Changed audited training source')
                before.append(audit[key])
            raw,report=read(audit['raw']['path']),read(audit['report']['path'])
            capture=audit['raw']['sha256'];counts=Counter()
            if report['controlled_temporal_attribution']['status']!='UNIQUE_CONTROLLED_TEMPORAL_CORRESPONDENCE':
                raise ValueError('Qualified controlled owner attribution required')
            refs=[next(ref for ref in report['sources'] if ref['path'].endswith('.sqlite')),
                next(ref for ref in report['sources'] if Path(ref['path']).name=='parsed-measured.json'),
                report['kernel_ready_observations']]
            for ref in refs:
                if receipt(ref['path'])!=ref:raise ValueError('Changed qualified kernel/context source')
                before.append(ref)
            parsed=read(refs[1]['path']);observations=read(refs[2]['path'])['rows']
            request_rows={row['request_id']:row for row in raw['request_records']}
            ready={(row['request_id'],row['observed_start_ns'],row['observed_end_ns']):row for row in observations}
            if len(ready)!=len(observations):raise ValueError('Ambiguous measured kernel observation')
            with sqlite3.connect(Path(refs[0]['path']).as_uri()+'?mode=ro',uri=True) as connection:
                connection.row_factory=sqlite3.Row
                events,brackets,issues=context_intervals(connection)
                mapping={row['alias']:tuple(row['context']) for row in report['controlled_temporal_attribution']['creation_windows']}
                tags={row['id']:row['name'] for row in connection.execute('SELECT * FROM ENUM_GPU_CTX_SWITCH')}
                event_prefix=ContextPrefix([(row['timestamp'],row['seqNo'],owner,tags[row['tag']])
                    for owner,context in mapping.items() for row in events[context]])
                index=ContextIndex([(begin,end,owner) for owner,context in mapping.items() for begin,end in brackets[context]],
                    [row['timestamp'] for row in issues])
                apis={}
                for row in connection.execute('SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
                    key=linux_pid(row['globalTid']),row['correlationId']
                    if key in apis:raise ValueError('Ambiguous runtime correlation')
                    apis[key]=dict(row)
                pid_owners={value['pid']:owner for owner,value in raw['ready'].items()}
                all_kernels=defaultdict(list);stream_previous={};previous={}
                for row in connection.execute('SELECT start,end,globalPid,correlationId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start,end'):
                    pid=linux_pid(row['globalPid'])
                    if pid not in pid_owners:continue
                    all_kernels[pid_owners[pid]].append((row['start'],row['end']))
                    key=(pid,row['streamId']);previous[(pid,row['correlationId'],row['start'],row['end'])]=stream_previous.get(key)
                    stream_previous[key]=max(stream_previous.get(key,0),row['end'])
            prefix=KernelPrefix(all_kernels)
            seen=set()
            for phase in parsed['phases']:
                rid=phase['request_id']
                if rid not in request_rows:raise ValueError('Measured foreign request phase')
                request=request_rows[rid];owner=request['model'];pid=raw['ready'][owner]['pid']
                for kernel in phase['kernels']:
                    begin,end=kernel['nsys_start_ns'],kernel['nsys_end_ns'];correlation=kernel['correlation_id']
                    key=(pid,correlation,begin,end)
                    if key in seen:raise ValueError('Duplicate canonical measured kernel')
                    seen.add(key)
                    observation=ready[rid,begin,end]
                    semantic=observation['phase']
                    api=apis[pid,correlation]
                    value=evidence(dict(start=begin,end=end),api,index,prefix,owner,event_prefix)
                    descriptor={field:kernel.get(field) for field in ('name','device','stream','context_id','green_context_id',
                        'registers_per_thread','static_shared_bytes','dynamic_shared_bytes','block','grid','cluster')}
                    signature=digest(descriptor);vocabulary[signature]=descriptor
                    value['prefix_features'].update(request_id=rid,task=request['task'],pool_id=request['pool_id'],
                        semantic_phase=semantic,launch_signature=signature,
                        measured_recorded_dependency_count=observation['recorded_dependency_count'],
                        measured_dependencies_known=observation['recorded_dependencies_known'])
                    value.update(schema=SCHEMA,capture=capture,source_node_id=observation['node_id'],
                        source_kernel_correlation_id=correlation,
                        dependency_observation=dict(certified_ready_bound_ns=observation['certified_ready_bound_ns'],
                            recorded_dependency_count=observation['recorded_dependency_count'],
                            recorded_dependencies_known=observation['recorded_dependencies_known']),
                        previous_same_stream_kernel_observed_end_ns=previous[key])
                    # Dependency counts/knownness were reconstructed from a
                    # complete DAG and are labels, not certified prefix features.
                    del value['prefix_features']['measured_recorded_dependency_count']
                    del value['prefix_features']['measured_dependencies_known']
                    labels=value['observation_labels'];part=labels['observed_lifetime_context_partition']
                    start_state,end_state=labels['context_at_kernel_start'],labels['context_at_kernel_end']
                    start_other=(start_state['status']=='QUALIFIED_PAIRED_CONTEXT' and not start_state['boundary']
                        and start_state['owner']!=owner)
                    end_other=(end_state['status']=='QUALIFIED_PAIRED_CONTEXT' and not end_state['boundary']
                        and end_state['owner']!=owner)
                    no_own=part['own_envelope_ns']==0 and part['gap_ns']==0 and part['other_envelope_ns']>0
                    internal_issue=(bisect_right(index.issue_times,end)>bisect_left(index.issue_times,begin))
                    wholly_other=no_own and start_other and end_other and not internal_issue
                    values=dict(kernels=1,observed_kernel_lifetime_sum_ns=end-begin,
                        kernel_START_inside_other_paired_context=int(start_other),
                        kernel_END_inside_other_paired_context=int(end_other),
                        kernel_START_and_END_other_no_own_envelope=int(wholly_other),
                        kernel_lifetime_wholly_other_no_own_envelope_ns=(end-begin if wholly_other else 0),
                        own_envelope_lifetime_overlap_sum_ns=part['own_envelope_ns'],
                        other_envelope_lifetime_overlap_sum_ns=part['other_envelope_ns'],
                        gap_lifetime_overlap_sum_ns=part['gap_ns'],
                        kernel_lifetime_overlaps_other_owner_kernels=int(bool(labels['other_owner_kernel_lifetime_union_overlap_ns'])),
                        API_return_after_kernel_START=int(api['end']>begin),
                        API_return_after_kernel_END=int(api['end']>end))
                    counts.update(values);strata[capture,owner,request['task'],request['pool_id'],semantic].update(values)
                    for endpoint,state,other in [('START',start_state,start_other),('END',end_state,end_other)]:
                        if other:pairs[owner,state['owner']][endpoint]+=1
                    if wholly_other and len(examples)<10:
                        examples.append(dict(capture=capture,request_id=rid,owner=owner,semantic_phase=semantic,
                            kernel=descriptor,prefix_features=value['prefix_features'],labels=labels))
                    stream.write(json.dumps(value,separators=(',',':'))+'\n')
            if len(seen)!=len(ready):raise ValueError('Every measured ready observation must retain its kernel row')
            captures.append(dict(capture=capture,counts=dict(counts),paired_contexts=len(index.rows),
                unpaired_context_issues=len(issues),measured_requests=len(request_rows)))
            all_counts.update(counts)
    after=[receipt(ref['path']) for ref in before]
    if before!=after:raise ValueError('Source bytes changed during training evidence extraction')
    vocab_path=output/'launch-signature-vocabulary.json'
    vocab_path.write_text(json.dumps(vocabulary,indent=2)+'\n')
    result=dict(schema=SCHEMA,status='TRAINING_ENDPOINT_QUEUE_EVIDENCE_ONLY',
        counts=dict(all_counts),captures=captures,
        by_owner_task_input_phase=[dict(capture=c,owner=o,task=t,pool_id=p,phase=s,counts=dict(v))
            for (c,o,t,p,s),v in sorted(strata.items())],
        nonheld_endpoint_owner_pairs=[dict(kernel_owner=a,context_holder=b,counts=dict(v)) for (a,b),v in sorted(pairs.items())],
        examples=examples,source_receipts_before=before,source_receipts_after=after,
        kernel_rows=receipt(row_path),launch_vocabulary=receipt(vocab_path),
        measurement_targets=['launch API return to observed kernel START (signed, not all context wait)',
            'Already-started status at API return; observed kernel END; interval-censored completion at later cutoffs',
            'Owned endpoint and observed lifetime relation to qualified paired context windows',
            'Observed cross-owner kernel lifetime overlap, without simultaneous-SM inference'],
        unavailable_targets=['Exclusive SM execution/service time during a CUPTI lifetime',
            'Per-kernel throughput from register/shared-memory/occupancy metadata',
            'Causal speedup of weight/cache reuse or complementary memory/compute pairing without separate profiles',
            'Physical CUDA dependency readiness from uncertified incomplete DAG records'],
        interpretation='Endpoint contradictions falsify a deterministic exclusive context-envelope gate under the audited attribution; lifetimes/overlap do not identify continuous execution or SM service.',
        training_only=True,GPU_started=False,validation_labels_used=False,predictor_changed=False,
        physical_service_fit=False,independent_accuracy_validated=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.supervision,args.output)
    print(json.dumps(dict(counts=result['counts'],captures=result['captures']),indent=2))
