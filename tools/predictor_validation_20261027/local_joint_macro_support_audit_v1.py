"""Read-only training support diagnostics for local joint response/notice atoms."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
import io
import json
import math
from pathlib import Path
import statistics
import unittest

from . import local_joint_macro_candidate_v1 as joint

ROOT=Path(__file__).resolve().parents[2]


def require(value,message):
    if not value:raise ValueError(message)


def summary(values):
    values=sorted(values)
    if not values:return dict(n=0)
    def q(p):return values[min(len(values)-1,int(p*(len(values)-1)))]
    return dict(n=len(values),min=values[0],median=statistics.median(values),
        p95_order_statistic=q(.95),max=values[-1])


def run(paths,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    law=joint.fit_training_atoms(paths)
    require(law['capture_count']==3,'Exactly three whole natural training captures required')
    refs=list(law['sources']);atoms=[a for c in law['cells'] for a in c['atoms']]
    provenance=law['provenance_not_used_for_generation']
    require(len(atoms)==len(provenance)==690,'Complete 690 language forward labels required')
    capture_by_atom=defaultdict(list)
    for p in provenance:capture_by_atom[p['atom_id']].append(p['capture_sha256'])
    require(all(len(x)==1 for x in capture_by_atom.values()),'Atom digest collision requires explicit provenance handling')
    folds=[]
    for heldout in law['training_captures']:
        records=[]
        for cell in law['cells']:
            train=[a for a in cell['atoms'] if capture_by_atom[a['atom_id']][0]!=heldout]
            test=[a for a in cell['atoms'] if capture_by_atom[a['atom_id']][0]==heldout]
            for atom in test:
                records.append(dict(features=cell['features'],atom_id=atom['atom_id'],
                    local_training_atoms=len(train),local_support=bool(train),
                    exact_response_training_atoms=sum(t['response_ns']==atom['response_ns'] for t in train),
                    prior_response_min_ns=min((t['response_ns'] for t in train),default=None),
                    prior_response_max_ns=max((t['response_ns'] for t in train),default=None),
                    heldout_response_ns=atom['response_ns'],
                    response_in_training_range=bool(train) and min(t['response_ns'] for t in train)<=atom['response_ns']<=max(t['response_ns'] for t in train)))
        require(len(records)==230,'All heldout forward labels retained, including unknowns')
        folds.append(dict(heldout_training_capture_sha256=heldout,heldout_forward_rows=len(records),
            known_local_cell_rows=sum(r['local_support'] for r in records),
            unknown_local_cell_rows=sum(not r['local_support'] for r in records),
            known_exact_response_rows=sum(r['exact_response_training_atoms']>0 for r in records),
            unknown_exact_response_rows=sum(r['exact_response_training_atoms']==0 for r in records),
            response_in_training_range_rows=sum(r['response_in_training_range'] for r in records),
            local_cell_training_atoms_histogram=dict(Counter(r['local_training_atoms'] for r in records)),
            conditional_joint_likelihood='UNDEFINED_CONTINUOUS_HELDOUT_ZERO_EXACT_EMPIRICAL_SUPPORT',records=records))
    lag=defaultdict(list);offsets=defaultdict(list);query_counts=Counter();late=Counter();finals=0
    for atom in atoms:
        world=joint.realize(atom,0)
        if atom['following_ready_tail_ns']>0:finals+=1
        seen_query=[]
        for m,n in zip(atom['marks'],world['notifications']):
            kind=m['payload']['semantic_kind']
            for clock,field in [('occurrence_to_reader','reader_delay_ns'),('reader_to_dequeue','dequeue_extra_ns'),
                    ('dequeue_to_processing','processing_extra_ns')]:lag[(kind,clock)].append(m[field])
            offsets[kind].append(m['occurrence_offset_ns']/atom['response_ns'])
            if n['processed_ns']>atom['response_ns']:late[kind]+=1
            if kind=='recorded_stream_query':
                seen_query.append(m['payload']['end_event_complete'])
                query_counts[(m['payload']['policy_site'],m['payload']['end_event_complete'])]+=1
                require(m['payload']['source_macro_id']!='native_end_to_worker_ready','Query must retain recorded source forward')
        require(seen_query==[False,True],'Observed two-query policy path changed; do not extrapolate absent branches')
    require(finals==138,'All final forward+complete READY tails retained')
    require(sum(query_counts.values())==1380,'All 1380 queries retained, including 138 occurring after native END')
    per_capture=[];frozen_protocol=[]
    for path in paths:
        doc=joint.checked(joint.receipt(path));review=joint.checked(doc['sources'][0]);raw=joint.checked(review['inputs']['raw.json'])
        ctx=joint.checked(review['inputs']['context.json']);rows=raw['request_records']
        streams=defaultdict(list);pid_by_owner=defaultdict(set)
        for row in rows:
            start=next(e for e in row['execution_events'] if e.get('event') is None)
            pid_by_owner[row['model']].add(start['pid'])
            for ordinal,event in enumerate(row['execution_events']):
                streams[row['model']].append((event['perf_ns'],event['received_time_ns'],event['collector_dequeued_time_ns'],
                    event['collector_processing_time_ns'],'EXECUTION',row['request_id'],ordinal))
            streams[row['model']].append((row['worker_response']['end_perf_ns'],row['response_reader_event_enqueue_time_ns'],
                row['response_collector_dequeued_time_ns'],row['response_collector_processing_time_ns'],'RESULT',row['request_id'],-1))
            ready=row['previous_handler_next_request_ready']
            streams[row['model']].append((ready['perf_ns'],ready['received_time_ns'],ready['collector_dequeued_time_ns'],
                ready['collector_processing_time_ns'],'WORKER_READY',row['request_id'],-1))
        inversions=Counter();adjacent=0
        for owner,events in streams.items():
            require(len(pid_by_owner[owner])==1,'This diagnostic requires one measured process per owner per capture')
            events.sort(key=lambda e:e[0]);adjacent+=len(events)-1
            for a,b in zip(events,events[1:]):
                for i,key in enumerate(('occurrence','reader','dequeue','processing')):inversions[key]+=a[i]>b[i]
        require(not any(inversions.values()),'Observed producer order inversion requires diagnosis')
        protocol=[]
        exact_paths={str(ROOT/p) for p in ('gsdeploy/runtime/deployment.py','gsdeploy/runtime/transport.py',
            'tools/predictor_validation_20261027/semantic_progress_worker_v1.py',
            'tools/predictor_validation_20261027/device_progress_observer_v1.py')}
        for source in ctx['execution_sources']:
            if source['original_path'] in exact_paths:
                require(joint.receipt(source['frozen']['path'])==source['frozen'],'Frozen protocol changed')
                protocol.append(source);frozen_protocol.append(source['frozen'])
        require(len(protocol)==4,'Four exact historical producer/observer/transport/collector sources required')
        per_capture.append(dict(raw=review['inputs']['raw.json'],requests=len(rows),jobs=len({r['profile_job_id'] for r in rows}),
            workers=len(streams),single_observed_pid_per_owner=True,adjacent_message_pairs=adjacent,
            monotone_clock_inversions=dict(inversions),all_message_sequence_recorded=False,
            sampled_closed_capture_loss_probability_identified=False,historical_protocol_sources=protocol))
    sources=[*refs,*frozen_protocol,joint.receipt(__file__),joint.receipt(ROOT/'tests/test_local_joint_macro_candidate_v1.py')]
    modules=['tests.test_local_joint_macro_candidate_v1','tests.test_response_notification_posterior_reference_v1',
        'tests.test_joint_macro_notification_contract_independent_v1']
    log=io.StringIO();tested=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    (output/'tests.txt').write_text(log.getvalue());require(tested.wasSuccessful(),'Counterexample tests failed')
    for ref in sources:require(joint.receipt(ref['path'])==ref,'Changed source during support audit')
    (output/'training-joint-atoms.json').write_text(json.dumps(law,indent=2,allow_nan=False)+'\n')
    (output/'whole-capture-support.json').write_text(json.dumps(folds,indent=2,allow_nan=False)+'\n')
    report=dict(schema='gsdeploy.local_joint_macro_support_diagnostics.v1',
        status='THREE_TRAINING_CAPTURE_EMPIRICAL_JOINT_CANDIDATE_NOT_PROSPECTIVE_PROVIDER',sources=sources,
        training_law=joint.receipt(output/'training-joint-atoms.json'),whole_capture_support=joint.receipt(output/'whole-capture-support.json'),
        captures=per_capture,training_capture_blocks=3,requests=150,forward_atoms=690,
        exact_local_cells=len(law['cells']),cell_atom_count_histogram=dict(Counter(len(c['atoms']) for c in law['cells'])),
        final_forward_plus_READY_tail_joint_atoms=finals,
        query_policy_counts=[dict(policy_site=k[0],end_event_complete=k[1],count=v) for k,v in query_counts.items()],
        notification_after_source_forward_macro_END=dict(late),
        lag_ns=[dict(semantic_kind=k[0],clock_leg=k[1],**summary(v)) for k,v in sorted(lag.items())],
        occurrence_position_relative_to_forward_response=[dict(semantic_kind=k,**summary(v)) for k,v in sorted(offsets.items())],
        whole_capture_leaveout=[{k:v for k,v in f.items() if k!='records'} for f in folds],
        tests=dict(run=tested.testsRun,failures=len(tested.failures),errors=len(tested.errors),log=joint.receipt(output/'tests.txt')),
        identified=['Observed paired host-response/internal milestone/three-leg notification delays in exact natural local input/shape/peer-at-BEGIN cells.',
            'Historical per-worker stdout, one reader and collector protocol supplies conditional FIFO within authenticated worker process lifetime.',
            'Query schedule follows after-forward-END and next-prehook/context-exit observer policy; no independent query arrival process.',
            'Complete final forward and native-END-to-READY wall, tail host events/query, RESULT and WORKER_READY remain one joint atom.'],
        unknown=['Continuous conditional joint density/bandwidth and support outside empirical atoms.',
            'Dynamic future peer response to changing covariate paths; current atom conditions on peer state at actual BEGIN only.',
            'Shared parent-queue/capture effects and cross-macro/request residual transition law; independent draws are an unvalidated candidate assumption.',
            'Notification loss and full registry completeness under truncation/restart; no persistent all-message producer sequence.',
            'Prospective generation/stop/shape branch and G2 full-N TTS work; current terminal-kind and decode shape are training conditioning labels.',
            'Independent posterior calibration, native assimilation, SM service, all-stream completion, P95/DeltaU ranking and main provider qualification.'],
        next_minimum_evidence=['Keep original exact public content, known arrival schedule, execution receipt and whole-capture identity; add unseen content/length only with independent template qualification.',
            'Capture stable worker/transport lifetime ID plus monotonically increasing all-message producer sequence, emit bracket and reader enqueue sequence; include RESULT/READY/errors/EOF and all unclosed stages.',
            'Record each observer poll site and source event ID, host query bracket, stream/device and current physical macro; retain query policy version and context-exit/post-sync branch.',
            'Use independent full training captures across parent CPU/reader load; retain shared queue state and all pending cross-macro/cross-request notifications for a validated transition model.',
            'Fit a fixed continuous conditional joint family on training captures, with whole-capture leaveout including unknown and unclosed denominators; only then freeze a prospective provider for validation.'],
        GPU_started=False,validation_labels_read=False,main_queue_provider_eligible=False,
        fitted_empirical_measure=True,conditional_probability_calibrated=False,
        native_assimilation_implemented=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return joint.receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.training_macros,args.output)))
