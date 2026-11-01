"""Post-freeze descriptive error decomposition, never fitting or selection."""
import argparse
from collections import defaultdict,Counter
import json
import math
from pathlib import Path
import random
import statistics

from . import joint_request_response_transport_candidate_v2 as work
from . import joint_request_protocol_transport_candidate_v3 as protocol
from . import local_joint_macro_candidate_v1 as local
from .macro_protocol_timing_reference_v3 import simulate

ROOT=Path(__file__).resolve().parents[2]


def f(x):return x['numerator']/x['denominator']


def stats(values):
    return dict(n=len(values),mean_ms=statistics.mean(values)/1e6,
        MAE_ms=statistics.mean(abs(x) for x in values)/1e6,
        RMSE_ms=math.sqrt(statistics.mean(x*x for x in values))/1e6)


def run(paths,fit_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fixed_ref=local.receipt(ROOT/'artifacts/predictor-joint-request-protocol-transport-v3-natural-20261028-training05/report.json')
    fixed=local.checked(fixed_ref)
    for ref in fixed['sources']:work.require(local.receipt(ref['path'])==ref,'Frozen training05 source changed')
    docs,fit,sources=work.verified_inputs(paths,fit_path)
    sources.extend([fixed_ref,local.receipt(__file__),local.receipt(Path(protocol.__file__)),
        local.receipt(Path(simulate.__code__.co_filename))])
    request_records=[];macro_records=[];query_records=[];station_records=[];initialization=[]
    for fold in fit['whole_capture_leaveout']:
        heldout=fold['heldout_capture_sha256'];test=next(d for d in docs if d['requests'][0]['macros'][0]['capture_sha256']==heldout)
        train=[d for d in docs if d is not test];law=protocol.build_law(train,fold['training_fit'])
        review=local.checked(test['sources'][0]);raw=local.checked(review['inputs']['raw.json']);rows=raw['request_records']
        doc_rows={r['request_id']:r for r in test['requests']};jobs=defaultdict(list)
        for row in rows:jobs[row['profile_job_id']].append(row)
        donor_docs={d['requests'][0]['macros'][0]['capture_sha256']:d for d in train}
        donor_pairs=[]
        for p,v in zip(law['provenance_not_used_for_lookup'],law['vectors']):
            donor_request=next(r for r in donor_docs[p['capture_sha256']]['requests'] if r['request_id']==p['request_id'])
            donor_pairs.append((v,donor_request))
        for captureq in (.25,.75):
            for seed in (0,1):
                rng=random.Random(seed);draws={};ordinals=Counter()
                for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'],r['model'])):
                    key=(row['model'],row['task'],row['pool_id'])
                    withinq=rng.random()
                    draw=work.choose_vector(law,key,capture_quantile=captureq,within_capture_quantile=withinq)
                    vector=draw['vector'];plan,slacks=protocol.to_reference(vector,public_dispatch_ordinal=ordinals[key])
                    donors=[(v,r) for v,r in donor_pairs if v['training_capture_sha256']==draw['common_training_capture_latent']
                        and v['public_key']==list(key)]
                    donor=donors[min(len(donors)-1,int(withinq*len(donors)))]
                    work.require(donor[0]['vector_id']==vector['vector_id'],'Ordinal donor provenance mismatch')
                    draws[row['request_id']]=(plan,slacks,vector,donor[1]);ordinals[key]+=1
                for job_rows in jobs.values():
                    scenario=sorted(job_rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index']))
                    origin=min(r['scheduled_time_ns'] for r in scenario);public=[];plans=defaultdict(list);slackmap={}
                    initialization.append(dict(heldout_capture=heldout,original_job=scenario[0]['profile_job_id'],
                        captureq=captureq,seed=seed,source_offset_from_capture_observation_start_ns=origin-raw['observation_start_ns'],
                        station_initial_state='EXPLICIT_EMPTY_PER_JOB_HYPOTHESIS_NOT_MEASURED_SOURCE_STATE',
                        measured_prior_capture_messages=sum(e['received_time_ns']<origin for r in rows for e in r['execution_events']),
                        source_priming_and_control_messages_not_in_decomposition=True))
                    for row in scenario:
                        key=(row['model'],row['task'],row['pool_id']);plan,slacks,_,_=draws[row['request_id']]
                        plans[key].append(plan);slackmap[plan]=slacks
                        public.append(dict(request_id='public:'+str(len(public)),owner=key[0],task=key[1],pool_id=key[2],arrival_ns=row['scheduled_time_ns']-origin))
                    end=max(r['arrival_ns'] for r in public)+1
                    generated=simulate(public,{k:tuple(v) for k,v in plans.items()},work.rate_table(law['beta']),window_start_ns=0,
                        window_end_ns=end,drain_end_ns=end+60_000_000_000,producer_fifo=tuple(k[0]+':modeled_stdout_lifetime' for k in plans),protocol_slacks=slackmap)
                    for source,offered in zip(scenario,public):
                        rid=offered['request_id'];trace=[e for e in generated['trace'] if e.get('request_id')==rid]
                        occurrence={e['identity']:e for e in trace if e['kind']=='latent_occurrence'}
                        notified={e['identity']:e for e in trace if e['kind']=='protocol_reader_available'}
                        processed={e['identity']:e for e in trace if e['kind']=='processed_notification'}
                        started=f(generated['starts'][rid]);resultocc=f(occurrence['RESULT']['occurrence_ns'])
                        actualstart=source['observed_worker_start_time_ns'];actualresult=source['worker_response']['end_perf_ns']
                        component_start=(started-offered['arrival_ns'])-(actualstart-source['scheduled_time_ns'])
                        component_response=(resultocc-started)-(actualresult-actualstart)
                        component_notice=(f(generated['RESULT_receipts'][rid])-resultocc)-(source['completion_time_ns']-actualresult)
                        total=(f(generated['RESULT_receipts'][rid])-offered['arrival_ns'])-(source['completion_time_ns']-source['scheduled_time_ns'])
                        work.require(abs(total-component_start-component_response-component_notice)<1e-4,'Additive E2E error identity failed')
                        request_records.append(dict(owner=source['model'],input=source['pool_id'],capture=heldout,
                            total_reader_E2E_error_ns=total,startup_parent_dispatch_plus_handoff_error_ns=component_start,
                            START_to_RESULT_occurrence_error_ns=component_response,RESULT_reader_notification_error_ns=component_notice,
                            parent_and_handoff_separately_identified=False))
                        target=doc_rows[source['request_id']];donor=draws[source['request_id']][3];vector=draws[source['request_id']][2]
                        completed=[e for e in trace if e['kind']=='macro_complete'];prev=started
                        for event,node,targetmacro,donormacro in zip(completed,vector['nodes'],target['macros'],donor['macros']):
                            at=f(event['at_ns']);wall=at-prev;prev=at
                            reference=f(node['reference_work_ns']);targetwork=float(work.progress(targetmacro,targetmacro['right_ns'],law['beta']))
                            macro_records.append(dict(owner=source['model'],macro_id=node['macro_id'],phase=node['phase'],
                                actual_wall_ns=targetmacro['response_wall_ns'],donor_wall_ns=donormacro['response_wall_ns'],predicted_wall_ns=wall,
                                wall_error_ns=wall-targetmacro['response_wall_ns'],
                                donor_wall_residual_difference_ns=donormacro['response_wall_ns']-targetmacro['response_wall_ns'],
                                generated_peer_protocol_transport_change_ns=wall-donormacro['response_wall_ns'],
                                donor_minus_target_reference_work_ns=reference-targetwork,
                                reference_work_difference_is_diagnostic_not_causal=True))
                        # Same frozen exact-input template: compare source phase
                        # and false/true role, never arbitrary producer ordinal.
                        targetqueries={(e['phase_id'],e['end_event_complete']):e for e in source['execution_events'] if e.get('kind')=='device_event_query'}
                        formarks={m['identity']:m for n in vector['nodes'] for m in n['marks']}
                        for ident,mark in formarks.items():
                            if mark['action']!='query':continue
                            targetq=targetqueries[(mark['source_macro_id'],mark['query_complete'])]
                            occ=f(occurrence[ident]['occurrence_ns']);reader=f(notified[ident]['reader_ns']);proc=f(processed[ident]['available_ns'])
                            query_records.append(dict(owner=source['model'],source_phase=mark['source_macro_id'],occurrence_macro=mark['occurrence_macro_id'],
                                complete=mark['query_complete'],
                                occurrence_relative_START_error_ns=(occ-started)-(targetq['perf_ns']-actualstart),
                                reader_delay_error_ns=(reader-occ)-(targetq['received_time_ns']-targetq['perf_ns']),
                                processing_delay_error_ns=(proc-occ)-(targetq['collector_processing_time_ns']-targetq['perf_ns'])))
                        targetevents={'event:'+str(i):e for i,e in enumerate(source['execution_events'])}
                        targetevents['RESULT']=dict(perf_ns=actualresult,received_time_ns=source['completion_time_ns'],collector_processing_time_ns=source['response_collector_processing_time_ns'])
                        targetevents['worker_READY']=source['previous_handler_next_request_ready']
                        for ident,n in notified.items():
                            targetevent=targetevents[ident];occ=f(occurrence[ident]['occurrence_ns'])
                            station_records.append(dict(owner=source['model'],action=n['action'],
                                reader_delay_error_ns=f(n['reader_ns'])-occ-(targetevent['received_time_ns']-targetevent['perf_ns']),
                                processing_delay_error_ns=f(n['processing_ns'])-occ-(targetevent['collector_processing_time_ns']-targetevent['perf_ns'])))
    def grouped(records,keys,fields):
        groups=defaultdict(list)
        for row in records:groups[tuple(row[k] for k in keys)].append(row)
        return [dict(**dict(zip(keys,key)),**{field:stats([r[field] for r in rows]) for field in fields}) for key,rows in sorted(groups.items(),key=lambda x:str(x[0]))]
    output_tables=dict(request_by_owner=grouped(request_records,['owner'],['total_reader_E2E_error_ns','startup_parent_dispatch_plus_handoff_error_ns',
        'START_to_RESULT_occurrence_error_ns','RESULT_reader_notification_error_ns']),
        macro_by_owner_phase=grouped(macro_records,['owner','macro_id','phase'],['wall_error_ns','donor_wall_residual_difference_ns',
            'generated_peer_protocol_transport_change_ns','donor_minus_target_reference_work_ns']),
        query_by_owner_phase_outcome=grouped(query_records,['owner','source_phase','complete'],['occurrence_relative_START_error_ns','reader_delay_error_ns','processing_delay_error_ns']),
        notifications_by_owner_action=grouped(station_records,['owner','action'],['reader_delay_error_ns','processing_delay_error_ns']))
    (output/'decomposition-tables.json').write_text(json.dumps(output_tables,indent=2)+'\n')
    (output/'raw-decomposition-records.json').write_text(json.dumps(dict(requests=request_records,macros=macro_records,queries=query_records,notifications=station_records,initialization=initialization),indent=2)+'\n')
    for ref in sources:work.require(local.receipt(ref['path'])==ref,'Frozen source changed during decomposition')
    report=dict(schema='gsdeploy.protocol_transport_error_decomposition.v3',status='DESCRIPTIVE_WHOLE_TRAINING_CAPTURE_DECOMPOSITION_ONLY',
        sources=sources,requests=len(request_records),macro_records=len(macro_records),query_records=len(query_records),
        tables=local.receipt(output/'decomposition-tables.json'),records=local.receipt(output/'raw-decomposition-records.json'),
        exact_additive_E2E_error_identity_verified=True,
        attribution_is_causal=False,parameters_or_bandwidth_selected=False,
        initial_station_state='empty per original offered job is an explicit hypothesis; unknown original priming/control/background state',
        source_exact_reconstruction='entire measured capture with original clocks and recurrence; does not authenticate per-job forward initialization',
        donor_work='computed from donor original processed peer path; moves jointly with all marks into generated peer path',
        parent_policy='after READY processing alternative; parent and handoff attribution not separately identified',
        GPU_started=False,validation_labels_read=False,main_queue_provider_eligible=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return local.receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--response-fit',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.training_macros,args.response_fit,args.output)))
