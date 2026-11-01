"""Same-donor v3/v4 CPU training diagnostics with explicit idle-pair endpoints."""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from . import actual_worker_twoFIFO_reference_v4 as actual
from .actual_worker_twoFIFO_query_adapter_v4_1 import query_bound_view
from . import joint_request_protocol_transport_candidate_v3 as protocol
from . import joint_request_response_transport_candidate_v2 as model
from . import local_joint_macro_candidate_v1 as local
from .macro_protocol_timing_reference_v3 import simulate as alternative
from .worker_stdin_fifo_contract_v1 import TransportDraw
from .implementation_source_closure_v27 import implementation_sources,strict_references


def numeric(value):return value['numerator']/value['denominator']


def ensemble(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['heldout_training_capture_sha256'],row['evaluation_only_heldout_request_id']].append(row)
    complete=[]
    for group in groups.values():
        if len(group)!=4:raise ValueError('Exactly four same-seed worlds per offered request required')
        if any(r['status']!='SIMULATED' or r['predicted_latency_ns'] is None for r in group):continue
        xs=[r['predicted_latency_ns'] for r in group];y=group[0]['actual_latency_ns']
        complete.append(dict(mean_error=statistics.mean(xs)-y,median_error=statistics.median(xs)-y,
            mean_world_error=statistics.mean(abs(x-y) for x in xs),
            empirical_CRPS=statistics.mean(abs(x-y) for x in xs)-sum(abs(a-b) for a in xs for b in xs)/(2*len(xs)**2)))
    result=dict(offered_world_requests=len(rows),offered_unique_requests=len(groups),complete_four_world_ensembles=len(complete),
        incomplete_or_unknown_ensembles=len(groups)-len(complete))
    if complete:result.update(
        mean_random_world_MAE_ms=statistics.mean(r['mean_world_error'] for r in complete)/1e6,
        four_world_mean_MAE_ms=statistics.mean(abs(r['mean_error']) for r in complete)/1e6,
        four_world_mean_RMSE_ms=math.sqrt(statistics.mean(r['mean_error']**2 for r in complete))/1e6,
        four_world_median_MAE_ms=statistics.mean(abs(r['median_error']) for r in complete)/1e6,
        four_atom_empirical_CRPS_ms=statistics.mean(r['empirical_CRPS'] for r in complete)/1e6)
    return result


def run(paths,fit_path,fifo_path,prior_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    docs,fit,sources=model.verified_inputs(paths,fit_path)
    prior_ref=local.receipt(prior_path);prior=local.checked(prior_ref);frozen_folds=local.checked(prior['diagnostics'])
    fifo_ref=local.receipt(fifo_path);fifo=local.checked(fifo_ref)
    sources.extend([prior_ref,prior['diagnostics'],fifo_ref,*prior['sources'],*fifo['sources']])
    model.require(all(local.receipt(r['path'])==r for r in sources),'Frozen training/FIFO source changed')
    donor_rows={(c['capture_sha256'],r['request_id']):r for c in fifo['captures'] for r in c['records']}
    records=[];scenarios=[];examples={};fold_summaries=[]
    for fold in fit['whole_capture_leaveout']:
        heldout=fold['heldout_capture_sha256'];test=next(d for d in docs if d['requests'][0]['macros'][0]['capture_sha256']==heldout)
        train=[d for d in docs if d is not test];law=protocol.build_law(train,fold['training_fit'])
        model.require(heldout not in law['captures'],'Heldout donor leak')
        paired=list(zip(law['vectors'],law['provenance_not_used_for_lookup']))
        review=local.checked(test['sources'][0]);raw=local.checked(review['inputs']['raw.json']);rows=raw['request_records']
        jobs=defaultdict(list)
        for row in rows:jobs[row['profile_job_id']].append(row)
        frozen=next(f for f in frozen_folds if f['heldout_training_capture_sha256']==heldout)['records'];cursor=0
        fold_records=[]
        for captureq in (.25,.75):
            for seed in (0,1):
                rng=random.Random(seed);draws={};ordinal=Counter()
                for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'],r['model'])):
                    key=row['model'],row['task'],row['pool_id'];withinq=rng.random()
                    chosen=model.choose_vector(law,key,capture_quantile=captureq,within_capture_quantile=withinq)
                    model.require(chosen['status']=='TRAINING_JOINT_REQUEST_VECTOR_DRAW_ONLY','Unsupported public key remains UNKNOWN')
                    supported=[(v,p) for v,p in paired if v['training_capture_sha256']==chosen['common_training_capture_latent'] and v['public_key']==list(key)]
                    vector,provenance=supported[min(len(supported)-1,int(withinq*len(supported)))]
                    model.require(vector==chosen['vector'],'Ordinal donor provenance mismatch')
                    donor=donor_rows[provenance['capture_sha256'],provenance['request_id']]
                    model.require(donor['observed_submit_to_START_ns']==vector['dispatch_to_worker_start_ns'],
                        'Donor transport pair and response vector must be jointly sourced')
                    plan,slacks=protocol.to_reference(vector,public_dispatch_ordinal=ordinal[key]);ordinal[key]+=1
                    draws[row['request_id']]=(plan,slacks,donor)
                for job_rows in jobs.values():
                    job=sorted(job_rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index']))
                    origin=min(r['scheduled_time_ns'] for r in job);public=[];oldplans=defaultdict(list);newplans=defaultdict(list)
                    oldslacks={};newslacks={};draw_by_public={};keyord=Counter()
                    for row in job:
                        key=row['model'],row['task'],row['pool_id'];plan,slacks,donor=draws[row['request_id']]
                        newplan=replace(plan,dispatch_to_worker_start_ns=0)
                        rid='public:'+str(len(public));public.append(dict(request_id=rid,owner=key[0],task=key[1],pool_id=key[2],arrival_ns=row['scheduled_time_ns']-origin))
                        oldplans[key].append(plan);newplans[key].append(newplan);oldslacks[plan]=slacks;newslacks[newplan]=slacks
                        draw_by_public[(*key,keyord[key])]=donor;keyord[key]+=1
                    end=max(row['arrival_ns'] for row in public)+1;drain=end+60_000_000_000
                    common=dict(window_start_ns=0,window_end_ns=end,drain_end_ns=drain,
                        producer_fifo=tuple(sorted({p.ready_producer_id for ps in oldplans.values() for p in ps})))
                    rates=model.rate_table(law['beta'])
                    old=alternative(public,{k:tuple(v) for k,v in oldplans.items()},rates,protocol_slacks=oldslacks,**common)
                    owners={row['owner'] for row in public};owner_counts=Counter(row['owner'] for row in public)
                    for endpoint in ('line_min','line_max'):
                        transports={}
                        for key,donor in draw_by_public.items():
                            d=donor['observed_submit_to_START_ns'];w=donor['submit_to_write_record_ns']
                            x=0 if endpoint=='line_min' else min(d,w)
                            # Zero enqueue/sender extras deliberately preserve v3's
                            # immediate dispatch baseline; no busy sender law fitted.
                            transports[key]=TransportDraw(0,0,w,x,d-x)
                        error=None;generated=None
                        try:
                            generated=actual.simulate(public,{k:tuple(v) for k,v in newplans.items()},rates,
                                protocol_slacks=newslacks,transport_draws=transports,
                                initial_sender_available_ns={owner:0 for owner in owners},
                                initial_worker_read_release_ns={owner:0 for owner in owners},
                                initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=()),**common)
                            generated=query_bound_view(generated,public,window_start_ns=0,window_end_ns=end,drain_end_ns=drain)
                        except UnknownEvidence as exc:error=str(exc)
                        same_START=generated is not None and generated['starts']==old['starts']
                        same_RESULT=generated is not None and generated['RESULT_receipts']==old['RESULT_receipts']
                        scenarios.append(dict(heldout_training_capture_sha256=heldout,evaluation_only_job_id=job[0]['profile_job_id'],
                            capture_quantile=captureq,seed=seed,endpoint=endpoint,offered=len(public),maximum_same_owner_requests=max(owner_counts.values()),
                            same_START_as_v3=same_START,same_RESULT_reader_as_v3=same_RESULT,unknown_reason=error,
                            query_fingerprint_sha256=None if generated is None else generated['query_fingerprint_sha256']))
                        if generated is not None and endpoint not in examples:examples[endpoint]=generated
                        for i,(source,request) in enumerate(zip(job,public)):
                            prior_record=frozen[cursor+i];rid=request['request_id'];oldreceipt=old['RESULT_receipts'].get(rid)
                            oldlatency=None if oldreceipt is None else numeric(oldreceipt)-request['arrival_ns']
                            expected=(source['model'],source['pool_id'],request['arrival_ns'],source['completion_time_ns']-source['scheduled_time_ns'])
                            model.require((prior_record['public_owner'],prior_record['public_input_id'],prior_record['offered_arrival_offset_ns'],
                                prior_record['actual_RESULT_reader_latency_ns'])==expected and prior_record['predicted_RESULT_reader_latency_ns']==oldlatency,
                                'Recomputed same-donor v3 policy differs from frozen training05')
                            received=None if generated is None else generated['RESULT_receipts'].get(rid)
                            record=dict(heldout_training_capture_sha256=heldout,evaluation_only_heldout_request_id=source['request_id'],
                                evaluation_only_job_id=source['profile_job_id'],owner=source['model'],public_input_id=source['pool_id'],
                                arrival_ns=request['arrival_ns'],capture_quantile=captureq,seed=seed,endpoint=endpoint,
                                actual_latency_ns=expected[-1],v3_same_donor_latency_ns=oldlatency,status='UNKNOWN' if error else 'SIMULATED',
                                unknown_reason=error,predicted_latency_ns=None if received is None else numeric(received)-request['arrival_ns'],
                                actual_FIFO_and_v3_START_identical=same_START,actual_FIFO_and_v3_RESULT_reader_identical=same_RESULT)
                            records.append(record);fold_records.append(record)
                    cursor+=len(public)
        model.require(cursor==len(frozen)==4*len(rows),'All original offered requests and fixed four scenarios required')
        fold_summaries.append(dict(heldout_training_capture_sha256=heldout,training_donor_captures=law['captures'],
            original_requests=len(rows),original_jobs=len(jobs),endpoint_summaries={endpoint:ensemble([r for r in fold_records if r['endpoint']==endpoint])
                for endpoint in ('line_min','line_max')}))
        print(json.dumps(dict(fold=heldout,offered=len(fold_records),worlds=len(scenarios))),flush=True)
    sources=strict_references([sources,implementation_sources(__file__)])
    model.require(all(local.receipt(r['path'])==r for r in sources),'Source changed during v4 diagnostics')
    files={}
    for name,data in [('records',records),('scenarios',scenarios),('example-worlds',examples)]:
        path=output/(name+'.json');path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');files[name]=local.receipt(path)
    report=dict(schema='gsdeploy.actual_worker_twoFIFO_training_diagnostics.v4',
        status='SAME_DONOR_ORIGINAL_FIFO_TRAINING_DIAGNOSTICS_WITH_EXPLICIT_IDLE_PAIR_TRANSFER',sources=sources,files=files,
        folds=fold_summaries,endpoint_summaries={endpoint:ensemble([r for r in records if r['endpoint']==endpoint]) for endpoint in ('line_min','line_max')},
        worlds=len(scenarios),offered_world_requests=len(records),all_offered_retained=True,
        maximum_same_owner_requests_per_job=max(r['maximum_same_owner_requests'] for r in scenarios),
        all_same_donor_START_equal_v3=all(r['same_START_as_v3'] for r in scenarios),
        all_same_donor_RESULT_reader_equal_v3=all(r['same_RESULT_reader_as_v3'] for r in scenarios),
        zero_policy_contrast_reason='Each offered job contains at most one request per owner, so READY release cannot gate a second request in these empty-initial worlds.',
        policy_error_identified_in_busy_queues=False,busy_transport_transfer_qualified=False,
        explicit_assumptions=['Zero enqueue and sender extras preserve v3 immediate dispatch baseline; not measured zero CPU demand.',
            'Idle donor d,w jointly reused with x endpoints and pre=d-x; no fitted busy pipe law.',
            'Each original diagnostic job starts with explicitly free workers/senders and empty notification stations; historical state unknown.',
            'Response work/marks and protocol slacks transfer jointly from training captures into generated peer paths.'],
        training_captures_are_replication_units=3,job_scenarios_are_independent_capture_replicates=False,
        unchanged_v3_frozen_report_reproduced=True,parameter_or_model_selection_performed=False,
        original_scorer_modified=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False,final_gain_eligible=False)
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return local.receipt(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--response-fit',required=True);parser.add_argument('--fifo-audit',required=True)
    parser.add_argument('--prior-v3-report',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.training_macros,args.response_fit,args.fifo_audit,args.prior_v3_report,args.output)))
