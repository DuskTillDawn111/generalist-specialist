"""Fixed four-world generated-path comparison, whole training captures only."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from . import joint_request_response_transport_candidate_v2 as work
from . import joint_request_protocol_transport_candidate_v3 as protocol
from . import phase_heterogeneous_protocol_candidate_v1 as candidate
from . import phase_heterogeneous_protocol_reference_v1 as reference
from . import protocol_transport_ensemble_diagnostics_v3 as scoring
from . import macro_joint_response_fit_v1 as base
from .implementation_source_closure_v27 import implementation_sources,strict_references


def score_worlds(worlds):
    """Incomplete worlds retain their whole four-sample request denominator."""
    grouped=defaultdict(list)
    for row in worlds:grouped[row['heldout_capture'],row['evaluation_request_id']].append(row)
    known=[];unknown=[]
    for (capture,rid),samples in grouped.items():
        base.require(len(samples)==4,'Exactly four fixed draws per offered request required')
        if any(s['status']!='SIMULATED' or s.get('predicted_ns') is None for s in samples):
            unknown.append(dict(capture=capture,evaluation_request_id=rid,samples=samples));continue
        y=samples[0]['actual_ns'];xs=[s['predicted_ns'] for s in samples]
        base.require(all(s['actual_ns']==y for s in samples),'Evaluation alignment changed')
        absdiff=statistics.mean(abs(x-y) for x in xs)
        known.append(dict(heldout_training_capture_sha256=capture,
            heldout_request_id_for_evaluation_only=rid,owner=samples[0]['owner'],
            actual_ns=y,samples_ns=xs,mean_world_absolute_error_ns=absdiff,
            ensemble_mean_error_ns=statistics.mean(xs)-y,ensemble_median_error_ns=statistics.median(xs)-y,
            empirical_CRPS_ns=absdiff-sum(abs(a-b) for a in xs for b in xs)/(2*len(xs)**2),
            four_sample_min_ns=min(xs),four_sample_max_ns=max(xs),within_four_sample_minmax=min(xs)<=y<=max(xs)))
    by_owner=defaultdict(list);by_capture=defaultdict(list)
    for r in known:by_owner[r['owner']].append(r);by_capture[r['heldout_training_capture_sha256']].append(r)
    return dict(offered_requests=len(grouped),complete_ensembles=len(known),unknown_ensembles=len(unknown),
        complete_subset_metrics=scoring.summarize(known) if known else None,
        by_owner={k:scoring.summarize(v) for k,v in by_owner.items()},
        by_capture={k:scoring.summarize(v) for k,v in by_capture.items()},
        whole_offered_population_metrics_qualified=not unknown,
        records=known,unknown=unknown)


def simulate_candidate(docs,fitted,original):
    worlds=[];conflicts=Counter();nodes=[]
    for fold in fitted['whole_capture_leaveout']:
        heldout=fold['heldout_capture_sha256'];train=[];test=None
        for doc in docs:
            if doc['requests'][0]['macros'][0]['capture_sha256']==heldout:test=doc
            else:train.append(doc)
        original_fold=next(f for f in original['whole_capture_leaveout'] if f['heldout_capture_sha256']==heldout)
        fit=dict(fold['training_fit'],candidate=fitted['candidate'])
        law=candidate.build_law(train,fit,original_phase_fit=original_fold['training_fit'])
        base.require(heldout not in law['captures'],'Heldout capture entered donor law')
        nodes.append(dict(heldout_capture=heldout,donor_vectors=len(law['vectors']),
            complete_donor_nodes=sum(len(v['nodes']) for v in law['vectors'])))
        review=base.checked(test['sources'][0]);raw=base.checked(review['inputs']['raw.json']);rows=raw['request_records']
        jobs=defaultdict(list)
        for row in rows:jobs[row['profile_job_id']].append(row)
        for capture_quantile in (.25,.75):
            for seed in (0,1):
                rng=random.Random(seed);draws={};ordinals=Counter()
                for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'],r['model'])):
                    key=(row['model'],row['task'],row['pool_id'])
                    draw=work.choose_vector(law,key,capture_quantile=capture_quantile,within_capture_quantile=rng.random())
                    base.require(draw['status']=='TRAINING_JOINT_REQUEST_VECTOR_DRAW_ONLY','Missing exact public donor support')
                    draws[row['request_id']]=protocol.to_reference(draw['vector'],public_dispatch_ordinal=ordinals[key]);ordinals[key]+=1
                for job_rows in jobs.values():
                    scenario=sorted(job_rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index']))
                    origin=min(r['scheduled_time_ns'] for r in scenario);public=[];plans=defaultdict(list);slacks={}
                    for row in scenario:
                        key=(row['model'],row['task'],row['pool_id'])
                        public.append(dict(request_id='public:'+str(len(public)),owner=key[0],task=key[1],pool_id=key[2],arrival_ns=row['scheduled_time_ns']-origin))
                        plan,timing=draws[row['request_id']];plans[key].append(plan);slacks[plan]=timing
                    end=max(r['arrival_ns'] for r in public)+1;error=None;result=None
                    try:
                        result=reference.simulate(public,{k:tuple(v) for k,v in plans.items()},candidate.rate_table(fit),
                            window_start_ns=0,window_end_ns=end,drain_end_ns=end+60_000_000_000,
                            producer_fifo=tuple(sorted({k[0]+':modeled_stdout_lifetime' for k in plans})),protocol_slacks=slacks)
                    except UnknownEvidence as exc:error=str(exc);conflicts[error]+=1
                    for row,request in zip(scenario,public):
                        receipt=None if result is None else result['RESULT_receipts'].get(request['request_id'])
                        worlds.append(dict(heldout_capture=heldout,evaluation_request_id=row['request_id'],
                            evaluation_job_id=row['profile_job_id'],owner=row['model'],public_input_id=row['pool_id'],
                            capture_quantile=capture_quantile,seed=seed,actual_ns=row['completion_time_ns']-row['scheduled_time_ns'],
                            status='UNKNOWN' if error else 'SIMULATED',reason=error,
                            predicted_ns=None if receipt is None else receipt['numerator']/receipt['denominator']-request['arrival_ns']))
    return dict(candidate=fitted['candidate'],world_request_denominator=len(worlds),
        unknown_world_reasons=dict(conflicts),donor_denominators=nodes,ensembles=score_worlds(worlds),worlds=worlds)


def run(training,original_fit,macro_paths,output):
    training_ref=base.receipt(training);trained=base.checked(training_ref)
    docs,original,sources=work.verified_inputs(macro_paths,original_fit)
    sources.extend([training_ref,*trained['sources']]);results=[];skipped=[]
    for fit in trained['candidates']:
        if not fit['whole_capture_numerically_qualified']:
            skipped.append(dict(candidate=fit['candidate'],reason='FULL_OR_FOLD_NUMERICAL_OR_IDENTIFICATION_FAILURE',
                full_rank=fit['joint_response_jacobian'],full_qualification=fit['numerical_qualification'],
                folds=[dict(heldout_capture=f['heldout_capture_sha256'],rank=f['training_fit']['joint_response_jacobian'],
                    numerical_qualification=f['training_fit']['numerical_qualification']) for f in fit['whole_capture_leaveout']]))
            continue
        results.append(simulate_candidate(docs,fit,original))
    sources=strict_references([sources,implementation_sources(__file__)])
    for source in sources:base.require(base.receipt(source['path'])==source,'Source changed during generated comparison')
    report=dict(schema='gsdeploy.phase_heterogeneous_generated_training_diagnostics.v1',
        status='FIXED_PHASE_FAMILY_GENERATED_ENSEMBLE_TRAINING_DIAGNOSTICS_ONLY',sources=sources,
        candidates=results,skipped_ineligible=skipped,fixed_draws_per_request=4,whole_capture_replication_units=3,
        complete_macro_denominator_per_capture=334,heldout_request_identity_is_evaluation_only=True,
        source_jobs_and_context_not_prediction_lookup=True,heldout_observed_peer_path_used_for_generation=False,
        empty_initial_protocol_stations_assumed=True,dispatch_contract='after_READY_processing reference hypothesis',
        finite_four_atom_CRPS_is_population_calibration=False,parameters_selected_from_generated_scores=False,
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False,final_gain_eligible=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return base.receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--original-fit',required=True)
    p.add_argument('--training-macros',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.training,a.original_fit,a.training_macros,a.output)))
