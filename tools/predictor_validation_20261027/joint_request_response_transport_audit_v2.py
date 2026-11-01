"""Whole-training-capture diagnostics of an explicitly assumed joint transport.

Every offered scenario request remains in the denominator on FIFO conflict.
No validation capture or GPU work; no model/hyperparameter selection here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict,Counter
import io
import json
import math
from pathlib import Path
import random
import statistics
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from . import joint_request_response_transport_candidate_v2 as model
from . import local_joint_macro_candidate_v1 as local
from .macro_response_queue_reference_v1 import simulate

ROOT=Path(__file__).resolve().parents[2]
IMPORTED_REFERENCE_RECEIPT=local.receipt(Path(simulate.__code__.co_filename))


def rational_value(x):
    return x['numerator']/x['denominator']


def summarize(rows):
    known=[r for r in rows if r['status']=='SIMULATED_WITHOUT_FIFO_CONFLICT']
    out=dict(offered=len(rows),simulated=len(known),unknown=len(rows)-len(known),
        completed=sum(r.get('predicted_RESULT_reader_latency_ns') is not None for r in known))
    valid=[r for r in known if r.get('predicted_RESULT_reader_latency_ns') is not None]
    if valid:
        errors=[(r['predicted_RESULT_reader_latency_ns']-r['actual_RESULT_reader_latency_ns'])/1e6 for r in valid]
        out.update(reader_E2E_latency_MAE_ms=statistics.mean(abs(e) for e in errors),
            reader_E2E_latency_RMSE_ms=math.sqrt(statistics.mean(e*e for e in errors)))
    return out


def run(paths,fit_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    docs,fit,sources=model.verified_inputs(paths,fit_path)
    model.require(local.receipt(IMPORTED_REFERENCE_RECEIPT['path'])==IMPORTED_REFERENCE_RECEIPT,
        'Reference source changed since import')
    sources.extend([local.receipt(__file__),local.receipt(Path(model.__file__)),
        IMPORTED_REFERENCE_RECEIPT])
    full=model.build_law(docs,fit)
    (output/'joint-request-vectors.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    folds=[];all_records=[];example=None
    for fold in fit['whole_capture_leaveout']:
        heldout=fold['heldout_capture_sha256'];train=[];test=None
        for doc in docs:
            capture=doc['requests'][0]['macros'][0]['capture_sha256']
            if capture==heldout:test=doc
            else:train.append(doc)
        model.require(test is not None and len(train)==2,'Exactly one whole heldout training capture required')
        law=model.build_law(train,fold['training_fit'])
        model.require(heldout not in law['captures'],'Heldout capture entered donor law')
        review=local.checked(test['sources'][0]);raw=local.checked(review['inputs']['raw.json']);rows=raw['request_records']
        jobs=defaultdict(list)
        for row in rows:jobs[row['profile_job_id']].append(row)
        scenario_records=[];scenario_count=0;conflicts=Counter()
        # Two equal-weight common capture latent choices, two fixed independent
        # within-capture draw seeds. These fixed diagnostics are not tuning.
        for capture_quantile in (.25,.75):
            for seed in (0,1):
                rng=random.Random(seed);all_draws={};ordinal=Counter()
                for row in sorted(rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index'],r['model'])):
                    key=(row['model'],row['task'],row['pool_id'])
                    draw=model.choose_vector(law,key,capture_quantile=capture_quantile,
                        within_capture_quantile=rng.random())
                    model.require(draw['status']=='TRAINING_JOINT_REQUEST_VECTOR_DRAW_ONLY',
                        'Unknown exact public input must remain unknown, not a fallback')
                    all_draws[row['request_id']]=model.to_reference_plan(draw['vector'],public_dispatch_ordinal=ordinal[key])
                    ordinal[key]+=1
                for job_rows in jobs.values():
                    scenario_count+=1;scenario=sorted(job_rows,key=lambda r:(r['scheduled_time_ns'],r['arrival_index']))
                    origin=min(r['scheduled_time_ns'] for r in scenario);public=[];plans=defaultdict(list)
                    for row in scenario:
                        key=(row['model'],row['task'],row['pool_id'])
                        public.append(dict(request_id='public:'+str(len(public)),owner=key[0],task=key[1],pool_id=key[2],
                            arrival_ns=row['scheduled_time_ns']-origin))
                        plans[key].append(all_draws[row['request_id']])
                    end=max(r['arrival_ns'] for r in public)+1;error=None;result=None
                    try:
                        result=simulate(public,{k:tuple(v) for k,v in plans.items()},model.rate_table(law['beta']),
                            window_start_ns=0,window_end_ns=end,drain_end_ns=end+60_000_000_000,
                            producer_fifo=tuple(sorted({k[0]+':modeled_stdout_lifetime' for k in plans})))
                    except UnknownEvidence as exc:error=str(exc);conflicts[error]+=1
                    if result is not None and example is None:example=result
                    for source,request in zip(scenario,public):
                        record=dict(heldout_training_capture_sha256=heldout,common_capture_quantile=capture_quantile,
                            within_capture_seed=seed,public_owner=source['model'],public_input_id=source['pool_id'],
                            offered_arrival_offset_ns=request['arrival_ns'],
                            actual_RESULT_reader_latency_ns=source['completion_time_ns']-source['scheduled_time_ns'])
                        if error:record.update(status='UNKNOWN_WHOLE_WORLD_FIFO_CONFLICT',reason=error)
                        else:
                            receipt=result['RESULT_receipts'].get(request['request_id'])
                            record.update(status='SIMULATED_WITHOUT_FIFO_CONFLICT',
                                predicted_RESULT_reader_latency_ns=None if receipt is None else rational_value(receipt)-request['arrival_ns'])
                        scenario_records.append(record)
        model.require(len(scenario_records)==len(rows)*4,'All offered scenario requests must remain in denominator')
        folds.append(dict(heldout_training_capture_sha256=heldout,
            training_donor_captures=law['captures'],beta_from_training_captures_only=law['beta'],
            original_requests=len(rows),offered_jobs=len(jobs),fixed_scenarios=scenario_count,
            diagnostics=summarize(scenario_records),unknown_world_reasons=dict(conflicts),records=scenario_records))
        all_records.extend(scenario_records)
    modules=['tests.test_joint_request_response_transport_candidate_v2','tests.test_local_joint_macro_candidate_v1']
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    (output/'tests.txt').write_text(log.getvalue());model.require(result.wasSuccessful(),'CPU counterexamples failed')
    sources.extend([local.receipt(ROOT/(m.replace('.','/')+'.py')) for m in modules])
    for ref in sources:model.require(local.receipt(ref['path'])==ref,'Source changed during transport audit')
    (output/'whole-training-capture-forward-diagnostics.json').write_text(json.dumps(folds,indent=2,allow_nan=False)+'\n')
    if example:(output/'one-generated-world.json').write_text(json.dumps(example,indent=2,allow_nan=False)+'\n')
    report=dict(schema='gsdeploy.joint_request_response_transport_training_diagnostics.v2',
        status='TRAINING_JOINT_REQUEST_VECTOR_VIRTUAL_TRANSPORT_DIAGNOSTICS_ONLY',sources=sources,
        law=local.receipt(output/'joint-request-vectors.json'),
        training_captures=len(full['captures']),requests=len(full['vectors']),
        macro_nodes=sum(len(v['nodes']) for v in full['vectors']),
        query_marks=sum(m['action']=='query' for v in full['vectors'] for n in v['nodes'] for m in n['marks']),
        complete_entry_OCR_and_READY_tail_retained=True,equal_capture_common_latent=True,
        whole_capture_leaveout=[{k:v for k,v in f.items() if k!='records'} for f in folds],
        pooled_forward_diagnostics=summarize(all_records),
        diagnostics=local.receipt(output/'whole-training-capture-forward-diagnostics.json'),
        tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=local.receipt(output/'tests.txt')),
        learned_response_candidate='visible_phase_aware_second_order_qualified',
        rational_rate_grid_denominator=model.RATE_DENOMINATOR,
        dynamic_future_peer_path_generated=True,observed_heldout_peer_path_used_for_simulation=False,
        donor_vectors_from_heldout_capture_used=False,public_input_identity_required=True,
        original_absolute_donor_times_replayed=False,cohort_or_request_id_template_lookup=False,
        query_true_ends_macro=False,physical_context_as_SM_service=False,
        notification_delay_transport='paired_three_leg_lags_unchanged; FIFO conflicts invalidate whole world',
        request_dispatch_contract='per-owner FIFO after READY processing; explicit reference hypothesis',
        RESULT_E2E_endpoint='scheduled_arrival_to_RESULT_reader_receipt',
        dispatch_to_worker_START_jointly_sampled=True,
        observed_submit_to_START_is_pure_handoff_qualified=False,
        whole_training_capture_is_replication_unit=True,
        model_selected_from_these_forward_diagnostics=False,probability_calibrated=False,
        native_assimilation_eligible=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False,
        limitations=['The response fit conditions on observed paths; transporting work and marks into generated paths is a new assumption.',
            'Residual vectors preserve all nodes within a request; one equally weighted capture latent is shared, with conditional independent request draws inside that block.',
            'Notice lags are empirical paired residuals, not identified reader/CPU service demands; a new wall can violate FIFO and that whole scenario stays UNKNOWN.',
            'Exact known input/owner and frozen generation shapes are required; no unseen content/stop/TTS template is qualified.',
            'The 26 jobs within each capture are public offered diagnostic scenarios, not independent capture replicates.',
            'Submit-to-START donor lag may contain transport/worker-queue wait; the actual sender can pipeline without waiting for READY processing. Parent dispatch-after-READY is an unqualified reference hypothesis.',
            'RESULT reader E2E is distinct from collector processing that controls peer visibility. No resource/quality/scoring provider, six-term utility, P95 or DeltaU qualification is supplied.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return local.receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--response-fit',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.training_macros,args.response_fit,args.output)))
