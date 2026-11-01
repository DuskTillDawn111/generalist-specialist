"""Correct finite-world versus point-prediction scoring, no new selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .local_joint_macro_candidate_v1 import checked,receipt


def require(v,message):
    if not v:raise ValueError(message)


def summarize(rows):
    return dict(requests=len(rows),
        average_random_world_absolute_error_ms=statistics.mean(r['mean_world_absolute_error_ns'] for r in rows)/1e6,
        four_world_ensemble_mean_MAE_ms=statistics.mean(abs(r['ensemble_mean_error_ns']) for r in rows)/1e6,
        four_world_ensemble_mean_RMSE_ms=math.sqrt(statistics.mean(r['ensemble_mean_error_ns']**2 for r in rows))/1e6,
        four_world_ensemble_median_MAE_ms=statistics.mean(abs(r['ensemble_median_error_ns']) for r in rows)/1e6,
        four_world_ensemble_median_RMSE_ms=math.sqrt(statistics.mean(r['ensemble_median_error_ns']**2 for r in rows))/1e6,
        finite_empirical_CRPS_ms=statistics.mean(r['empirical_CRPS_ns'] for r in rows)/1e6,
        four_sample_minmax_coverage=sum(r['within_four_sample_minmax'] for r in rows)/len(rows),
        four_sample_mean_minmax_width_ms=statistics.mean(r['four_sample_max_ns']-r['four_sample_min_ns'] for r in rows)/1e6)


def run(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ref=receipt(source);report=checked(ref);folds=checked(report['diagnostics']);records=[]
    macro_documents=[];manifest_refs=[]
    for entry in report['sources']:
        path=Path(entry['path'])
        if path.name.startswith('capture') and path.name.endswith('-macros.json'):
            doc=checked(entry)
            if doc.get('schema')=='gsdeploy.composable_response_macro_training.v1':
                macro_documents.append(doc);manifest_refs.append(entry)
    require(len(macro_documents)==3,'Exactly three source-bound macro manifests required')
    for fold in folds:
        worlds=fold['records'];count=fold['original_requests']
        doc=next(d for d in macro_documents if d['requests'][0]['macros'][0]['capture_sha256']==fold['heldout_training_capture_sha256'])
        training_review=checked(doc['sources'][0]);raw_ref=training_review['inputs']['raw.json'];raw=checked(raw_ref)
        manifest_refs.extend([doc['sources'][0],raw_ref]);jobs={}
        for row in raw['request_records']:jobs.setdefault(row['profile_job_id'],[]).append(row)
        manifest=[row for job in jobs.values() for row in sorted(job,key=lambda r:(r['scheduled_time_ns'],r['arrival_index']))]
        require(len(manifest)==count and len({r['request_id'] for r in manifest})==count,
            'Complete distinct heldout request manifest required')
        require(len(worlds)==4*count,'Fixed four-world per request scenario ordering required')
        for ordinal in range(count):
            samples=[worlds[k*count+ordinal] for k in range(4)]
            original=manifest[ordinal]
            origin=min(r['scheduled_time_ns'] for r in jobs[original['profile_job_id']])
            expected=(original['model'],original['pool_id'],original['scheduled_time_ns']-origin,
                original['completion_time_ns']-original['scheduled_time_ns'])
            require(all(r['status']=='SIMULATED_WITHOUT_FIFO_CONFLICT' for r in samples),
                'Incomplete ensembles must remain UNKNOWN; cannot silently average successful subset')
            require(all((r['public_owner'],r['public_input_id'],r['offered_arrival_offset_ns'],
                r['actual_RESULT_reader_latency_ns'])==expected for r in samples),
                'Source-rederived original offered ordinal alignment changed')
            y=samples[0]['actual_RESULT_reader_latency_ns'];xs=[r['predicted_RESULT_reader_latency_ns'] for r in samples]
            require(all(x is not None for x in xs),'Unfinished ensemble has unknown full latency')
            absdiff=statistics.mean(abs(x-y) for x in xs)
            pairwise=sum(abs(a-b) for a in xs for b in xs)/(2*len(xs)**2)
            records.append(dict(heldout_training_capture_sha256=fold['heldout_training_capture_sha256'],
                offered_request_ordinal=ordinal,owner=samples[0]['public_owner'],public_input_id=samples[0]['public_input_id'],
                heldout_request_id_for_evaluation_only=original['request_id'],
                heldout_job_id_for_evaluation_only=original['profile_job_id'],
                actual_ns=y,samples_ns=xs,mean_world_absolute_error_ns=absdiff,
                ensemble_mean_error_ns=statistics.mean(xs)-y,ensemble_median_error_ns=statistics.median(xs)-y,
                empirical_CRPS_ns=absdiff-pairwise,
                four_sample_min_ns=min(xs),four_sample_max_ns=max(xs),within_four_sample_minmax=min(xs)<=y<=max(xs)))
    by_owner=defaultdict(list)
    for r in records:by_owner[r['owner']].append(r)
    result=dict(schema='gsdeploy.protocol_transport_finite_ensemble_diagnostics.v3',
        status='TRAINING_FOUR_WORLD_ENSEMBLE_SCORING_DIAGNOSTICS_ONLY',sources=[ref,report['diagnostics'],receipt(__file__),*manifest_refs],
        pooled=summarize(records),by_owner={k:summarize(v) for k,v in by_owner.items()},
        worlds_per_request=4,replication_unit='three_whole_training_captures_not_150_independent_captures',
        world_score_is_point_prediction_MAE=False,four_sample_minmax_is_90pct_interval=False,
        alignment='source-rederived raw-request/job insertion order and each job scheduled/arrival-index order; fixed four scenario blocks',
        heldout_identity_used_for_evaluation_only=True,actual_latency_used_as_grouping_key=False,
        CRPS_scope='proper score of the explicit four-atom empirical forecast; small chosen ensemble not calibrated population law',
        parameter_or_model_selection_performed=False,validation_labels_read=False,GPU_started=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,records=records)
    for r in result['sources']:require(receipt(r['path'])==r,'Source changed during scoring')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n');return receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.source,a.output)))
