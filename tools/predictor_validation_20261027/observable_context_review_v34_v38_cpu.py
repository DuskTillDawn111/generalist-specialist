"""Read-only actual v34 numerical audit and frozen v38 failure diagnosis."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import whole57_observable_context_v34 as v
from .unknown_preflight_archive_v1 import save_synced


def read(path):return json.loads(Path(path).read_text())


def run(model_directory,preflight_directory,output):
    output=Path(output); assert not output.exists()
    model_directory=Path(model_directory); preflight_directory=Path(preflight_directory)
    model=read(model_directory/'model.json'); report=read(model_directory/'report.json')
    assert report['model']==v.receipt(model_directory/'model.json')
    candidate=v.ObservableContextCandidate(model)  # Frozen digest, shape, finite coefficients and all sources.
    marks=[]; risks=[]; counts=[]
    for capture,entry in model['captures'].items():
        assert v.receipt(entry['report']['path'])==entry['report']
        source=read(entry['report']['path'])
        def rows(ref):
            assert v.receipt(ref['path'])==ref
            with gzip.open(ref['path'],'rt') as handle:return [json.loads(line) for line in handle]
        cm=rows(source['mark_records']); cr=rows(source['risk_chronology'])
        assert all(r['capture']==capture for r in [*cm,*cr])
        v.verify_native_accounting(source,read(source['source_adapter']['path']),cm,cr)
        marks.extend(cm);risks.extend(cr)
        targets=Counter(r['next_owner'] for r in cm)
        fold=next(f for f in report['whole_capture_holdout'] if f['held_out_capture']==capture)
        assert set(fold['training_captures'])==set(model['captures'])-{capture}
        assert fold['held_out_route']==entry['route']
        assert fold['metrics']['physical_mark']['all']['n']==len(cm)
        for owner in v.OWNERS:assert fold['metrics']['physical_mark']['target='+owner]['n']==targets[owner]
        counts.append(dict(capture=capture,route=entry['route'],marks=len(cm),targets=dict(targets),
            risk_intervals=len(cr),exposure_ns=sum(r['end_ns']-r['start_ns'] for r in cr),
            paired_SAVE=sum(r['physical_SAVE_END_observed'] for r in cr),
            declared_fold_capture_disjointness_verified=True,
            reported_fold_OCR_accuracy=fold['metrics']['physical_mark']['target=ocr#0']['accuracy']))
    # Independent stable-softmax calculation from retained coefficients, not evaluator output.
    mark=model['fitted']['mark']; x=np.asarray([v.mark_vector(r['features']) for r in marks])
    logits=((x-np.asarray(mark['center']))/np.asarray(mark['scale']))@np.asarray(mark['coefficients'])
    weights=np.exp(logits-logits.max(axis=1,keepdims=True)); probabilities=weights/weights.sum(axis=1,keepdims=True)
    actual=np.asarray([v.OWNERS.index(r['next_owner']) for r in marks]); chosen=probabilities.argmax(axis=1)
    metrics=dict(n=len(marks),accuracy=float(np.mean(chosen==actual)),
        NLL=float(-np.log(probabilities[np.arange(len(marks)),actual]).mean()),
        multiclass_Brier=float(((probabilities-np.eye(4)[actual])**2).sum(axis=1).mean()),
        same_held_prediction_count=sum(v.OWNERS[p]==r['features']['held_owner'] for p,r in zip(chosen,marks)))
    for key,value in metrics.items():assert abs(value-report['training_metrics']['physical_mark']['all'][key])<1e-12
    assert Counter(r['next_owner'] for r in marks)==Counter(mark['label_counts'])
    # Independently evaluate the saved aggregate hazard design as an event count.
    hazard=model['fitted']['hazard']; coefs=dict(zip(hazard['columns'],hazard['coefficients']))
    expected=0.
    for row in model['fitted']['risk_accounting']['rows']:
        linear=coefs['intercept']+coefs.get('age_bin:'+str(row['age_bin']),0.)+sum(coefs.get(f,0.) for f in row['features'])
        expected+=float(np.exp(linear))*row['exposure_ns']/1e6
    recorded=next(r for r in report['training_metrics']['physical_SAVE'] if r['stratum']=='all')
    assert abs(expected-recorded['expected_SAVE'])<1e-7
    ocr_features=dict(held_owner='ocr#0',previous_held_owner='text#0',context_age_ns=77768,
        handler_active={o:None for o in v.OWNERS})
    output.mkdir()
    audit=dict(schema='gsdeploy.actual_v34_fit_independent_CPU_review.v1',status='PASS_WITH_PREDICTION_LIMITATIONS',
        model=v.receipt(model_directory/'model.json'),author_report=v.receipt(model_directory/'report.json'),
        implementation=v.receipt(v.__file__),review_implementation=v.receipt(__file__),
        sources_fully_verified=len(model['sources']),captures=counts,recomputed_final_mark_metrics=metrics,
        recomputed_final_hazard_expected_SAVE=expected,observed_SAVE=recorded['observed_SAVE'],
        OCR_historical_failure_query=dict(features=ocr_features,next_owner=candidate.next_owner(ocr_features),
            SAVE=candidate.save_hazard(ocr_features,1_000_000)),
        fold_numerical_predictions_independently_recomputed=False,
        fold_limitation='Fold models were not retained. Capture filtering, identities, counts, source code and reported convergence were checked; fold probabilities cannot be independently reconstructed without refitting.',
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        prediction_accuracy_certified=False,active_service_bridge_validated=False)
    save_synced(output/'v34-fit-review.json',audit)
    preflight=read(preflight_directory/'report.json'); cells=[]
    for cell in preflight['cells']:
        ref=cell['result']['artifact']; assert v.receipt(ref['path'])==ref
        encoded=gzip.decompress(Path(ref['path']).read_bytes())
        assert hashlib.sha256(encoded).hexdigest()==cell['result']['uncompressed_sha256']
        assert len(encoded)==cell['result']['uncompressed_bytes']
        forecast=json.loads(encoded);scheduler=forecast['scheduler_diagnostic'];oracle=scheduler['oracle']
        latest=oracle['latest_cumulative_hazard'];state=latest['state'];lifecycle=state['lifecycle'];held=state['held_owner']
        assert forecast['reason']=='UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION'
        assert oracle['decisions']==[] and len(forecast['full_public_cohort'])==57
        active=[o for o,life in lifecycle.items() if life['handler_active']]
        assert active==[held] and state['frontier']==[]
        query=dict(held_owner=held,previous_held_owner=None,context_age_ns=latest['restore_age_ns'],
            handler_active={o:None for o in v.OWNERS})
        physical=candidate.next_owner(query)
        cells.append(dict(candidate_id=cell['candidate_id'],artifact=ref,status=forecast['status'],
            reason=forecast['reason'],held_owner=held,projected_engine_lifecycle=lifecycle,
            SAVE_clock_ns=latest['absolute_save_candidate_ns'],attained_context_age_ns=latest['restore_age_ns'],
            elapsed_from_observation_ns=latest['absolute_save_candidate_ns']-latest['observation_ns'],
            visible_update_count=state['visible_update_count'],uncertainty_reasons=latest['uncertainty_reasons'],
            hazard_budget=state['hazard_budget'],consumed_hazard=state['consumed_hazard'],
            physical_probabilities_recomputed=physical['physical_owner_probabilities'],
            conditional_engine_work_domain_probabilities={held:1.},guard_preserved=True,
            initial_held_owner_is_observed_history=scheduler['initial_restore_is_observed_history'],
            source_bound_gzip_roundtrip_verified=True))
    diagnostic=dict(schema='gsdeploy.actual_v38_single_handler_independent_CPU_diagnosis.v1',
        status='CONFIRMED_MODEL_SERVICE_INTERFACE_GAP_AND_INCOMPLETE_CAMPAIGN',
        preflight_report=v.receipt(preflight_directory/'report.json'),model=audit['model'],cells=cells,
        failure=preflight['failure'],
        campaign_limitation='Two candidates ran; repeated exclusive progress-report creation stopped the third and fourth. The final report binds both balanced and skewed gzip evidence; the stale progress report binds only balanced.',
        physical_service_conclusion='An unconditioned physical SAVE candidate exhausts its budget when only held handler remains eligible. The work-domain bridge normalizes retained physical held mass to one, and the original guard correctly returns UNKNOWN. This does not prove a physical service interruption.',
        GPU_started=False,validation_labels_read=False,source_bytes_unchanged=preflight['source_bytes_unchanged'])
    save_synced(output/'v38-diagnostic.json',diagnostic)
    print(json.dumps(dict(fit_review=v.receipt(output/'v34-fit-review.json'),
        diagnostic=v.receipt(output/'v38-diagnostic.json'),recomputed_metrics=metrics,cells=cells)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory',required=True);p.add_argument('--preflight-directory',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.model_directory,args.preflight_directory,args.output)
