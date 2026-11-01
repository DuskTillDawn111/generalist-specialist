"""Read-only actual v42 coefficient/accounting and v44 numeric bridge review."""
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import projected_handler_context_v42 as v
from . import projected_handler_bridge_v44 as bridge
from . import projected_handler_forecast_v44 as forecast
from .implementation_source_closure_v27 import implementation_sources
from .owner_lifecycle_mark_v21 import OwnerLifecycle
from .unknown_preflight_archive_v1 import save_synced


def run(directory,output):
    directory=Path(directory);output=Path(output);assert not output.exists()
    model=json.loads((directory/'model.json').read_text());report=json.loads((directory/'report.json').read_text())
    candidate=v.ObservableContextCandidate(model)
    assert report['model']==v.receipt(directory/'model.json')
    marks=[];capture_summary=[]
    for capture,entry in model['captures'].items():
        assert v.receipt(entry['report']['path'])==entry['report']
        source=json.loads(Path(entry['report']['path']).read_text())
        assert v.receipt(source['mark_records']['path'])==source['mark_records']
        with gzip.open(source['mark_records']['path'],'rt') as handle:rows=[json.loads(line) for line in handle]
        assert len(rows)==source['mark_record_count'] and all(r['capture']==capture for r in rows)
        targets=Counter(r['next_owner'] for r in rows)
        fold=next(f for f in report['whole_capture_holdout'] if f['held_out_capture']==capture)
        assert set(fold['training_captures'])==set(model['captures'])-{capture}
        assert fold['metrics']['physical_mark']['all']['n']==len(rows)
        for owner in v.OWNERS:assert fold['metrics']['physical_mark']['target='+owner]['n']==targets[owner]
        capture_summary.append(dict(capture=capture,route=entry['route'],marks=len(rows),target_counts=dict(targets),
            fold_capture_disjointness=True,reported_fold_metrics=fold['metrics']['physical_mark']['all']))
        marks.extend(rows)
    fitted=model['fitted']['mark'];x=np.asarray([v.mark_vector(r['features']) for r in marks])
    logits=((x-np.asarray(fitted['center']))/np.asarray(fitted['scale']))@np.asarray(fitted['coefficients'])
    weights=np.exp(logits-logits.max(axis=1,keepdims=True));prob=weights/weights.sum(axis=1,keepdims=True)
    actual=np.asarray([v.OWNERS.index(r['next_owner']) for r in marks]);chosen=prob.argmax(axis=1)
    metrics=dict(n=len(marks),accuracy=float(np.mean(chosen==actual)),NLL=float(-np.log(prob[np.arange(len(marks)),actual]).mean()),
        multiclass_Brier=float(((prob-np.eye(4)[actual])**2).sum(axis=1).mean()),
        same_held_prediction_count=sum(v.OWNERS[p]==r['features']['held_owner'] for p,r in zip(chosen,marks)))
    for key,value in metrics.items():assert abs(value-report['training_metrics']['physical_mark']['all'][key])<1e-12
    hazard=model['fitted']['hazard'];coefficients=dict(zip(hazard['columns'],hazard['coefficients']));expected=0.
    for row in model['fitted']['risk_accounting']['rows']:
        eta=coefficients['intercept']+coefficients.get('age_bin:'+str(row['age_bin']),0.)+sum(coefficients.get(f,0.) for f in row['features'])
        expected+=float(np.exp(eta))*row['exposure_ns']/1e6
    target=next(r for r in report['training_metrics']['physical_SAVE'] if r['stratum']=='all')
    assert abs(expected-target['expected_SAVE'])<1e-7
    # Constructor/source verification above is reused; bridge methods are real.
    mark=object.__new__(bridge.ObservableReferenceMark);mark.candidate=candidate;mark.states=mark.clock=None
    states={o:OwnerLifecycle(o=='text#0',None if o=='text#0' else False,False) for o in v.OWNERS}
    mark.observe_view(SimpleNamespace(clock_ns=20000,owner_lifecycle=tuple(states.items())))
    query=bridge.feature_projection('text#0',None,20000,states)
    result=mark.predict_given_save_now(held_owner='text#0',context_age_ns=20000,pending_frontier=[])
    assert result['next_owner_probabilities']['text#0']==1. and set(result['physical_owner_probabilities'])==set(v.OWNERS)
    assert result['same_holder_probability_removed'] is False and result['active_reference_service_bridge_is_unvalidated_hypothesis'] is True
    assert result['physical_owner_probabilities']==candidate.next_owner(query)['physical_owner_probabilities']
    sources=implementation_sources(forecast.__file__)
    assert {'projected_handler_context_v42.py','projected_handler_bridge_v44.py','learned_lifecycle_scheduler_v27.py'}<={Path(r['path']).name for r in sources}
    output.mkdir()
    review=dict(schema='gsdeploy.actual_v42_fit_v44_bridge_independent_CPU_review.v1',status='PASS_WITH_UNVALIDATED_SERVICE_HYPOTHESIS',
        model=v.receipt(directory/'model.json'),author_report=v.receipt(directory/'report.json'),
        implementation=v.receipt(v.__file__),review_implementation=v.receipt(__file__),sources_verified=len(model['sources']),
        v44_implementation_closure=sources,captures=capture_summary,recomputed_training_mark_metrics=metrics,
        recomputed_training_expected_SAVE=expected,risk_accounting=model['fitted']['risk_accounting']['counts'],
        real_v44_bridge_query=query,real_v44_bridge_result=result,SAVE_query=candidate.save_hazard(query,1_000_000),
        independent_bridge_tests=3,independent_model_tests=4,
        independent_test_sources=[v.receipt('tests/test_projected_handler_bridge_v44_independent_review.py'),v.receipt('tests/test_projected_handler_context_v42_independent_review.py')],
        fold_numeric_prediction_recomputed=False,fold_limitation='Fold coefficients were not saved; checked filtering, source identities, target counts and declared convergence, without refit.',
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        physical_context_equals_service=False,worker_handler_projection_calibrated=False,
        independent_prediction_accuracy_validated=False,implementation_findings=[])
    save_synced(output/'report.json',review)
    print(json.dumps(dict(report=v.receipt(output/'report.json'),metrics=metrics,bridge=result)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.directory,a.output)
