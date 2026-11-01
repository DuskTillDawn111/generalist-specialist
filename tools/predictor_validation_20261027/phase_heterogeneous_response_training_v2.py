"""Checkpointed fixed designs; retain exact structural failures without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np

from . import phase_heterogeneous_response_training_v1 as model
from . import macro_joint_response_fit_v1 as base
from . import macro_projection_contrast_training_v1 as numeric
from .implementation_source_closure_v27 import strict_references,implementation_sources


def rank_diagnostic(rows,features):
    profile=base.profile(np.zeros(len(features)),rows);derivative=profile[3];groups=sorted(profile[1],key=repr)
    nuisance=np.zeros((len(rows),len(groups)))
    for j,g in enumerate(groups):
        for i,row in enumerate(rows):
            if row['group']==g:nuisance[i,j]=-1
    joint=np.column_stack((nuisance,derivative))
    return dict(rank=int(np.linalg.matrix_rank(joint)),parameters=len(groups)+len(features),
        full_column_rank=int(np.linalg.matrix_rank(joint))==len(groups)+len(features),
        evaluated_at_zero_response_coefficients=True)


def structural_failure(rows,features):
    zeros=[features.index('same@'+c) for c in ('text:ready_tail','generalist_2:ready_tail','ocr:ocr_inference','ocr:ready_tail')]
    a=features.index('count@ocr:entry');b=features.index('same@ocr:entry')
    zero_columns=all(np.all(row['matrix'][:,zeros]==0) for row in rows)
    same_columns=all(np.array_equal(row['matrix'][:,a],row['matrix'][:,b]) for row in rows)
    base.require(zero_columns and same_columns,'Exact piecewise structural proof required before skipping numerical fit')
    return dict(status='STRUCTURAL_RANK_FAILURE_NUMERICAL_FIT_SKIPPED',
        exact_all_piecewise_zero_columns=[features[i] for i in zeros],
        exact_all_piecewise_identical_columns=[features[a],features[b]],
        original_feature_dimension=len(features),no_columns_deleted=True,
        all_macros_retained=len(rows),numerical_optimizer_not_run=True)


def run(paths,prior_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prior_ref=base.receipt(prior_path);prior=base.checked(prior_ref)
    base.require(prior['schema']=='gsdeploy.macro_joint_response_second_order_training_leaveout.v1'
        and prior['validation_labels_read'] is False,'Original phase-aware source required')
    docs=[];sources=[prior_ref,*prior['sources']]
    for path in paths:
        doc,closure=base.verified_document(path);docs.append(doc);sources.extend(closure)
    captures=sorted({m['capture_sha256'] for d in docs for r in d['requests'] for m in r['macros']})
    base.require(len(docs)==len(captures)==3,'Three whole training captures required')
    original=next(c for c in prior['candidates'] if c['candidate']=='visible_phase_aware')
    diagnostics,refs=model.residual_diagnostics(docs,original);sources.extend(refs)
    (output/'residual-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    checkpoints=[];results=[]
    def save(label,value):
        name='checkpoint-'+str(len(checkpoints)).zfill(2)+'-'+label+'.json'
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        checkpoints.append(base.receipt(output/name));print(json.dumps(dict(checkpoint=name)),flush=True)
    for name,features in model.DESIGNS.items():
        rows=model.prepared(docs,features)
        if name=='own_family_phase_count_and_same':
            failure=structural_failure(rows,features);folds=[]
            for heldout in captures:
                train=[r for r in rows if r['capture']!=heldout]
                folds.append(dict(heldout_capture_sha256=heldout,training_fit=dict(
                    numerical_qualification=False,joint_response_jacobian=rank_diagnostic(train,features),
                    structural_failure=structural_failure(train,features))))
            fitted=dict(candidate=name,features=list(features),numerical_qualification=False,
                whole_capture_numerically_qualified=False,joint_response_jacobian=rank_diagnostic(rows,features),
                structural_failure=failure,whole_capture_leaveout=folds,
                all_macro_nodes_retained=True,pooled_leaveout=None)
            results.append(fitted);save(name+'-structural-failure',fitted);continue
        full,beta,mu,variance=numeric.fit(rows,features);save(name+'-full',full);folds=[];pooled=[]
        for ordinal,heldout in enumerate(captures):
            fitted,b,m,v=numeric.fit([r for r in rows if r['capture']!=heldout],features)
            evaluated=base.evaluate([r for r in rows if r['capture']==heldout],b,m,v)
            fold=dict(heldout_capture_sha256=heldout,training_fit=fitted,evaluation=evaluated)
            folds.append(fold);pooled.extend(evaluated['records']);save(name+'-fold'+str(ordinal),fold)
        full.update(candidate=name,whole_capture_leaveout=folds,pooled_leaveout=base.summarize(pooled),
            whole_capture_numerically_qualified=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification'] for f in folds)),
            all_macro_nodes_retained=len(rows)==1002,coefficients_are_free_real=True)
        results.append(full)
    sources=strict_references([sources,implementation_sources(__file__),checkpoints])
    for source in sources:base.require(base.receipt(source['path'])==source,'Source changed during checkpointed fit')
    report=dict(schema='gsdeploy.phase_heterogeneous_response_training.v2',
        status='FIXED_PHASE_FAMILY_RESPONSE_CHECKPOINTED_TRAINING_DIAGNOSTICS',sources=sources,candidates=results,
        residual_diagnostics=base.receipt(output/'residual-diagnostics.json'),macro_rows=1002,capture_replication_units=3,
        design_fixed_before_this_fit=True,structural_failure_columns_deleted=False,
        context_or_job_lookup_used_for_fit=False,conditional_NLL_does_not_qualify_generated_queue=True,
        main_queue_provider_eligible=False,final_gain_eligible=False,selected_for_prospective_provider=None,
        independent_validation_read=False,validation_labels_read=False,GPU_started=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return base.receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training-macros',nargs='+',required=True);p.add_argument('--prior-fit',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.training_macros,a.prior_fit,a.output)))
