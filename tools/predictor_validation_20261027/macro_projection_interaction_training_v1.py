"""Complete-domain partial projection interaction diagnostic, CPU only.

Own support is retained as explicit metadata. It is not fitted as a standalone
main effect: the group response baseline remains empirical, while unknown
interactions use U and known interactions use bounded contrast sums. The prior
unsupported-main-effect rank failure is preserved as a separate report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import macro_projection_contrast_training_v1 as projection
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA='gsdeploy.partial_decoder_projection_interaction_macro_training.v1'
CANDIDATES={
    'phase_and_unknown_projection_pairs':(projection.COUNT,projection.PHASE,projection.UNKNOWN),
    'phase_unknown_pairs_projection_contrast':(projection.COUNT,projection.PHASE,projection.UNKNOWN,projection.CONTRAST),
}


def run(paths,prior_path,inventory_path):
    base=projection.base;prior_ref=base.receipt(prior_path);prior=base.checked(prior_ref)
    base.require(prior['schema']==projection.SCHEMA and prior['validation_labels_read'] is False,
        'Complete-domain source-bound contrast report required')
    for ref in prior['sources']:
        base.require(base.receipt(ref['path'])==ref,'Prior contrast source changed')
    inventory,bindings,sources=projection.checked_static_inventory(inventory_path)
    sources.append(prior_ref);docs=[];records=[]
    for path in paths:
        doc,closure=base.verified_document(path);docs.append(doc);sources.extend(closure)
        enriched,refs=projection.enrich(doc,bindings,inventory);records.extend(enriched);sources.extend(refs)
    base.require(records==prior['feature_records'],'Interaction features must exactly rederive from unchanged full contrast report')
    captures={r['capture_sha256'] for r in records}
    base.require(len(docs)==3 and len(captures)==3 and len(records)==1002,'Three complete captures/all1002 macros required')
    results=[]
    for candidate,features in CANDIDATES.items():
        rows=projection.prepared(docs,records,features);full,beta,mu,variance=projection.fit(rows,features)
        folds=[];all_records=[]
        for heldout in sorted(captures):
            fitted,b,m,v=projection.fit([r for r in rows if r['capture']!=heldout],features)
            evaluated=base.evaluate([r for r in rows if r['capture']==heldout],b,m,v)
            all_records.extend(evaluated['records']);folds.append(dict(heldout_capture_sha256=heldout,
                training_fit=fitted,evaluation=evaluated))
        full.update(candidate=candidate,whole_capture_leaveout=folds,pooled_leaveout=base.summarize(all_records),
            leaveout_selection_eligible=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification']
                and f['evaluation']['summary']['unknown_rows']==0 for f in folds)),
            all_capture_conditional_fit=base.evaluate(rows,beta,mu,variance))
        results.append(full)
    # Include original lower-dimensional references in the same conditional
    # comparison. Failed own-main-effect candidates remain visibly ineligible.
    all_candidates=[*prior['candidates'],*results]
    eligible=[r for r in all_candidates if r['leaveout_selection_eligible']]
    selected=min(eligible,key=lambda r:r['pooled_leaveout']['mean_nll'])['candidate'] if eligible else None
    contrast=results[-1];without=results[0]
    differences=[dict(capture_sha256=a['heldout_capture_sha256'],
        contrast_minus_without_contrast_mean_nll=a['evaluation']['summary']['mean_nll']-b['evaluation']['summary']['mean_nll'])
        for a,b in zip(contrast['whole_capture_leaveout'],without['whole_capture_leaveout'])]
    closure=strict_references([sources,implementation_sources(__file__)])
    base.require(all(base.receipt(r['path'])==r for r in closure),'Source changed during interaction fit')
    metadata={k:v for k,v in prior.items() if k not in ('schema','status','sources','candidates','selected_candidate','feature_records')}
    return dict(metadata,schema=SCHEMA,status='COMPLETE_DOMAIN_PARTIAL_PROJECTION_INTERACTION_TRAINING_DIAGNOSTIC',
        sources=closure,prior_failed_main_effect_report=prior_ref,candidates=all_candidates,selected_candidate=selected,
        feature_records_source=dict(report=prior_ref,key='feature_records',exact_rederivation_performed=True),
        own_support_still_recorded_for_all_pieces=True,
        standalone_own_unsupported_main_effect_fitted_in_new_candidates=False,
        own_response_baseline='Empirical owner/input/macro nuisance; no structural own-service extrapolation.',
        interaction_missingness_policy='U covers every unknown pair; K known-pair contrast is an interaction sum, never an imputed unknown own resource.',
        comparison_is_additional_training_design_after_rank_failure=True,
        independent_validation_of_new_candidate_design=False,
        heldout_capture_contrast_increment=differences,
        contrast_increment_inference='Three paired training-capture differences only; no robust significance or prospective gain claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--prior-report',required=True);parser.add_argument('--static-inventory',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.training_macros,args.prior_report,args.static_inventory)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],selected_candidate=result['selected_candidate'])))
