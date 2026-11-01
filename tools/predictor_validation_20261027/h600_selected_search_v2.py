"""Fresh plain/bounded DFS on the reserved H600 four-route model domain."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from . import h600_selected_validation_v2 as validation
from . import h600_resources_v1, h600_search_core_v2 as core
from .native_flow import checked,receipt,save
from .probe_validation import read
from .h600_selected_validation_v2 import predictor_sources


def run(protocol_path,model_path,resource_manifest_path,quality_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=validation.audit_protocol(protocol_path)
    public_plan=validation.audit_plan(protocol['target_plan']['path'],protocol_path)
    model_verification=validation.audit_model(model_path,protocol['selection']['path'])
    resource_manifest=validation.audit_resources(resource_manifest_path)
    full=resource_manifest['full'];resources=read(checked(full['resources']))
    model=read(model_path);quality_doc=read(quality_path)
    context=public_plan['execution_context_sha256']
    if (model.get('execution_context_sha256')!=context or resources.get('execution_context_sha256')!=context
            or quality_doc.get('execution_context_sha256',context)!=context
            or quality_doc.get('split','training')!='training'):
        raise ValueError('Matching execution identities and training-only quality required')
    plan=copy.deepcopy(public_plan)
    for candidate in plan['candidates']:
        override=full['candidate_overrides'][candidate['id']]
        for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):candidate[key]=override[key]
    quality=quality_doc.get('loss_by_family_task',quality_doc)
    execution=[*public_plan.get('execution_sources',[]),*[public_plan[k] for k in
        ('runner_source','context','profile_plan','catalog') if k in public_plan]]
    if public_plan.get('prephase_memory_evidence'):execution.append(public_plan['prephase_memory_evidence']['source'])
    refs=[receipt(p) for p in (protocol_path,model_path,resource_manifest_path,quality_path,__file__,
        validation.__file__,h600_resources_v1.__file__,core.__file__,core.original_rho.__code__.co_filename)]
    refs.extend([protocol['target_plan'],full['resources'],*protocol['source_receipts'],*resource_manifest['source_receipts'],
        *core._sources(model),*core._sources(resources),*quality_doc.get('source_receipts',[]),*execution,
        *predictor_sources()])
    for ref in refs:checked(ref)
    refs=list({(ref['path'],ref['sha256']):ref for ref in refs}.values())
    catalog=read(checked(public_plan['catalog']))
    plain=core.search(plan,model,resources,quality,catalog=catalog)
    bounded=core.search(plan,model,resources,quality,bounded=True,catalog=catalog)
    for ref in refs:checked(ref)
    # The overlay belongs only to this model evaluation, never the public plan.
    if read(checked(protocol['target_plan']))!=public_plan:raise ValueError('Public plan changed during search')
    same=(plain['all_tie_ids']==bounded['all_tie_ids']
        and plain['maximum_model_comparison_U']==bounded['maximum_model_comparison_U'])
    unknown=bool(plain['unknown_ids'] or bounded['unknown_ids'])
    feasible=plain['maximum_model_comparison_U'] is not None and bool(plain['all_tie_ids'])
    status='UNKNOWN' if unknown else 'FAIL' if not same else 'PASS' if feasible else 'NO_MODEL_FEASIBLE_CANDIDATE'
    pruned=len(bounded['bound_pruned'])
    value=dict(schema='gsdeploy.H600_selected_numerical_v2_same_domain_DFS_check.v1',
        estimator_selection=protocol['estimator_selection'],forecast_numerics='selected_completion_event_v2',status=status,
        validation_protocol=receipt(protocol_path),public_plan=protocol['target_plan'],model=receipt(model_path),
        resource_manifest=receipt(resource_manifest_path),model_verification=model_verification,
        candidate_resource_overlays={c['id']:{k:c[k] for k in
            ('predicted_startup_joules','predicted_prephase_peak_mib')} for c in plan['candidates']},
        candidate_count=len(plan['candidates']),request_count=len(plan['trace']),cohort_sha256=core.cohort_sha(plan),
        candidate_ids=[c['id'] for c in plan['candidates']],
        same_search_output=same,same_optimum_and_all_ties=same if not unknown and feasible else None,
        all_candidates_known=not plain['unknown_ids'],no_model_feasible_candidate=not unknown and not feasible,
        maximum_model_comparison_U=plain['maximum_model_comparison_U'],all_tie_ids=plain['all_tie_ids'],
        original_rho_admission_unknown_ids=plain['original_rho_admission_unknown_ids'],
        original_gate_constrained_optimality_certified=False,physical_optimality_certified=False,
        physical_eligibility=None,final_gain_eligible=False,GPU_started=False,
        stored_prediction_or_outcome_used=False,public_plan_modified=False,
        model_constraints=dict(window_seconds=plan['window_seconds'],drain_seconds=plan['drain_seconds'],
            memory_budget_mib=plan['memory_budget_mib'],memory_safe_factor=plan['memory_safe_factor']),
        domain='Only the four reserved static shared routes, fixed resident aliases/executor, same complete public arrival/input trace; model memory and bounded drain constraints',
        unsearched=['other deployments','replica counts','unlisted routes','resource allocations',
            'request admission executors','joint batch/cache executors','adaptive policies'],
        rho_scope='Original catalog planned-rate resolver recorded separately, including UNKNOWN; finite forecast load proxy never grants original-rho admission',
        objective_scope='Six-term U with H600 full-training startup and sampled owned peak overlays; empirical resource estimates grant no physical bound',
        pruning_count=pruned,forecast_calls_saved=plain['fresh_forecast_calls']-bounded['fresh_forecast_calls'],
        search_acceleration_claim=False,
        pruning_scope='Prune counts and timings are descriptive; when pruning_count is zero there is no forecast-call reduction, and this check makes no performance claim',
        upper_bound_proof='Frozen search upper bound uses nonnegative six-term costs; Q is exact training proxy or zero lower bound, M is at least safe*declared empirical peak/budget; strict bound<incumbent-1e-12 preserves every tolerance tie',
        source_receipts=refs,plain={k:v for k,v in plain.items() if k!='records'},
        bounded={k:v for k,v in bounded.items() if k!='records'})
    output.mkdir(parents=True,exist_ok=False)
    for name,document in (('summary',value),('plain-dfs',plain),('bounded',bounded)):save(output/(name+'.json'),document)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','model','resource-manifest','quality','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();result=run(args.protocol,args.model,args.resource_manifest,args.quality,args.output)
    print(json.dumps({key:result[key] for key in ('status','candidate_count','maximum_model_comparison_U','all_tie_ids','pruning_count')}))
