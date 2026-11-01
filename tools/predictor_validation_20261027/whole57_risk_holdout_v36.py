"""Independent risk holdout collector: evidence-only AST wrapper over frozen v23.

Preparation and audit are CPU-only. Execute requires an explicit CLI flag and
the original owned-process lock/watchdog/budget. No training consumer accepts
this validation schema. No GPU operation occurs while importing this module.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import uuid

from . import whole57_joint_training_v23 as original
from .horizon_recovery_domain_v31 import leaf_differences
from .native_collect import context_identity,receipt,save

PLAN_SCHEMA='gsdeploy.whole57_independent_risk_holdout_plan.v36'
RAW_SCHEMA='gsdeploy.whole57_independent_risk_holdout_raw.v36'
SCHEMA='gsdeploy.whole57_independent_risk_holdout.v36'
FUNCTIONS=('clock_anchor','register_and_offer','cohort','outer_command','execute')
# Exhaustive constant substitutions: no arithmetic, ordering, waits, lifecycle,
# monitor/worker/runtime/drain/priming call or branch is modified.
STRING_MAP={
    'training':'risk_validation',
    'TRAINING_ONLY_NOT_VALIDATION':'INDEPENDENT_RISK_HOLDOUT_NOT_FOR_TRAINING',
    'TRAINING_OBSERVATIONS_NOT_INDEPENDENT_VALIDATION':'INDEPENDENT_RISK_HOLDOUT_NO_TRAINING_CONSUMER',
    'predeclared_training_recipe':'predeclared_same_recipe_risk_holdout',
    'tools.predictor_validation_20261027.whole57_joint_training_v23':'tools.predictor_validation_20261027.whole57_risk_holdout_v36',
    'gsdeploy.whole57_joint_training_execution.v23':'gsdeploy.whole57_independent_risk_holdout_execution.v36',
    'COLLECTED_TRAINING_REQUIRES_INDEPENDENT_REVIEW':'COLLECTED_RISK_HOLDOUT_REQUIRES_INDEPENDENT_REVIEW',
}


class EvidenceOnlyTransform(ast.NodeTransformer):
    def __init__(self):self.changes=[]
    def visit_Constant(self,node):
        if type(node.value) is str and node.value in STRING_MAP:
            self.changes.append(dict(kind='string_constant',line=node.lineno,original=node.value,replacement=STRING_MAP[node.value]))
            return ast.copy_location(ast.Constant(STRING_MAP[node.value]),node)
        return node
    def visit_keyword(self,node):
        if node.arg=='training_only' and isinstance(node.value,ast.Constant) and node.value.value is True:
            self.changes.append(dict(kind='evidence_keyword',line=node.value.lineno,original='training_only=True',replacement='training_only=False'))
            node.value=ast.copy_location(ast.Constant(False),node.value)
            return node
        return self.generic_visit(node)


def build_execution_namespace():
    namespace=dict(vars(original));namespace.update(PLAN_SCHEMA=PLAN_SCHEMA,RAW_SCHEMA=RAW_SCHEMA,audit=audit)
    comparisons=[]
    for name in FUNCTIONS:
        source=inspect.getsource(getattr(original,name));tree=ast.parse(source)
        transformer=EvidenceOnlyTransform();transformed=transformer.visit(copy.deepcopy(tree));ast.fix_missing_locations(transformed)
        exec(compile(transformed,str(Path(__file__).resolve())+'::'+name,'exec'),namespace)
        comparisons.append(dict(function=name,original_body=source,derived_body=ast.unparse(transformed),
            original_AST_SHA256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
            derived_AST_SHA256=hashlib.sha256(ast.dump(transformed,include_attributes=False).encode()).hexdigest(),
            exact_original_AST_with_only_declared_evidence_changes=True,declared_changes=transformer.changes))
    return namespace,comparisons


def execution_projection(context):
    result=copy.deepcopy(context)
    for key in ('execution_nonce','whole57_training_route_id','training_only'):result.pop(key,None)
    result['collection_schema']=original.RAW_SCHEMA
    # This exact single extra wrapper file is evidence-only and AST-proven.
    extra=[s for s in result['execution_sources'] if s['original_path']==str(Path(__file__).resolve())]
    if len(extra)>1:raise ValueError('Duplicate holdout wrapper source')
    if extra:
        result['execution_sources'].remove(extra[0]);result['sources'].remove(extra[0]['frozen'])
    return result


def projected_plan(plan):
    trace=[{key:value for key,value in row.items() if key!='request_id'} for row in plan['trace']]
    route_sequence=[plan['routes'][row['request_id']] for row in plan['trace']]
    return dict(trace=trace,route_sequence=route_sequence,recipe=plan['recipe'],inputs=plan['inputs'],
        resource_configuration=plan['resource_configuration'],routing_weights=plan['routing_weights'],weights=plan['weights'])


def audit(run):
    run=Path(run).resolve();plan=original.read(run/'plan.json');context=original.read(run/'context.json')
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('split')!='risk_validation' or plan.get('training_only') is not False:
        raise ValueError('Explicit independent risk validation schema required')
    if plan.get('plan_sha256')!=original.digest({k:v for k,v in plan.items() if k!='plan_sha256'}):raise ValueError('Holdout plan mutated')
    source_plan=original.checked(plan['source_training_plan']);source_context=original.checked(plan['source_training_context'])
    original.audit(Path(plan['source_training_plan']['path']).parent)
    if not re.fullmatch(r'whole57-risk-holdout-[A-Za-z0-9_-]+',plan['training_id']):raise ValueError('Independent holdout identity required')
    if len(plan['trace'])!=57 or len({r['request_id'] for r in plan['trace']})!=57 or any(not r['request_id'].startswith(plan['training_id']+':req') for r in plan['trace']):
        raise ValueError('Complete fresh holdout IDs required')
    if plan['route_id']!=source_plan['route_id'] or plan['domain']!=original.NATURAL or plan['recipe']!=original.RECIPE:
        raise ValueError('Exact same natural four-route recipe required')
    if projected_plan(plan)!=projected_plan(source_plan):raise ValueError('Holdout execution schedule/input/route/resource/drain changed')
    if context_identity(context)!=plan['execution_context_sha256'] or context['collection_schema']!=RAW_SCHEMA or context.get('training_only') is not False:
        raise ValueError('Exact original holdout context and validation evidence schema required')
    if execution_projection(context)!=execution_projection(source_context):raise ValueError('Unexpected physical/execution context difference')
    if context['execution_nonce']==source_context['execution_nonce']:raise ValueError('Fresh exclusive nonce required')
    _,comparisons=build_execution_namespace()
    if original.read(run/'execution-body-equivalence.json')!=dict(schema=SCHEMA+'.execution_equivalence',original_source=receipt(original.__file__),wrapper_source=receipt(__file__),functions=comparisons,string_map=STRING_MAP):
        raise ValueError('Evidence-only execution body equivalence changed')
    original.source_audit(run);original.existing.priming_items(plan);original.existing.build_registry(context)
    return dict(status='CPU_INDEPENDENT_RISK_HOLDOUT_PREPARATION_AUDIT_PASS',requests=57,priming_requests=12,
        route_id=plan['route_id'],split='risk_validation',training_only=False,GPU_started=False,
        execution_projection_equal_after_explicit_evidence_mapping=True,
        actual_READY_qualification='PENDING_RUNTIME_READINESS_ONLY',risk_predictions_sealed=False)


def prepare(prepared_training_root,risk_domain_path,output):
    source_root=Path(prepared_training_root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    from . import horizon_recovery_risk_v31 as risk
    domain=original.read(risk_domain_path);risk.check_domain(domain,require_actual=True)
    refs=[receipt(__file__),receipt(original.__file__),receipt(risk_domain_path)]
    _,comparisons=build_execution_namespace();equivalence=dict(schema=SCHEMA+'.execution_equivalence',original_source=refs[1],wrapper_source=refs[0],functions=comparisons,string_map=STRING_MAP)
    output.mkdir(parents=True);source_snapshot=output/'source-snapshot'/Path(__file__).name
    source_snapshot.parent.mkdir();shutil.copy2(__file__,source_snapshot)
    runs=[];target_members=[]
    for route in ('balanced','skewed','spill_low','spill_high'):
        source=source_root/route;original.audit(source)
        source_plan=original.read(source/'plan.json');source_context=original.read(source/'context.json')
        if source_plan['execution_context_sha256'] not in domain['member_execution_context_sha256']:raise ValueError('Training source is outside exact frozen risk domain')
        run=output/route;run.mkdir();plan=copy.deepcopy(source_plan);context=copy.deepcopy(source_context)
        context.update(execution_nonce=uuid.uuid4().hex,collection_schema=RAW_SCHEMA,training_only=False)
        context['execution_sources'].append(dict(original_path=str(Path(__file__).resolve()),frozen=receipt(source_snapshot)))
        context['sources'].append(receipt(source_snapshot))
        holdout_id='whole57-risk-holdout-r01-'+route
        mapping={row['request_id']:holdout_id+':req'+str(index).zfill(3) for index,row in enumerate(plan['trace'])}
        for row in plan['trace']:row['request_id']=mapping[row['request_id']]
        plan['routes']={mapping[key]:value for key,value in plan['routes'].items()}
        plan.update(schema=PLAN_SCHEMA,split='risk_validation',training_only=False,training_id=holdout_id,
            trace_sha256=original.digest(plan['trace']),execution_context_sha256=context_identity(context),
            source_training_plan=receipt(source/'plan.json'),source_training_context=receipt(source/'context.json'),
            validation_scope='FRESH_EXECUTION_SAME_RECIPE_CONTENT_INDEPENDENT_OBSERVER_RISK_ONLY',
            all_training_consumers_forbidden=True,routing_gain_claimed=False,
            input_generalization_claimed=False,model_refit_allowed=False,
            predictor_risk_domain=receipt(risk_domain_path),source_identity_limits=[
                'Explicit new risk validation evidence schema; exact evidence-only AST wrapper over frozen v23 execution.',
                'Fresh request IDs/nonce; same original inputs, arrivals, routes, 30s window, 120s drain and 57+12 priming.',
                'Actual READY equality must qualify before scoring; prepared projection equality is insufficient.',
                'Expanded model bindings must preserve all fitted numeric parameters; no target outcomes enter fit.'])
        plan['plan_sha256']=original.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        save(run/'context.json',context);save(run/'plan.json',plan);save(run/'execution-body-equivalence.json',equivalence)
        check=audit(run);runs.append(dict(run=str(run),plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),audit=check))
        target_members.append(dict(route_id=route,original_execution_context_sha256=plan['execution_context_sha256'],
            context=receipt(run/'context.json'),plan=receipt(run/'plan.json'),training_source_context=receipt(source/'context.json'),
            complete_context_differences=leaf_differences(source_context,context),
            explicit_evidence_mapping_projection_equal=execution_projection(source_context)==execution_projection(context)))
        refs.extend([receipt(source/'plan.json'),receipt(source/'context.json')])
    expansion=dict(schema=SCHEMA+'.prepared_risk_domain_expansion',status='PREPARED_EXPLICIT_TARGET_CONTEXTS_WAITING_FOR_READINESS_ONLY_QUALIFICATION',
        training_risk_domain=receipt(risk_domain_path),training_risk_domain_sha256=domain['risk_domain_sha256'],target_members=target_members,
        evidence_equivalence=equivalence,original_context_SHA_rewritten=False,fit_numeric_parameters_changed=False,
        target_service_queue_labels_read=False,actual_READY_qualified=False,measured_accuracy_established=False,
        current_v31_v32_models_automatically_accept_targets=False,requires_separate_expanded_provider=True,
        exact_execution_domain_transferred=False)
    save(output/'prepared-risk-domain-expansion.json',expansion)
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Frozen preparation sources changed')
    report=dict(schema=SCHEMA+'.campaign',status='CPU_PREPARED_NOT_EXECUTED_REQUIRES_ROOT_REVIEW',runs=runs,
        prepared_expansion=receipt(output/'prepared-risk-domain-expansion.json'),source_receipts_before=refs,source_receipts_after=after,
        GPU_started=False,training_only=False,independent_capture_count=4,requests_per_run=57,priming_per_run=12,
        no_retry=True,calibrated_probability=False,routing_gain_claimed=False,six_term_utility_changed=False,rho_changed=False)
    save(output/'campaign.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--training-root',required=True);prep.add_argument('--risk-domain',required=True);prep.add_argument('--output',required=True)
    for name in ('audit','cohort','execute'):
        cmd=sub.add_parser(name);cmd.add_argument('--run',required=True)
        if name=='execute':cmd.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.training_root,args.risk_domain,args.output)
    elif args.command=='audit':result=audit(args.run)
    else:
        namespace,_=build_execution_namespace();result=namespace[args.command](args.run)
    print(json.dumps(dict(status=result['status']),indent=2))
    return 1 if result.get('status') in ('failed','FAILED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
