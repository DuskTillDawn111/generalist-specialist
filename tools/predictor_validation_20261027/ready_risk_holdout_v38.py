"""Fresh risk holdout: qualify READY and freeze rebound models before priming.

CPU preparation never starts GPU. The frozen v36 evidence-only AST wrapper is
retained; the sole new execution hook runs after native start, before priming.
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import shutil
import uuid

from gsdeploy.resources import identity
from . import horizon_recovery_risk_v31 as risk
from . import whole57_risk_holdout_v36 as base
from . import whole57_joint_training_v23 as original
from .horizon_recovery_domain_v31 import leaf_differences
from .native_collect import context_identity,receipt,save

SCHEMA='gsdeploy.ready_qualified_independent_risk_holdout.v38'
PLAN_SCHEMA=SCHEMA+'.plan';RAW_SCHEMA=SCHEMA+'.raw'


def static_source_closure(entry=__file__):
    project=Path(__file__).resolve().parents[2]
    def resolve(module):
        path=project.joinpath(*module.split('.'))
        for candidate in (path.with_suffix('.py'),path/'__init__.py'):
            if candidate.is_file():return candidate
        return None
    pending=[Path(entry).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);package=list(path.relative_to(project).with_suffix('').parts)[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                if node.module:prefix+=node.module.split('.')
                modules=['.'.join(prefix),*['.'.join([*prefix,alias.name]) for alias in node.names]]
            for module in modules:
                if module.split('.')[0] not in {'gsdeploy','tools'}:continue
                parts=module.split('.')
                for length in range(1,len(parts)+1):
                    candidate=resolve('.'.join(parts[:length]))
                    if candidate is not None and candidate not in seen:pending.append(candidate)
    return [receipt(path) for path in sorted(seen)]


def numeric_model(model):
    return {key:value for key,value in model.items() if key not in {'risk_domain_sha256','model_sha256'}}


def rebind_model(model,source_domain,expanded_domain):
    risk.check_domain(source_domain,require_actual=True);risk.check_domain(expanded_domain,require_actual=True)
    if model['risk_domain_sha256']!=source_domain['risk_domain_sha256'] or model['model_sha256']!=identity({k:v for k,v in model.items() if k!='model_sha256'}):
        raise ValueError('Original frozen training model/domain required')
    if expanded_domain['independent_target_extension']['source_risk_domain_sha256']!=source_domain['risk_domain_sha256']:
        raise ValueError('Explicit predeclared source domain extension required')
    rebound=copy.deepcopy(model);rebound['risk_domain_sha256']=expanded_domain['risk_domain_sha256']
    rebound['model_sha256']=identity({k:v for k,v in rebound.items() if k!='model_sha256'})
    if numeric_model(rebound)!=numeric_model(model):raise ValueError('Numeric fit/provenance changed during metadata-only rebind')
    return rebound


def projection(context,declared_extra_sources):
    value=copy.deepcopy(context)
    for key in ('execution_nonce','whole57_training_route_id','training_only'):value.pop(key,None)
    value['collection_schema']=original.RAW_SCHEMA
    for ref in declared_extra_sources:
        matches=[row for row in value['execution_sources'] if row['original_path']==ref['original_path']]
        if not matches:continue
        if len(matches)!=1 or matches[0]!=ref:raise ValueError('Unexpected wrapper source substitution')
        value['execution_sources'].remove(ref);value['sources'].remove(ref['frozen'])
    return value


def extension_ref(plan):
    path=Path(plan['prepared_domain_extension_path'])
    ref=original.read(path.parent/'prepared-domain-extension-receipt.json')
    if ref!=receipt(path):raise ValueError('Predeclared domain extension changed')
    return ref


def audit(run):
    run=Path(run).resolve();plan=original.read(run/'plan.json');context=original.read(run/'context.json')
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('split')!='risk_validation' or plan.get('training_only') is not False:
        raise ValueError('Honest independent READY-qualified validation schema required')
    if plan['plan_sha256']!=original.digest({k:v for k,v in plan.items() if k!='plan_sha256'}):raise ValueError('Holdout plan changed')
    source_plan=original.checked(plan['source_training_plan']);source_context=original.checked(plan['source_training_context'])
    original.audit(Path(plan['source_training_plan']['path']).parent)
    if base.projected_plan(plan)!=base.projected_plan(source_plan) or plan['domain']!=original.NATURAL or plan['route_id']!=source_plan['route_id']:
        raise ValueError('Original arrival/input/route/window/drain/priming execution changed')
    expected=[plan['training_id']+':req'+str(i).zfill(3) for i in range(57)]
    if not plan['training_id'].startswith('whole57-risk-ready-holdout-') or [r['request_id'] for r in plan['trace']]!=expected:
        raise ValueError('Exact independent fresh cohort IDs required')
    if context_identity(context)!=plan['execution_context_sha256'] or context['collection_schema']!=RAW_SCHEMA or context.get('training_only') is not False:
        raise ValueError('Exact READY-qualified target context required')
    if {row['original_path'] for row in plan['declared_evidence_wrapper_sources']}!={str(Path(base.__file__).resolve()),str(Path(__file__).resolve())} or len(plan['declared_evidence_wrapper_sources'])!=2:
        raise ValueError('Only exact v36/v38 evidence wrappers may extend collection sources')
    if projection(context,plan['declared_evidence_wrapper_sources'])!=projection(source_context,[]):
        raise ValueError('Real execution/profiling context differs after exact declared evidence mapping')
    if context['execution_nonce']==source_context['execution_nonce']:raise ValueError('Fresh nonce required')
    for ref in plan['model_sources']+[plan['source_risk_domain'],extension_ref(plan)]:original.checked(ref)
    if plan['complete_ready_hook_source_closure']!=static_source_closure():raise ValueError('READY hook full local source closure changed')
    original.source_audit(run);original.existing.priming_items(plan);original.existing.build_registry(context)
    namespace,proof=base.build_execution_namespace()
    if original.read(run/'execution-body-equivalence.json')!=proof:raise ValueError('Frozen v36 execution AST proof changed')
    if original.read(run/'startup-ready-hook-proof.json')!=dict(wrapper_source=receipt(__file__),hook_body=inspect.getsource(execution_namespace),
            placement='After native DeploymentRuntime.start and owned PID check; before native cohort start returns to first priming enqueue'):
        raise ValueError('Explicit startup-only READY hook body changed')
    return dict(status='CPU_READY_QUALIFIED_RISK_HOLDOUT_AUDIT_PASS',GPU_started=False,requests=57,priming_requests=12,
        actual_READY_qualification='PENDING_RUNTIME_READINESS_ONLY',target_outcomes_read=False)


def qualify_ready(run,ready):
    """Only actual READY payload and frozen CPU preparation, no service records."""
    run=Path(run).resolve();audit(run);plan=original.read(run/'plan.json')
    root=run/'readiness-only'
    if root.exists():raise FileExistsError('READY stage is single-use')
    root.mkdir();save(root/'ready.json',dict(schema=SCHEMA+'.ready_only',ready=ready,
        execution_context_sha256=plan['execution_context_sha256'],before_any_priming=True,service_queue_or_outcomes_included=False))
    source_domain=original.checked(plan['source_risk_domain']);risk.check_domain(source_domain,require_actual=True)
    projections=[item['projection']['ready'] for item in source_domain['actual_members'].values()]
    if not projections or any(p!=projections[0] for p in projections):raise ValueError('Original training READY projections differ')
    actual={alias:{key:copy.deepcopy(value) for key,value in worker.items() if key!='pid'} for alias,worker in sorted(ready.items())}
    differences=leaf_differences(projections[0],actual)
    if differences:
        save(root/'qualification.json',dict(status='UNKNOWN_ACTUAL_READY_DIFFERENCE_STOP_BEFORE_PRIMING',differences=differences,ready=receipt(root/'ready.json')))
        raise ValueError('Independent actual READY differs; no priming or measured requests permitted')
    extension=original.checked(extension_ref(plan))
    member=next(row for row in extension['target_members'] if row['execution_context_sha256']==plan['execution_context_sha256'])
    if member['plan']!=receipt(run/'plan.json') or member['context']!=receipt(run/'context.json'):raise ValueError('Target differs from predeclared domain member')
    expanded=copy.deepcopy(source_domain);expanded.pop('risk_domain_sha256')
    expanded['member_execution_context_sha256'].append(plan['execution_context_sha256'])
    expanded['independent_target_extension']=dict(source_risk_domain_sha256=source_domain['risk_domain_sha256'],
        prepared_extension=extension_ref(plan),target_member=member,readiness_only=receipt(root/'ready.json'),
        READY_projection_exact_equal=True,before_priming=True,target_service_queue_or_labels_read=False,
        original_context_SHA_rewritten=False,source_collector_AST_preserved=True)
    expanded['scope']='same_recipe_content_fresh_execution_reader_visible_risk_validation_only'
    expanded['risk_domain_sha256']=identity(expanded);save(root/'expanded-risk-domain.json',expanded)
    models=[]
    for ref in plan['model_sources']:
        model=original.checked(ref);bound=rebind_model(model,source_domain,expanded)
        path=root/('horizon-'+str(model['horizon_ns'])+'.json');save(path,bound)
        models.append(dict(original=ref,rebound=receipt(path),numeric_model_exact_equal=numeric_model(model)==numeric_model(bound)))
    report=dict(schema=SCHEMA+'.readiness_qualification',status='ACTUAL_READY_EXACT_PASS_MODELS_BOUND_BEFORE_PRIMING',
        ready=receipt(root/'ready.json'),source_domain=plan['source_risk_domain'],expanded_domain=receipt(root/'expanded-risk-domain.json'),
        model_bindings=models,target_service_queue_or_labels_read=False,fit_called=False,
        numeric_model_parameters_changed=False,original_context_SHA_rewritten=False)
    save(root/'qualification.json',report);return report


def execution_namespace(run):
    namespace,proof=base.build_execution_namespace()
    class QualifiedRuntime(original.DeploymentRuntime):
        def start(self):
            result=super().start()
            for alias,worker in self.workers.items():
                pid=self.ready[alias].get('pid')
                if type(pid) is not int or not original.existing.descendant_or_self(pid,worker.process.pid):
                    raise RuntimeError('Unowned READY PID before qualification: '+alias)
            qualify_ready(run,copy.deepcopy(self.ready))
            return result
    def outer_command(run,context,domain):
        if domain!=original.NATURAL:raise ValueError('Natural-only independent risk validation')
        return [context['controller_python'],'-m','tools.predictor_validation_20261027.ready_risk_holdout_v38','cohort','--run',str(run)]
    namespace.update(audit=audit,PLAN_SCHEMA=PLAN_SCHEMA,RAW_SCHEMA=RAW_SCHEMA,DeploymentRuntime=QualifiedRuntime,outer_command=outer_command)
    return namespace,proof


def prepare(training_root,source_domain_path,final_model_paths,output):
    training_root=Path(training_root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    domain=original.read(source_domain_path);risk.check_domain(domain,require_actual=True)
    models=[original.read(path) for path in final_model_paths]
    if len(models)!=3 or {model['horizon_ns'] for model in models}!=set(risk.HORIZONS_NS):raise ValueError('Three final frozen training models required')
    for model in models:
        if model['risk_domain_sha256']!=domain['risk_domain_sha256'] or len(model['provenance']['run_ids'])!=4:
            raise ValueError('Only four-training-capture final models may validate fresh runs')
    output.mkdir(parents=True);extra=[];closure=static_source_closure();snapshots=[]
    project=Path(__file__).resolve().parents[2]
    for ref in closure:
        target=output/'ready-hook-source-snapshot'/Path(ref['path']).relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ref['path'],target)
        snapshots.append(dict(original=ref,snapshot=receipt(target)))
    for index,path in enumerate([Path(base.__file__).resolve(),Path(__file__).resolve()]):
        target=output/'source-snapshot'/str(index)/path.name;target.parent.mkdir(parents=True)
        shutil.copy2(path,target);extra.append(dict(original_path=str(path),frozen=receipt(target)))
    _,proof=base.build_execution_namespace();runs=[];members=[]
    extension_path=output/'prepared-domain-extension.json'
    # Plans bind the extension by path; its SHA is sealed in a separate receipt
    # file to avoid circular plan↔extension hashes. Audit checks that receipt.
    for route in ('balanced','skewed','spill_low','spill_high'):
        source=training_root/route;original.audit(source)
        source_plan=original.read(source/'plan.json');source_context=original.read(source/'context.json')
        run=output/route;run.mkdir();plan=copy.deepcopy(source_plan);context=copy.deepcopy(source_context)
        context.update(execution_nonce=uuid.uuid4().hex,collection_schema=RAW_SCHEMA,training_only=False)
        context['execution_sources'].extend(extra);context['sources'].extend(row['frozen'] for row in extra)
        holdout_id='whole57-risk-ready-holdout-r01-'+route
        mapping={row['request_id']:holdout_id+':req'+str(i).zfill(3) for i,row in enumerate(plan['trace'])}
        for row in plan['trace']:row['request_id']=mapping[row['request_id']]
        plan['routes']={mapping[key]:value for key,value in plan['routes'].items()}
        plan.update(schema=PLAN_SCHEMA,split='risk_validation',training_only=False,training_id=holdout_id,
            execution_context_sha256=context_identity(context),trace_sha256=original.digest(plan['trace']),
            source_training_plan=receipt(source/'plan.json'),source_training_context=receipt(source/'context.json'),
            source_risk_domain=receipt(source_domain_path),model_sources=[receipt(path) for path in final_model_paths],
            complete_ready_hook_source_closure=closure,
            prepared_domain_extension_path=str(extension_path),declared_evidence_wrapper_sources=extra,
            actual_ready_hook='After native runtime.start, before first priming enqueue; fail closed on any READY difference.',
            no_training_consumer=True,input_generalization_claimed=False,routing_gain_claimed=False,
            model_refit_allowed=False,validation_scope='FRESH_EXECUTION_SAME_RECIPE_CONTENT_OBSERVER_RISK_ONLY')
        plan['plan_sha256']=original.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        save(run/'context.json',context);save(run/'plan.json',plan);save(run/'execution-body-equivalence.json',proof)
        save(run/'startup-ready-hook-proof.json',dict(wrapper_source=receipt(__file__),hook_body=inspect.getsource(execution_namespace),
            placement='After native DeploymentRuntime.start and owned PID check; before native cohort start returns to first priming enqueue'))
        members.append(dict(route_id=route,execution_context_sha256=plan['execution_context_sha256'],plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
            source_context=receipt(source/'context.json'),complete_context_differences=leaf_differences(source_context,context),
            exact_projection_equal_after_declared_evidence_mapping=projection(context,extra)==projection(source_context,[])))
        runs.append(dict(run=str(run),plan=receipt(run/'plan.json'),context=receipt(run/'context.json')))
    extension=dict(schema=SCHEMA+'.prepared_domain_extension',status='PREDECLARED_TARGETS_WAITING_FOR_ACTUAL_READY_ONLY',
        source_domain=receipt(source_domain_path),model_sources=[receipt(path) for path in final_model_paths],target_members=members,
        allowed_context_evidence_differences=['execution_nonce','training_only','collection_schema','exact_declared_v36_v38_wrapper_sources'],
        execution_hook='Native start followed by READY-only qualification before priming; fixed runtime execution body otherwise unchanged.',
        model_refit=False,numeric_fit_parameters_changed=False,actual_READY_qualified=False,GPU_started=False)
    save(extension_path,extension);save(output/'prepared-domain-extension-receipt.json',receipt(extension_path))
    for item in runs:item['audit']=audit(item['run'])
    report=dict(schema=SCHEMA+'.campaign',status='CPU_PREPARED_READY_GATE_NOT_EXECUTED',runs=runs,
        prepared_extension=receipt(extension_path),request_cohort=57,priming_cohort=12,window_seconds=30,drain_seconds=120,
        complete_source_snapshots=snapshots,
        GPU_started=False,training_only=False,model_refit=False,requires_root_review_before_GPU=True)
    save(output/'campaign.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--training-root',required=True);prep.add_argument('--risk-domain',required=True)
    prep.add_argument('--model',action='append',required=True);prep.add_argument('--output',required=True)
    for name in ('audit','cohort','execute'):
        cmd=sub.add_parser(name);cmd.add_argument('--run',required=True)
        if name=='execute':cmd.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.training_root,args.risk_domain,args.model,args.output)
    elif args.command=='audit':result=audit(args.run)
    else:namespace,_=execution_namespace(args.run);result=namespace[args.command](args.run)
    print(json.dumps(dict(status=result['status']),indent=2));return 1 if result.get('status') in ('failed','FAILED_NO_RETRY') else 0


if __name__=='__main__':raise SystemExit(main())
